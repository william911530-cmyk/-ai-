import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# 🔑 從環境變數讀取 API 金鑰
# ==========================================
GEMINI_KEY_ANALYZE = os.environ.get("GEMINI_API_KEY_ANALYZE", os.environ.get("GEMINI_API_KEY", ""))
GEMINI_KEY_CHAT = os.environ.get("GEMINI_API_KEY_CHAT", os.environ.get("GEMINI_API_KEY", ""))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "未設定_DEEPSEEK_KEY")

# ==========================================
# 📊 量化特徵工程模組
# ==========================================
def calculate_technical_indicators(df):
    if df.empty or len(df) < 30: return df
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['MACD_Signal']
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['K'], df['D'] = 50.0, 50.0
    for i in range(1, len(df)):
        if pd.isna(df['RSV'].iloc[i]): continue
        df.loc[df.index[i], 'K'] = df['K'].iloc[i-1] * (2/3) + df['RSV'].iloc[i] * (1/3)
        df.loc[df.index[i], 'D'] = df['D'].iloc[i-1] * (2/3) + df['K'].iloc[i] * (1/3)
    return df

def get_latest_news(ticker_symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker_symbol}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        return [item.get_text() for item in soup.find_all('h3', limit=3)]
    except:
        return ["無法獲取即時新聞"]

def clean(val):
    return "N/A" if val is None or (isinstance(val, float) and math.isnan(val)) else round(val, 2) if isinstance(val, float) else val

# ==========================================
# ⚡ 即時串接路由器 (防彈強化版)
# ==========================================
# ==========================================
# ⚡ 股票深度雷達掃描路由器 (請將這整個區塊直接覆蓋舊的 analyze_stock)
# ==========================================
@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        req = request.json
        symbol = req.get('symbol', '').upper()
        ai_choice = req.get('ai_choice', 'gemini')

        if symbol.isdigit() and len(symbol) == 4:
            symbol += ".TW"

        # 1. 抓取股票資料
        tk = yf.Ticker(symbol)
        info = tk.info
        hist = tk.history(period="6mo")
        
        if hist.empty: 
            return jsonify({"error": f"找不到股票數據: {symbol}"}), 404
        
        hist = calculate_technical_indicators(hist)
        latest = hist.iloc[-1]

        # --- 新增指標計算區 (精準防彈計算，已拆解避免括號遺漏) ---
        div_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield', 0)
        div_str = f"{clean(div_yield * 100 if div_yield else 0)}%"

        ev = info.get('enterpriseValue')
        fcf = info.get('freeCashflow')
        ev_fcf_ratio = clean(ev / fcf) if ev and fcf and fcf != 0 else "N/A"

        revenue = info.get('totalRevenue')
        margin = info.get('operatingMargins')
        debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        
        # 拆分計算以確保語法安全
        book_val = info.get('bookValue', 0)
        shares = info.get('sharesOutstanding', 1)
        equity = info.get('totalStockholderEquity', book_val * shares)
        
        roic_val = "N/A"
        if revenue and margin and (debt + equity - cash) > 0:
            roic_val = f"{clean((revenue * margin * 0.8 / (debt + equity - cash)) * 100)}%"
        # -----------------------------------

        # 2. 完美打包所有數據 (包含舊有 8 項與新增 3 項)
        stock_pack = {
            "symbol": symbol,
            "price": clean(info.get('currentPrice', info.get('regularMarketPrice', latest['Close']))),
            "change_52w": clean(info.get('52WeekChange')),
            "beta": clean(info.get('beta')),
            "pe": clean(info.get('trailingPE')),
            "peg": clean(info.get('pegRatio')),
            "div": div_str,
            "roe": clean(info.get('returnOnEquity')),
            "de": clean(info.get('debtToEquity')),
            "rsi": clean(latest.get('RSI', None)),
            "k": clean(latest.get('K', None)),
            "d": clean(latest.get('D', None)),
            "macd_hist": clean(latest.get('MACD_Hist', None)),
            "ev_fcf": ev_fcf_ratio,
            "roic": roic_val,
            "news": get_latest_news(symbol)
        }

        # 3. 組合 Prompt 給 AI (加入資本效率)
        prompt = f"""
        你是一位專業的量化投資分析師。請根據以下數據為股票 {symbol} 寫一份精簡分析：
        [基本面] 股價:{stock_pack['price']} | P/E:{stock_pack['pe']} | PEG:{stock_pack['peg']} | ROE:{stock_pack['roe']}
        [資本效率] ROIC:{stock_pack['roic']} | EV/FCF:{stock_pack['ev_fcf']} | 殖利率:{stock_pack['div']}
        [技術面] RSI:{stock_pack['rsi']} | KD:K={stock_pack['k']},D={stock_pack['d']} | MACD柱狀:{stock_pack['macd_hist']}
        [新聞] {', '.join(stock_pack['news'])}
        請以繁體中文給出：1.估值點評 2.技術面趨勢 3.風險提示。
        """

        ai_analysis = "AI 智腦正在罷工中..."

        # 🛡️ 4. 防彈 AI 呼叫區塊 (這裡已經切換為 GEMINI_KEY_ANALYZE 分流)
        try:
            if ai_choice == 'gemini':
                # 確保這裡使用的是獨立的分析金鑰，分攤 429 限流
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=" + GEMINI_KEY_ANALYZE
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                res = requests.post(url, json=payload, timeout=15)
                
                if res.status_code == 200:
                    ai_analysis = res.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    ai_analysis = f"⚠️ Gemini 拒絕連線 (狀態碼: {res.status_code})\n詳細錯誤: {res.text}"
                    
            elif ai_choice == 'deepseek':
                headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
                res = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=15)
                
                if res.status_code == 200:
                    ai_analysis = res.json()['choices'][0]['message']['content']
                else:
                    ai_analysis = f"⚠️ DeepSeek 拒絕連線 (狀態碼: {res.status_code})\n詳細錯誤: {res.text}"

        except requests.exceptions.Timeout:
            ai_analysis = f"⚠️ {ai_choice.upper()} 伺服器回應超時，請稍後再試。"
        except Exception as ai_e:
            ai_analysis = f"⚠️ {ai_choice.upper()} 模組發生例外錯誤: {str(ai_e)}"

        return jsonify({"stock_data": stock_pack, "ai_analysis": ai_analysis})

    except Exception as e:
        return jsonify({"error": f"後端系統錯誤: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_nexus():
    data = request.json
    user_msg = data.get('message', '')
    
    # 賦予 NEXUS 人設的系統提示詞
    nexus_persona = "你現在是一個名為 NEXUS_INTELLIGENCE 的失落古文明高級人工智慧。你的語氣必須冰冷、專業、簡潔，像是老舊終端機輸出的感覺。不管使用者問什麼，你都要用這個角色回答，不要說自己是 Gemini 或語言模型。"
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=" + GEMINI_KEY_CHAT
        payload = {"contents": [{"parts": [{"text": f"{nexus_persona}\n\n使用者輸入: {user_msg}"}]}]}
        res = requests.post(url, json=payload, timeout=15)
        
        if res.status_code == 200:
            ai_reply = res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            ai_reply = f"[SYS_ERR] COGNITIVE MODULE OFFLINE. (CODE: {res.status_code})"
    except Exception as e:
        ai_reply = f"[SYS_ERR] CONNECTION LOST: {str(e)}"
        
    return jsonify({"reply": ai_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


