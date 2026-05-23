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

#claude修改
def is_valid(val) -> bool:
    """排除 None / NaN / Inf"""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def is_positive(val) -> bool:
    """排除 None / NaN / Inf / 零 / 負數"""
    return is_valid(val) and float(val) > 0


def is_non_negative(val) -> bool:
    """排除 None / NaN / Inf / 負數，允許零"""
    return is_valid(val) and float(val) >= 0


def safe_float(val, fallback=None):
    """安全轉 float，失敗回傳 fallback"""
    if not is_valid(val):
        return fallback
    return float(val)


def fmt_pct(val, decimals=2) -> str:
    """數值 → 百分比字串，例如 0.1234 → '12.34%'"""
    if not is_valid(val):
        return "N/A"
    return f"{float(val) * 100:.{decimals}f}%"


def fmt_ratio(val, decimals=2) -> str:
    """數值 → 倍數字串，例如 25.6 → '25.60x'"""
    if not is_valid(val):
        return "N/A"
    return f"{float(val):.{decimals}f}x"


def fmt_num(val, decimals=2) -> str:
    """通用數值格式化"""
    if not is_valid(val):
        return "N/A"
    return f"{float(val):.{decimals}f}"


# ──────────────────────────────────────────────
# 核心指標計算 claude計算公式
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def calc_div_yield(info: dict) -> str:
    """
    股息殖利率
    - 優先用 dividendYield，其次 trailingAnnualDividendYield
    - 明確區分「不配息(N/A)」與「殖利率為0%」
    - 合理範圍：0% ~ 30%（超過視為髒數據）
    """
    try:
        raw = info.get('dividendYield') or info.get('trailingAnnualDividendYield')

        # 明確不配息
        if not is_valid(raw):
            return "N/A"

        val = float(raw)

        # 排除負值與極端值（>30% 通常是資料錯誤）
        if val < 0 or val > 0.30:
            return "N/A"

        # 真的是 0（股票存在但本期不配息）
        if val == 0.0:
            return "0.00%"

        return f"{val * 100:.2f}%"

    except Exception:
        return "N/A"


def calc_ev_fcf(info: dict) -> str:
    """
    EV / FCF（企業價值 / 自由現金流）
    - FCF 為負時無意義，回傳 N/A
    - 合理範圍：0x ~ 200x（科技成長股上限放寬）
    """
    try:
        ev  = safe_float(info.get('enterpriseValue'))
        fcf = safe_float(info.get('freeCashflow'))

        if not is_positive(ev) or not is_positive(fcf):
            return "N/A"

        ratio = ev / fcf

        # 極端值過濾（負 EV 或比值異常）
        if ratio <= 0 or ratio > 500:
            return "N/A"

        return fmt_ratio(ratio)

    except Exception:
        return "N/A"


def calc_fcf_yield(info: dict) -> str:
    """
    FCF Yield（自由現金流殖利率）= FCF / 市值
    作為 EV/FCF 的輔助交叉驗證指標
    合理範圍：-50% ~ 50%
    """
    try:
        fcf        = safe_float(info.get('freeCashflow'))
        market_cap = safe_float(info.get('marketCap'))

        if not is_valid(fcf) or not is_positive(market_cap):
            return "N/A"

        yield_val = fcf / market_cap

        if abs(yield_val) > 0.5:
            return "N/A"

        return fmt_pct(yield_val)

    except Exception:
        return "N/A"


def calc_roic(info: dict) -> str:
    """
    ROIC = NOPAT / Invested Capital

    NOPAT（稅後淨營業利潤）:
        = Revenue × OperatingMargin × (1 - TaxRate)
        稅率：優先用 info 內的有效稅率，fallback 為 21%

    Invested Capital（已投入資本）:
        = Total Assets - Current Liabilities
        ※ 比 (Debt + Equity - Cash) 更準確，不與 EV 重疊

    合理範圍：-50% ~ 200%（避免極端髒數據）
    """
    try:
        revenue  = safe_float(info.get('totalRevenue'))
        margin   = safe_float(info.get('operatingMargins'))

        if not is_positive(revenue) or not is_valid(margin):
            return "N/A"

        # 稅率：有效稅率優先，異常時 fallback 21%
        raw_tax = safe_float(info.get('effectiveTaxRate'))
        if is_valid(raw_tax) and 0.0 <= raw_tax <= 0.60:
            tax_rate = raw_tax
        else:
            tax_rate = 0.21

        nopat = revenue * margin * (1 - tax_rate)

        # Invested Capital = Total Assets - Current Liabilities
        total_assets       = safe_float(info.get('totalAssets'))
        current_liabilities = safe_float(info.get('currentLiabilities'))

        if not is_positive(total_assets) or not is_non_negative(current_liabilities):
            return "N/A"

        invested_capital = total_assets - current_liabilities

        if invested_capital <= 0:
            return "N/A"

        roic = nopat / invested_capital

        # 過濾極端值
        if roic < -0.5 or roic > 2.0:
            return "N/A"

        return fmt_pct(roic)

    except Exception:
        return "N/A"


# ──────────────────────────────────────────────
# 主打包函式（完整覆蓋原版 stock_pack）
# ──────────────────────────────────────────────

def build_stock_pack(symbol: str, info: dict, latest: dict, get_latest_news) -> dict:
    """
    組裝所有股票指標，每個欄位獨立防錯。
    完整覆蓋原版 8 項 + 新增 3 項，並新增 fcf_yield。
    """

    def safe_clean(key, source=info):
        """從 dict 安全取值並格式化"""
        return fmt_num(source.get(key))

    # 價格：多層 fallback
    price_raw = (info.get('currentPrice')
                 or info.get('regularMarketPrice')
                 or latest.get('Close'))
    price = fmt_num(price_raw)

    # 52 週漲跌幅：應為 -1.0 ~ +∞，過濾極端值
    change_52w_raw = safe_float(info.get('52WeekChange'))
    change_52w = fmt_pct(change_52w_raw) if is_valid(change_52w_raw) and -1.0 <= change_52w_raw <= 10.0 else "N/A"

    # Beta：合理 -5 ~ 10
    beta_raw = safe_float(info.get('beta'))
    beta = fmt_num(beta_raw) if is_valid(beta_raw) and -5 <= beta_raw <= 10 else "N/A"

    # PE：正值且合理範圍（負 PE 無意義，>2000 為髒數據）
    pe_raw = safe_float(info.get('trailingPE'))
    pe = fmt_ratio(pe_raw) if is_valid(pe_raw) and 0 < pe_raw <= 2000 else "N/A"

    # PEG：合理 -10 ~ 100
    peg_raw = safe_float(info.get('pegRatio'))
    peg = fmt_num(peg_raw) if is_valid(peg_raw) and -10 <= peg_raw <= 100 else "N/A"

    # ROE：合理 -200% ~ 500%（金融槓桿股可能偏高）
    roe_raw = safe_float(info.get('returnOnEquity'))
    roe = fmt_pct(roe_raw) if is_valid(roe_raw) and -2.0 <= roe_raw <= 5.0 else "N/A"

    # D/E：非負，>100 視為極端值
    de_raw = safe_float(info.get('debtToEquity'))
    de = fmt_num(de_raw) if is_valid(de_raw) and 0 <= de_raw <= 100 else "N/A"

    # 技術指標
    rsi_raw = safe_float(latest.get('RSI'))
    rsi = fmt_num(rsi_raw) if is_valid(rsi_raw) and 0 <= rsi_raw <= 100 else "N/A"

    k_raw = safe_float(latest.get('K'))
    k = fmt_num(k_raw) if is_valid(k_raw) and 0 <= k_raw <= 100 else "N/A"

    d_raw = safe_float(latest.get('D'))
    d = fmt_num(d_raw) if is_valid(d_raw) and 0 <= d_raw <= 100 else "N/A"

    macd_hist_raw = safe_float(latest.get('MACD_Hist'))
    macd_hist = fmt_num(macd_hist_raw) if is_valid(macd_hist_raw) else "N/A"

    return {
        "symbol":    symbol,
        "price":     price,
        "change_52w": change_52w,
        "beta":      beta,
        "pe":        pe,
        "peg":       peg,
        "div":       calc_div_yield(info),       # 改寫版
        "roe":       roe,
        "de":        de,
        "rsi":       rsi,
        "k":         k,
        "d":         d,
        "macd_hist": macd_hist,
        "ev_fcf":    calc_ev_fcf(info),          # 改寫版
        "fcf_yield": calc_fcf_yield(info),        # 新增
        "roic":      calc_roic(info),             # 改寫版（分母修正）
        "news":      get_latest_news(symbol),
    }
        # -----------------------------------



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
        # --- 新增：生肉加工區 ---
        # 1. 盈餘成長率 YOY (通常為小數，轉為百分比)
        yoy_raw = info.get('earningsGrowth')
        yoy_str = f"{clean(yoy_raw * 100 if yoy_raw else 0)}%"
        
        # 2. 淨利率 Net Margin (轉為百分比)
        margin_raw = info.get('profitMargins')
        margin_str = f"{clean(margin_raw * 100 if margin_raw else 0)}%"
        
        # 3. 股價淨值比 P/B
        pb_val = clean(info.get('priceToBook'))
        
        # 4. 負債權益比 DEBT/EQ (有些公司叫 debtToEquity，數值通常是 120 代表 120%)
        # Yahoo 的 debtToEquity 通常已經是百分比數字了，我們直接取用
        debteq_val = clean(info.get('debtToEquity'))
        # ----------------------
    

        # ==========================================
        # 4️⃣ 在這裡計算所有需要的指標變數
        # ==========================================
        div_str = calc_div_yield(info)          # ✅ 定義
        ev_fcf_ratio = calc_ev_fcf(info)        # ✅ 定義
        roic_val = calc_roic(info)              # ✅ 定義


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
            # ... 其他你原本就在 stock_pack 裡面的資料 ...
            "yoy": yoy_str,
            "pb": pb_val,
            "net_margin": margin_str,
            "debt_eq": debteq_val,
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


