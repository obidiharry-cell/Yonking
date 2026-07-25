import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import datetime

st.set_page_config(page_title="YonKing", page_icon="📈", layout="wide")
st.title("📈 YonKing - Forex Analysis Dashboard")
st.caption("AI-powered price prediction, news sentiment, institutional positioning, and timeframe analysis")

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
GROQ_KEY = st.secrets["GROQ_KEY"]
ALPHA_KEY = st.secrets["ALPHA_KEY"]
TWELVE_KEY = st.secrets["TWELVE_KEY"]

@st.cache_data(ttl=3600)
def fetch_daily_history(from_sym, to_sym):
    url = "https://www.alphavantage.co/query"
    params = {"function": "FX_DAILY", "from_symbol": from_sym, "to_symbol": to_sym,
              "apikey": ALPHA_KEY, "outputsize": "full"}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data["Time Series FX (Daily)"]
    rows = []
    for date, values in prices.items():
        rows.append({"date": date, "open": float(values["1. open"]), "high": float(values["2. high"]),
                      "low": float(values["3. low"]), "close": float(values["4. close"])})
    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_gold_history():
    url = "https://www.alphavantage.co/query"
    params = {"function": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily", "apikey": ALPHA_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    records = data["data"]
    rows = [{"date": e["date"], "close": float(e["price"])} for e in records]
    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)

def build_features(df, has_ohlc=True):
    df = df.copy()
    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)
    df["change_1d"] = df["close"].diff(1)
    df["change_3d"] = df["close"].diff(3)
    df["change_5d"] = df["close"].diff(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_diff"] = df["ma_5"] - df["ma_20"]
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))
    if has_ohlc:
        df["swing_high_20"] = df["high"].shift(1).rolling(20).max()
        df["swing_low_20"] = df["low"].shift(1).rolling(20).min()
    else:
        df["swing_high_20"] = df["close"].shift(1).rolling(20).max()
        df["swing_low_20"] = df["close"].shift(1).rolling(20).min()
    df["dist_to_swing_high"] = (df["swing_high_20"] - df["close"]) / df["close"]
    df["dist_to_swing_low"] = (df["close"] - df["swing_low_20"]) / df["close"]
    def calc_slope(series):
        y = series.values
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0
    df["trend_slope"] = df["close"].rolling(10).apply(calc_slope, raw=False)
    df["fib_range"] = df["swing_high_20"] - df["swing_low_20"]
    df["fib_position"] = (df["close"] - df["swing_low_20"]) / df["fib_range"]
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_mid"] - (2 * df["bb_std"])
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df.dropna().reset_index(drop=True)

features = ["change_1d", "change_3d", "change_5d", "ma_diff", "rsi",
            "dist_to_swing_high", "dist_to_swing_low", "trend_slope",
            "fib_position", "macd", "macd_hist", "bb_position"]

def analyze_price_model(df):
    split_point = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_point], df.iloc[split_point:].copy()
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train_df[features], train_df["target"])
    test_df["prob_up"] = model.predict_proba(test_df[features])[:, 1]
    wins, losses = 0, 0
    for i in range(len(test_df) - 1):
        p = test_df["prob_up"].iloc[i]
        if p >= 0.55: d = 1
        elif p <= 0.45: d = -1
        else: continue
        change = (test_df["close"].iloc[i+1] - test_df["close"].iloc[i]) / test_df["close"].iloc[i]
        if (d == 1 and change > 0) or (d == -1 and change < 0): wins += 1
        else: losses += 1
    win_rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 50
    prob = model.predict_proba(df.iloc[[-1]][features])[0][1]
    return round(win_rate, 2), "BUY" if prob > 0.5 else "SELL", df["close"].iloc[-1]

@st.cache_data(ttl=1800)
def get_news_sentiment():
    params = {"q": '"USD/JPY" OR "Bank of Japan" OR "Japanese yen" OR "Federal Reserve" OR "interest rate" OR "safe haven" OR oil OR geopolitical',
              "language": "en", "sortBy": "publishedAt",
              "domains": "reuters.com,bloomberg.com,forexlive.com,investing.com,cnbc.com",
              "pageSize": 10, "apiKey": NEWS_API_KEY}
    r = requests.get("https://newsapi.org/v2/everything", params=params).json()
    if r.get("status") != "ok": return "NEUTRAL", []
    headlines = [a["title"] for a in r["articles"]]
    prompt = f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\nRespond in EXACTLY this format:\nBIAS: [BULLISH_USD or BEARISH_USD or NEUTRAL]"
    gr = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]}).json()
    bias = "NEUTRAL"
    if "choices" in gr:
        for line in gr["choices"][0]["message"]["content"].split("\n"):
            if line.startswith("BIAS:"): bias = line.replace("BIAS:", "").strip()
    return bias, headlines

pair_structure = {"USD/JPY": "USD", "GBP/USD": "GBP", "USD/CAD": "USD", "XAU/USD": "XAU"}
def get_news_direction(pair, news_bias):
    usd_base = pair_structure[pair] == "USD"
    if "BEARISH" in news_bias: return "SELL" if usd_base else "BUY"
    if "BULLISH" in news_bias: return "BUY" if usd_base else "SELL"
    return "NEUTRAL"

@st.cache_data(ttl=1800)
def fetch_intraday(symbol, interval):
    params = {"symbol": symbol, "interval": interval, "outputsize": 30, "apikey": TWELVE_KEY}
    r = requests.get("https://api.twelvedata.com/time_series", params=params).json()
    if r.get("status") != "ok": return None
    df = pd.DataFrame(r["values"])
    df["close"] = df["close"].astype(float)
    return df.iloc[::-1].reset_index(drop=True)

def get_tf_direction(df):
    if df is None or len(df) < 10: return "UNKNOWN"
    ma = df["close"].rolling(10).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if pd.isna(ma): return "UNKNOWN"
    return "UP" if price > ma else ("DOWN" if price < ma else "FLAT")

twelvedata_symbols = {"USD/JPY": "USD/JPY", "GBP/USD": "GBP/USD", "USD/CAD": "USD/CAD", "XAU/USD": "XAU/USD"}

def get_timeframe_direction(pair):
    symbol = twelvedata_symbols[pair]
    results = {tf: get_tf_direction(fetch_intraday(symbol, tf)) for tf in ["4h", "1h", "30min", "15min"]}
    up = list(results.values()).count("UP")
    down = list(results.values()).count("DOWN")
    return "BUY" if up > down else ("SELL" if down > up else "NEUTRAL")

@st.cache_data(ttl=21600)
def get_cot_positioning(contract_name, dataset_url, field_prefix, suffix=""):
    params = {"$where": f"market_and_exchange_names = '{contract_name}'",
              "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 1}
    r = requests.get(dataset_url, params=params).json()
    if not r: return "NEUTRAL"
    row = r[0]
    net = int(row.get(f"{field_prefix}_long{suffix}", 0)) - int(row.get(f"{field_prefix}_short{suffix}", 0))
    return "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL")

TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
DISAGG_URL = "https://publicreporting.cftc.gov/resource/kh3c-gbw2.json"
cot_contracts = {
    "USD/JPY": ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", TFF_URL, "lev_money_positions", ""),
    "GBP/USD": ("BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", TFF_URL, "lev_money_positions", ""),
    "USD/CAD": ("CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", TFF_URL, "lev_money_positions", ""),
    "XAU/USD": ("GOLD - COMMODITY EXCHANGE INC.", DISAGG_URL, "m_money_positions", "_all")
}

def get_pivot_direction(df, has_ohlc=True):
    y = df.iloc[-2]
    close = y["close"]
    if has_ohlc:
        high, low = y["high"], y["low"]
    else:
        high, low = close * 1.005, close * 0.995
    pivot = (high + low + close) / 3
    return "BUY" if df["close"].iloc[-1] > pivot else "SELL"

st.divider()
if st.button("🔄 Run Full YonKing Analysis"):
    news_bias, headlines = get_news_sentiment()
    st.subheader(f"News Sentiment: {news_bias}")

    results_table = []
    currency_pairs = {"USD/JPY": ("USD", "JPY"), "GBP/USD": ("GBP", "USD"), "USD/CAD": ("USD", "CAD")}

    for pair, (fs, ts) in currency_pairs.items():
        with st.spinner(f"Analyzing {pair}..."):
            raw_df = fetch_daily_history(fs, ts)
            featured_df = build_features(raw_df, True)
            win_rate, price_dir, current_price = analyze_price_model(featured_df)
            news_dir = get_news_direction(pair, news_bias)
            tf_dir = get_timeframe_direction(pair)
            contract, url, prefix, suffix = cot_contracts[pair]
            cot_dir = get_cot_positioning(contract, url, prefix, suffix)
            pivot_dir = get_pivot_direction(raw_df, True)

            buy_score = sell_score = 0
            if price_dir == "BUY": buy_score += win_rate
            else: sell_score += win_rate
            for d, w in [(news_dir, 52), (tf_dir, 55), (cot_dir, 58), (pivot_dir, 50)]:
                if d == "BUY": buy_score += w
                elif d == "SELL": sell_score += w

            total = buy_score + sell_score
            buy_conf = round((buy_score/total)*100, 1) if total > 0 else 0
            sell_conf = round((sell_score/total)*100, 1) if total > 0 else 0
            decision = "BUY" if buy_conf >= 60 else ("SELL" if sell_conf >= 60 else "WAIT")

            results_table.append({
                "Pair": pair, "Price": round(current_price, 4), "Price Model": f"{price_dir} ({win_rate}%)",
                "News": news_dir, "Timeframe": tf_dir, "COT": cot_dir, "Pivot": pivot_dir,
                "Buy Conf": f"{buy_conf}%", "Sell Conf": f"{sell_conf}%", "DECISION": decision
            })

    with st.spinner("Analyzing XAU/USD..."):
        gold_raw = fetch_gold_history()
        gold_featured = build_features(gold_raw, False)
        win_rate, price_dir, current_price = analyze_price_model(gold_featured)
        news_dir = get_news_direction("XAU/USD", news_bias)
        tf_dir = get_timeframe_direction("XAU/USD")
        contract, url, prefix, suffix = cot_contracts["XAU/USD"]
        cot_dir = get_cot_positioning(contract, url, prefix, suffix)
        pivot_dir = get_pivot_direction(gold_raw, False)

        buy_score = sell_score = 0
        if price_dir == "BUY": buy_score += win_rate
        else: sell_score += win_rate
        for d, w in [(news_dir, 52), (tf_dir, 55), (cot_dir, 58), (pivot_dir, 50)]:
            if d == "BUY": buy_score += w
            elif d == "SELL": sell_score += w

        total = buy_score + sell_score
        buy_conf = round((buy_score/total)*100, 1) if total > 0 else 0
        sell_conf = round((sell_score/total)*100, 1) if total > 0 else 0
        decision = "BUY" if buy_conf >= 60 else ("SELL" if sell_conf >= 60 else "WAIT")

        results_table.append({
            "Pair": "XAU/USD", "Price": round(current_price, 4), "Price Model": f"{price_dir} ({win_rate}%)",
            "News": news_dir, "Timeframe": tf_dir, "COT": cot_dir, "Pivot": pivot_dir,
            "Buy Conf": f"{buy_conf}%", "Sell Conf": f"{sell_conf}%", "DECISION": decision
        })

    st.dataframe(pd.DataFrame(results_table), use_container_width=True)

    st.divider()
    st.subheader("📊 Live Charts")

    tradingview_symbols = {
        "USD/JPY": "FX:USDJPY",
        "GBP/USD": "FX:GBPUSD",
        "USD/CAD": "FX:USDCAD",
        "XAU/USD": "OANDA:XAUUSD"
    }

    selected_pair = st.selectbox("Select a pair to view its live chart:", list(tradingview_symbols.keys()))
    tv_symbol = tradingview_symbols[selected_pair]

    tradingview_widget = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 500,
        "symbol": "{tv_symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    st.components.v1.html(tradingview_widget, height=520)

    with st.expander("Headlines used for news sentiment"):
        for h in headlines:
            st.write(f"- {h}")
