import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth as firebase_auth
import json
from streamlit_cookies_manager import EncryptedCookieManager

st.set_page_config(page_title="YonKing", page_icon="📈", layout="wide")

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
GROQ_KEY = st.secrets["GROQ_KEY"]
ALPHA_KEY = st.secrets["ALPHA_KEY"]
TWELVE_KEY = st.secrets["TWELVE_KEY"]

if not firebase_admin._apps:
    firebase_key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

cookies = EncryptedCookieManager(prefix="yonking_", password=st.secrets["ALPHA_KEY"])
if not cookies.ready():
    st.stop()

def signup_user(email, password, phone, age, location):
    try:
        user = firebase_auth.create_user(email=email, password=password, phone_number=phone)
        db.collection("pending_users").document(user.uid).set({
            "email": email, "phone": phone, "age": age, "location": location,
            "approved": False, "signup_date": datetime.date.today().strftime("%Y-%m-%d")
        })
        return True, "Account created! Waiting for approval before you can log in."
    except Exception as e:
        return False, str(e)

def check_approval(email):
    users = db.collection("pending_users").where("email", "==", email).stream()
    for u in users:
        data = u.to_dict()
        return data.get("approved", False)
    return False

def login_user(email, password):
    try:
        user = firebase_auth.get_user_by_email(email)
        if email == "obidiharry@gmail.com":
            return True, "Login successful!"
        if not check_approval(email):
            return False, "Your account is still pending approval."
        return True, "Login successful!"
    except Exception as e:
        return False, "Invalid email or account not found."

if "logged_in" not in st.session_state:
    saved_email = cookies.get("user_email")
    if saved_email:
        st.session_state.logged_in = True
        st.session_state.user_email = saved_email
    else:
        st.session_state.logged_in = False
        st.session_state.user_email = None

if not st.session_state.logged_in:
    st.title("📈 Welcome to YonKing")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            success, message = login_user(login_email, login_password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_email = login_email
                cookies["user_email"] = login_email
                cookies.save()
                st.rerun()
            else:
                st.error(message)
    with tab2:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_phone = st.text_input("Phone Number (e.g. +2348012345678)", key="signup_phone")
        signup_age = st.number_input("Age", min_value=1, max_value=120, key="signup_age")
        signup_location = st.text_input("Location (City, Country)", key="signup_location")
        if st.button("Sign Up"):
            if signup_email and signup_password and signup_phone and signup_location:
                success, message = signup_user(signup_email, signup_password, signup_phone, signup_age, signup_location)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please fill in all fields.")
    st.stop()

st.sidebar.write(f"Logged in as: {st.session_state.user_email}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = None
    cookies["user_email"] = ""
    cookies.save()
    st.rerun()

twelvedata_symbols = {"USD/JPY": "USD/JPY", "GBP/USD": "GBP/USD", "USD/CAD": "USD/CAD", "XAU/USD": "XAU/USD"}

def backtest_intraday_model(pair, interval, symbols_map):
    try:
        symbol = symbols_map[pair]
        params = {"symbol": symbol, "interval": interval, "outputsize": 5000, "apikey": TWELVE_KEY}
        r = requests.get("https://api.twelvedata.com/time_series", params=params).json()
        if "values" not in r:
            return None
        df = pd.DataFrame(r["values"])
        df["close"] = df["close"].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        df["next_close"] = df["close"].shift(-1)
        df["target"] = (df["next_close"] > df["close"]).astype(int)
        df["change_1"] = df["close"].diff(1)
        df["change_3"] = df["close"].diff(3)
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_diff"] = df["ma_5"] - df["ma_20"]
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + rs))
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_position"] = (df["close"] - (df["bb_mid"] - 2*df["bb_std"])) / (4*df["bb_std"])
        df = df.dropna().reset_index(drop=True)
        feats = ["change_1", "change_3", "ma_diff", "rsi", "macd", "macd_hist", "bb_position"]
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:].copy()
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(train[feats], train["target"])
        test["prob_up"] = model.predict_proba(test[feats])[:, 1]
        wins, losses = 0, 0
        for i in range(len(test) - 1):
            p = test["prob_up"].iloc[i]
            if p >= 0.55: d = 1
            elif p <= 0.45: d = -1
            else: continue
            change = (test["close"].iloc[i+1] - test["close"].iloc[i]) / test["close"].iloc[i]
            if (d == 1 and change > 0) or (d == -1 and change < 0): wins += 1
            else: losses += 1
        win_rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
        return win_rate, wins, losses, len(df)
    except Exception:
        return None

if st.sidebar.button("TEST: Backtest 1H model honestly"):
    for pair in ["USD/JPY", "GBP/USD", "USD/CAD", "XAU/USD"]:
        result = backtest_intraday_model(pair, "1h", twelvedata_symbols)
        if result:
            win_rate, wins, losses, total_candles = result
            st.sidebar.write(f"**{pair}** ({total_candles} candles): {win_rate:.2f}% win rate ({wins}W/{losses}L)")
        else:
            st.sidebar.write(f"{pair}: data unavailable")

if st.sidebar.button("TEST: Backtest 4H model honestly"):
    for pair in ["USD/JPY", "GBP/USD", "USD/CAD", "XAU/USD"]:
        result = backtest_intraday_model(pair, "4h", twelvedata_symbols)
        if result:
            win_rate, wins, losses, total_candles = result
            st.sidebar.write(f"**{pair}** ({total_candles} candles): {win_rate:.2f}% win rate ({wins}W/{losses}L)")
        else:
            st.sidebar.write(f"{pair}: data unavailable")

ADMIN_EMAIL = "obidiharry@gmail.com"
if st.session_state.user_email == ADMIN_EMAIL:
    with st.sidebar.expander("🔑 Admin Panel"):
        st.write("Pending Approvals")
        pending = db.collection("pending_users").where("approved", "==", False).stream()
        for p in pending:
            data = p.to_dict()
            st.write(f"**{data['email']}** | Age: {data['age']} | Location: {data['location']} | Phone: {data['phone']}")
            if st.button(f"Approve {data['email']}", key=f"approve_{p.id}"):
                db.collection("pending_users").document(p.id).update({"approved": True})
                st.success(f"Approved {data['email']}")
                st.rerun()

st.title("📈 YonKing - Forex Analysis Dashboard")
st.caption("AI-powered price prediction, news sentiment, institutional positioning, and timeframe analysis")

@st.cache_data(ttl=3600)
def fetch_daily_history(from_sym, to_sym):
    url = "https://www.alphavantage.co/query"
    params = {"function": "FX_DAILY", "from_symbol": from_sym, "to_symbol": to_sym,
              "apikey": ALPHA_KEY, "outputsize": "full"}
    response = requests.get(url, params=params)
    data = response.json()
    if "Time Series FX (Daily)" not in data:
        st.warning(f"⚠️ Data temporarily unavailable for {from_sym}/{to_sym} (API limit reached). Try again in a few minutes.")
        st.stop()
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
    if "data" not in data:
        st.warning("⚠️ Gold data temporarily unavailable (API limit reached). Try again in a few minutes.")
        st.stop()
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
    if has_ohlc:
        df["prev_close_val"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["prev_close_val"])
        df["tr3"] = abs(df["low"] - df["prev_close_val"])
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(14).mean()
    else:
        df["atr"] = df["close"].diff().abs().rolling(14).mean()
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

def detect_liquidity_sweep_and_retest(df, has_ohlc=True):
    lookback = 20
    if len(df) < lookback + 5:
        return False, False
    recent = df.iloc[-lookback-5:-1]
    current = df.iloc[-1]
    if has_ohlc:
        prior_high = recent["high"].iloc[:-1].max()
        prior_low = recent["low"].iloc[:-1].min()
        swept_high = recent["high"].iloc[-1] > prior_high and current["close"] < prior_high
        swept_low = recent["low"].iloc[-1] < prior_low and current["close"] > prior_low
    else:
        prior_high = recent["close"].iloc[:-1].max()
        prior_low = recent["close"].iloc[:-1].min()
        swept_high = recent["close"].iloc[-1] > prior_high and current["close"] < prior_high
        swept_low = recent["close"].iloc[-1] < prior_low and current["close"] > prior_low
    liquidity_sweep = swept_high or swept_low
    price_range = df["close"].tail(5).max() - df["close"].tail(5).min()
    avg_range = df["close"].diff().abs().tail(20).mean()
    retest_happening = price_range < (avg_range * 2) if avg_range > 0 else False
    return liquidity_sweep, retest_happening

def calculate_confidence_score(price_dir, daily_dir, tf_4h_dir, tf_1h_dir, bos_detected,
                                 retest_detected, atr_expanding, rsi_value, macd_agrees, news_favorable):
    score = 0
    if daily_dir == price_dir:
        score += 15
    if tf_4h_dir == price_dir:
        score += 15
    if tf_1h_dir == price_dir:
        score += 10
    if bos_detected:
        score += 15
    if retest_detected:
        score += 15
    if atr_expanding:
        score += 10
    if macd_agrees:
        score += 5
    if 45 <= rsi_value <= 65:
        score += 5
    if news_favorable:
        score += 10
    return score

def get_confidence_label(score):
    if score < 60:
        return "NO TRADE"
    elif score < 75:
        return "MEDIUM"
    elif score < 85:
        return "HIGH"
    else:
        return "VERY HIGH"

def calculate_adx(df, period=14):
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

def is_good_trading_session():
    hour = datetime.datetime.utcnow().hour
    london = 7 <= hour < 16
    new_york = 12 <= hour < 21
    return london or new_york

def detect_fair_value_gap(df, direction, has_ohlc=True):
    if len(df) < 3:
        return False
    if has_ohlc:
        c1_high, c1_low = df["high"].iloc[-3], df["low"].iloc[-3]
        c3_high, c3_low = df["high"].iloc[-1], df["low"].iloc[-1]
    else:
        c1_high = c1_low = df["close"].iloc[-3]
        c3_high = c3_low = df["close"].iloc[-1]
    if direction == "BUY":
        return c3_low > c1_high
    else:
        return c3_high < c1_low

def calculate_trade_levels(direction, current_price, atr, swing_high, swing_low):
    if direction == "BUY":
        dist_to_support = current_price - swing_low
        if 0 < dist_to_support < (atr * 3):
            stop_loss_price = swing_low - (atr * 0.3)
        else:
            stop_loss_price = current_price - (atr * 1.5)
        dist_to_resist = swing_high - current_price
        if dist_to_resist > 0:
            take_profit_price = swing_high - (atr * 0.2)
            if take_profit_price <= current_price:
                take_profit_price = current_price + (atr * 1.5)
        else:
            take_profit_price = current_price + (atr * 1.5)
    else:
        dist_to_resist = swing_high - current_price
        if 0 < dist_to_resist < (atr * 3):
            stop_loss_price = swing_high + (atr * 0.3)
        else:
            stop_loss_price = current_price + (atr * 1.5)
        dist_to_support = current_price - swing_low
        if dist_to_support > 0:
            take_profit_price = swing_low + (atr * 0.2)
            if take_profit_price >= current_price:
                take_profit_price = current_price - (atr * 1.5)
        else:
            take_profit_price = current_price - (atr * 1.5)
    return round(stop_loss_price, 5), round(take_profit_price, 5)

def analyze_intraday_signal(pair, interval, daily_dir):
    try:
        symbol = twelvedata_symbols[pair]
        df = fetch_intraday(symbol, interval)
        if df is None or len(df) < 30:
            return "WAIT", None, None, None, 0

        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + rs))
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal_line"]
        df["tr"] = (df["close"] - df["close"].shift(1)).abs()
        df["atr"] = df["tr"].rolling(14).mean()

        current_price = df["close"].iloc[-1]
        ma5 = df["ma_5"].iloc[-1]
        ma20 = df["ma_20"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        macd_hist = df["macd_hist"].iloc[-1]
        atr_now = df["atr"].iloc[-1]
        atr_avg = df["atr"].tail(20).mean()

        if pd.isna(ma5) or pd.isna(ma20) or pd.isna(rsi) or pd.isna(atr_now):
            return "WAIT", None, None, None, 0

        candidate_dir = "BUY" if ma5 > ma20 else "SELL"

        adx = calculate_adx(df)
        liquidity_sweep, retest = detect_liquidity_sweep_and_retest(df, has_ohlc=False)
        fvg = detect_fair_value_gap(df, candidate_dir, has_ohlc=False)
        session_ok = is_good_trading_session()
        atr_expanding = atr_now > atr_avg
        macd_agrees = (macd_hist > 0 and candidate_dir == "BUY") or (macd_hist < 0 and candidate_dir == "SELL")

        swing_high = df["close"].rolling(20).max().iloc[-1]
        swing_low = df["close"].rolling(20).min().iloc[-1]
        bos_detected = (candidate_dir == "BUY" and current_price > swing_high * 0.999) or \
                       (candidate_dir == "SELL" and current_price < swing_low * 1.001)

        score = calculate_confidence_score(
            price_dir=candidate_dir, daily_dir=daily_dir, tf_4h_dir=candidate_dir, tf_1h_dir=candidate_dir,
            bos_detected=bos_detected, retest_detected=(retest or fvg), atr_expanding=atr_expanding,
            rsi_value=rsi, macd_agrees=macd_agrees, news_favorable=True
        )

        if adx < 20:
            score -= 20
        if not session_ok:
            score -= 15

        if score < 60:
            return "WAIT", current_price, atr_now, (swing_high, swing_low), score

        return candidate_dir, current_price, atr_now, (swing_high, swing_low), score
    except Exception:
        return "WAIT", None, None, None, 0

def has_open_trade(pair, timeframe="Daily"):
    docs = db.collection("trade_history").where("pair", "==", pair).where("timeframe", "==", timeframe).where("outcome", "==", "OPEN").stream()
    return any(True for _ in docs)

def save_decisions_to_firestore(results_table):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    for row in results_table:
        tf_label = row.get("Timeframe_Label", "Daily")
        if row["DECISION"] != "WAIT":
            if has_open_trade(row["Pair"], tf_label):
                continue
            doc_id = f"{today_str}_{row['Pair'].replace('/', '')}_{tf_label}_{now_str.replace(' ','').replace(':','')}"
            db.collection("trade_history").document(doc_id).set({
                "date": today_str, "pair": row["Pair"], "timeframe": tf_label, "decision": row["DECISION"],
                "entry_price": row["Price"], "stop_loss": row["Stop Loss"],
                "take_profit": row["Take Profit"], "outcome": "OPEN",
                "opened_at": now_str, "closed_at": ""
            })

def check_previous_trades():
    docs = db.collection("trade_history").stream()
    results = []

    for doc in docs:
        trade = doc.to_dict()

        pair = trade.get("pair")
        outcome = trade.get("outcome", "OPEN")
        decision = trade.get("decision", "WAIT")

        # IMPORTANT:
        # Never close OPEN trades using stale weekend prices.
        if outcome == "OPEN" and market_closed:
            results.append({
                "Pair": pair,
                "Timeframe": trade.get("timeframe", "Daily"),
                "Decision": decision,
                "Entry": trade.get("entry_price", "-"),
                "Exit/Current": "-",
                "Opened": trade.get("opened_at", trade.get("date", "-")),
                "Closed": "OPEN",
                "Status": "OPEN"
            })
            continue

        # Only check live price when the market is open.
        if outcome == "OPEN":
            try:
                if pair == "XAU/USD":
                    current_df = fetch_gold_history()
                else:
                    fs, ts = {
                        "USD/JPY": ("USD", "JPY"),
                        "GBP/USD": ("GBP", "USD"),
                        "USD/CAD": ("USD", "CAD")
                    }[pair]

                    current_df = fetch_daily_history(fs, ts)

                current_price = current_df["close"].iloc[-1]

            except Exception:
                continue

            sl = trade.get("stop_loss")
            tp = trade.get("take_profit")

            if sl is None or tp is None:
                continue

            if decision == "BUY":
                if current_price >= tp:
                    outcome = "WIN"
                elif current_price <= sl:
                    outcome = "LOSS"

            elif decision == "SELL":
                if current_price <= tp:
                    outcome = "WIN"
                elif current_price >= sl:
                    outcome = "LOSS"

            # Save result ONLY when a real price has actually hit SL/TP.
            if outcome != "OPEN":
                now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

                db.collection("trade_history").document(doc.id).update({
                    "outcome": outcome,
                    "closed_at": now_str,
                    "exit_price": round(current_price, 5)
                })

                trade["closed_at"] = now_str
                trade["exit_price"] = round(current_price, 5)

            else:
                trade["exit_price"] = round(current_price, 5)

        results.append({
            "Pair": pair,
            "Timeframe": trade.get("timeframe", "Daily"),
            "Decision": decision,
            "Entry": trade.get("entry_price", "-"),
            "Exit/Current": trade.get("exit_price", "-"),
            "Opened": trade.get("opened_at", trade.get("date", "-")),
            "Closed": trade.get("closed_at", "OPEN") if outcome != "OPEN" else "OPEN",
            "Status": outcome
        })

    return results

now_utc = datetime.datetime.utcnow()
weekday = now_utc.weekday()
hour = now_utc.hour
market_closed = False
if weekday == 5: market_closed = True
elif weekday == 6 and hour < 21: market_closed = True
elif weekday == 4 and hour >= 21: market_closed = True

if market_closed:
    st.warning("⚠️ Forex market is currently CLOSED (weekend). Prices shown may be from Friday's close.")

if "results_table" not in st.session_state:
    st.session_state.results_table = None
    st.session_state.headlines = None
    st.session_state.news_bias = None

def run_full_analysis():
    news_bias, headlines = get_news_sentiment()
    results_table = []
    daily_directions = {}
    currency_pairs = {"USD/JPY": ("USD", "JPY"), "GBP/USD": ("GBP", "USD"), "USD/CAD": ("USD", "CAD")}
    for pair, (fs, ts) in currency_pairs.items():
        raw_df = fetch_daily_history(fs, ts)
        featured_df = build_features(raw_df, True)
        win_rate, price_dir, current_price = analyze_price_model(featured_df)
        daily_directions[pair] = price_dir
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
        sl_price, tp_price = "-", "-"
        if decision != "WAIT":
            atr = featured_df["atr"].iloc[-1]
            swing_high = featured_df["swing_high_20"].iloc[-1]
            swing_low = featured_df["swing_low_20"].iloc[-1]
            sl_price, tp_price = calculate_trade_levels(decision, current_price, atr, swing_high, swing_low)
        results_table.append({
            "Pair": pair, "Price": round(current_price, 4), "Price Model": f"{price_dir} ({win_rate}%)",
            "News": news_dir, "Timeframe": tf_dir, "COT": cot_dir, "Pivot": pivot_dir,
            "Buy Conf": f"{buy_conf}%", "Sell Conf": f"{sell_conf}%", "DECISION": decision,
            "Stop Loss": sl_price, "Take Profit": tp_price, "Timeframe_Label": "Daily"
        })

    gold_raw = fetch_gold_history()
    gold_featured = build_features(gold_raw, False)
    win_rate, price_dir, current_price = analyze_price_model(gold_featured)
    daily_directions["XAU/USD"] = price_dir
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
    sl_price, tp_price = "-", "-"
    if decision != "WAIT":
        atr = gold_featured["atr"].iloc[-1]
        swing_high = gold_featured["swing_high_20"].iloc[-1]
        swing_low = gold_featured["swing_low_20"].iloc[-1]
        sl_price, tp_price = calculate_trade_levels(decision, current_price, atr, swing_high, swing_low)
    results_table.append({
        "Pair": "XAU/USD", "Price": round(current_price, 4), "Price Model": f"{price_dir} ({win_rate}%)",
        "News": news_dir, "Timeframe": tf_dir, "COT": cot_dir, "Pivot": pivot_dir,
        "Buy Conf": f"{buy_conf}%", "Sell Conf": f"{sell_conf}%", "DECISION": decision,
        "Stop Loss": sl_price, "Take Profit": tp_price, "Timeframe_Label": "Daily"
    })

    intraday_allowed = {"1H": ["USD/JPY", "USD/CAD", "XAU/USD"], "4H": ["GBP/USD"]}

    for label, interval in [("1H", "1h"), ("4H", "4h")]:
        for pair in intraday_allowed[label]:
            try:
                daily_dir_for_pair = daily_directions.get(pair, "WAIT")
                direction, price, atr, levels, score = analyze_intraday_signal(pair, interval, daily_dir_for_pair)
                if direction != "WAIT" and price and atr and levels:
                    swing_high, swing_low = levels
                    sl, tp = calculate_trade_levels(direction, price, atr, swing_high, swing_low)
                    results_table.append({
                        "Pair": pair, "Price": round(price, 4), "Price Model": "-", "News": "-",
                        "Timeframe": label, "COT": "-", "Pivot": "-", "Buy Conf": "-",
                        "Sell Conf": f"Score: {score}", "DECISION": direction,
                        "Stop Loss": sl, "Take Profit": tp, "Timeframe_Label": label
                    })
            except Exception as e:
                st.sidebar.write(f"⚠️ Skipped {pair} {label} due to error: {str(e)[:50]}")
                continue

    st.session_state.results_table = results_table
    st.session_state.headlines = headlines
    st.session_state.news_bias = news_bias
    save_decisions_to_firestore(results_table)

st.divider()
st.subheader("📜 Trade History & Results")
past_results = check_previous_trades()
if past_results:
    df_results = pd.DataFrame(past_results)
    def color_status(row):
        if row["Status"] == "WIN":
            return ["background-color: #1a4d2e"] * len(row)
        elif row["Status"] == "LOSS":
            return ["background-color: #5c1a1a"] * len(row)
        return [""] * len(row)
    st.dataframe(df_results.style.apply(color_status, axis=1), use_container_width=True)
else:
    st.write("No trade history yet - results will appear here after your first analysis run.")

if st.session_state.results_table is None and not market_closed:
    with st.spinner("Running YonKing's full analysis automatically..."):
        run_full_analysis()

elif st.session_state.results_table is None and market_closed:
    st.info("⏸️ YonKing analysis is paused because the market is closed. No new paper trades will be created.")

if st.button("🔄 Refresh Analysis"):
    if market_closed:
        st.warning("⏸️ Market is closed. YonKing will not create a new trade from stale weekend prices.")
    else:
        with st.spinner("Refreshing..."):
            run_full_analysis()

if st.session_state.results_table is not None:
    st.subheader(f"News Sentiment: {st.session_state.news_bias}")
    st.dataframe(pd.DataFrame(st.session_state.results_table), use_container_width=True)
    st.info("💡 For any BUY/SELL decision: enter at current Price, set Stop Loss and Take Profit to the exact values shown above in MT5/MT4.")

    st.divider()
    st.subheader("📊 YonKing's Calculated Chart View")
    chart_pairs = {"USD/JPY": ("USD", "JPY", True), "GBP/USD": ("GBP", "USD", True),
                   "USD/CAD": ("USD", "CAD", True), "XAU/USD": (None, None, False)}
    selected_pair = st.selectbox("Select a pair to view its chart with calculated levels:", list(chart_pairs.keys()))
    fs, ts, has_ohlc = chart_pairs[selected_pair]
    if has_ohlc:
        chart_df = fetch_daily_history(fs, ts).tail(60).reset_index(drop=True)
    else:
        chart_df = fetch_gold_history().tail(60).reset_index(drop=True)
    fig = go.Figure()
    if has_ohlc:
        fig.add_trace(go.Candlestick(x=chart_df["date"], open=chart_df["open"], high=chart_df["high"],
                                       low=chart_df["low"], close=chart_df["close"], name=selected_pair))
    else:
        fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["close"], mode="lines", name=selected_pair))
    if has_ohlc:
        swing_high = chart_df["high"].rolling(20, min_periods=1).max().iloc[-1]
        swing_low = chart_df["low"].rolling(20, min_periods=1).min().iloc[-1]
    else:
        swing_high = chart_df["close"].rolling(20, min_periods=1).max().iloc[-1]
        swing_low = chart_df["close"].rolling(20, min_periods=1).min().iloc[-1]
    fig.add_hline(y=swing_high, line_dash="dash", line_color="red", annotation_text=f"Resistance: {round(swing_high,4)}")
    fig.add_hline(y=swing_low, line_dash="dash", line_color="green", annotation_text=f"Support: {round(swing_low,4)}")
    y_prev = chart_df.iloc[-2]
    close_p = y_prev["close"]
    if has_ohlc:
        high_p, low_p = y_prev["high"], y_prev["low"]
    else:
        high_p, low_p = close_p * 1.005, close_p * 0.995
    pivot = (high_p + low_p + close_p) / 3
    fig.add_hline(y=pivot, line_dash="dot", line_color="yellow", annotation_text=f"Pivot: {round(pivot,4)}")
    recent_lows = chart_df["low"] if has_ohlc else chart_df["close"]
    min_idx = recent_lows.idxmin()
    if min_idx < len(chart_df) - 1:
        fig.add_trace(go.Scatter(
            x=[chart_df["date"].iloc[min_idx], chart_df["date"].iloc[-1]],
            y=[recent_lows.iloc[min_idx], chart_df["close"].iloc[-1]],
            mode="lines", line=dict(color="cyan", width=2, dash="solid"), name="Trend line"
        ))
    fig.update_layout(title=f"{selected_pair} - Price with Calculated Levels", template="plotly_dark",
                       height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Headlines used for news sentiment"):
        for h in st.session_state.headlines:
            st.write(f"- {h}")
