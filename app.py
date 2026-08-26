import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import datetime
import json
import re
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth as firebase_auth
from streamlit_cookies_manager import EncryptedCookieManager

st.set_page_config(page_title="YonKing", page_icon="📈", layout="wide")

# ===================== CONFIG =====================
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
GROQ_KEY = st.secrets["GROQ_KEY"]
ALPHA_KEY = st.secrets["ALPHA_KEY"]
TWELVE_KEY = st.secrets["TWELVE_KEY"]
FIREBASE_KEY = st.secrets["FIREBASE_KEY"]

PAIRS = ["USD/JPY", "GBP/USD", "USD/CAD", "XAU/USD"]
CURRENCY_PAIRS = {"USD/JPY": ("USD", "JPY"), "GBP/USD": ("GBP", "USD"), "USD/CAD": ("USD", "CAD")}
SYMBOLS = {p: p for p in PAIRS}
ADMIN_EMAIL = "obidiharry@gmail.com"

# ===================== FIREBASE SAFE INIT =====================
firebase_key = json.loads(FIREBASE_KEY)
try:
    firebase_app = firebase_admin.get_app("yonking")
except ValueError:
    firebase_app = firebase_admin.initialize_app(credentials.Certificate(firebase_key), name="yonking")

db = firestore.client(app=firebase_app)

# ===================== COOKIES =====================
cookies = EncryptedCookieManager(prefix="yonking_", password=ALPHA_KEY)
if not cookies.ready():
    st.stop()

# ===================== SESSION =====================
for k, v in {
    "logged_in": False,
    "user_email": None,
    "results_table": None,
    "headlines": [],
    "news_data": {},
    "backtest_results": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

saved_email = cookies.get("user_email")
if not st.session_state.logged_in and saved_email:
    st.session_state.logged_in = True
    st.session_state.user_email = saved_email

# ===================== AUTH =====================
def signup_user(email, password, phone, age, location):
    try:
        user = firebase_auth.create_user(email=email, password=password, phone_number=phone, app=firebase_app)
        db.collection("pending_users").document(user.uid).set({
            "email": email, "phone": phone, "age": int(age), "location": location,
            "approved": False, "signup_date": datetime.date.today().strftime("%Y-%m-%d")
        })
        return True, "Account created! Waiting for approval before you can log in."
    except Exception as e:
        return False, str(e)

def check_approval(email):
    try:
        for u in db.collection("pending_users").where("email", "==", email).stream():
            return bool(u.to_dict().get("approved", False))
    except Exception:
        pass
    return False

def login_user(email, password):
    try:
        firebase_auth.get_user_by_email(email, app=firebase_app)
        if email != ADMIN_EMAIL and not check_approval(email):
            return False, "Your account is still pending approval."
        return True, "Login successful!"
    except Exception:
        return False, "Invalid email or account not found."

if not st.session_state.logged_in:
    st.title("📈 Welcome to YonKing")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True):
            ok, msg = login_user(email, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                cookies["user_email"] = email
                cookies.save()
                st.rerun()
            st.error(msg)
    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        phone = st.text_input("Phone Number (e.g. +2348012345678)", key="signup_phone")
        age = st.number_input("Age", min_value=1, max_value=120, value=18, key="signup_age")
        location = st.text_input("Location (City, Country)", key="signup_location")
        if st.button("Sign Up", use_container_width=True):
            if email and password and phone and location:
                ok, msg = signup_user(email, password, phone, age, location)
                (st.success if ok else st.error)(msg)
            else:
                st.warning("Please fill in all fields.")
    st.stop()

# ===================== MARKET =====================
def market_open():
    now = datetime.datetime.utcnow()
    if now.weekday() == 5:
        return False
    if now.weekday() == 6 and now.hour < 21:
        return False
    if now.weekday() == 4 and now.hour >= 21:
        return False
    return True

MARKET_OPEN = market_open()

# ===================== SIDEBAR =====================
st.sidebar.write(f"Logged in as: {st.session_state.user_email}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = None
    cookies["user_email"] = ""
    cookies.save()
    st.rerun()

if MARKET_OPEN:
    st.sidebar.success("🟢 Market open. Strict analysis enabled.")
else:
    st.sidebar.warning("🔴 Market closed. No new paper trades will be saved.")

# ===================== API =====================
def get_json(url, params, timeout=25):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=1800)
def fetch_daily_history(frm, to):
    d = get_json("https://www.alphavantage.co/query", {
        "function": "FX_DAILY", "from_symbol": frm, "to_symbol": to,
        "apikey": ALPHA_KEY, "outputsize": "full"
    })
    if not d or "Time Series FX (Daily)" not in d:
        return None
    rows = []
    for date, v in d["Time Series FX (Daily)"].items():
        try:
            rows.append({"date": date, "open": float(v["1. open"]), "high": float(v["2. high"]), "low": float(v["3. low"]), "close": float(v["4. close"])})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else None

@st.cache_data(ttl=1800)
def fetch_gold_history():
    d = get_json("https://www.alphavantage.co/query", {"function": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily", "apikey": ALPHA_KEY})
    if not d or "data" not in d:
        return None
    rows = []
    for x in d["data"]:
        try:
            rows.append({"date": x["date"], "close": float(x["price"])})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else None

@st.cache_data(ttl=600)
def fetch_intraday(symbol, interval, outputsize=300):
    d = get_json("https://api.twelvedata.com/time_series", {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_KEY})
    if not d or "values" not in d:
        return None
    df = pd.DataFrame(d["values"])
    if "close" not in df:
        return None
    for c in ["open", "high", "low", "close"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).iloc[::-1].reset_index(drop=True)

# ===================== FEATURES / MODEL =====================
FEATURES = ["change_1d", "change_3d", "change_5d", "ma_diff", "rsi", "dist_to_swing_high", "dist_to_swing_low", "trend_slope", "fib_position", "macd", "macd_hist", "bb_position"]

def build_features(df, ohlc=True):
    if df is None or len(df) < 80:
        return None
    x = df.copy()
    x["next_close"] = x["close"].shift(-1)
    x["target"] = np.where(x["next_close"].notna(), (x["next_close"] > x["close"]).astype(int), np.nan)
    x["change_1d"] = x["close"].pct_change(1)
    x["change_3d"] = x["close"].pct_change(3)
    x["change_5d"] = x["close"].pct_change(5)
    x["ma_5"] = x["close"].rolling(5).mean()
    x["ma_20"] = x["close"].rolling(20).mean()
    x["ma_diff"] = (x["ma_5"] - x["ma_20"]) / x["close"]
    delta = x["close"].diff()
    gain = delta.clip(loelseling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    x["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    if ohlc:
        x["swing_high_20"] = x["high"].shift(1).rolling(20).max()
        x["swing_low_20"] = x["low"].shift(1).rolling(20).min()
    else:
        x["swing_high_20"] = x["close"].shift(1).rolling(20).max()
        x["swing_low_20"] = x["close"].shift(1).rolling(20).min()
    x["dist_to_swing_high"] = (x["swing_high_20"] - x["close"]) / x["close"]
    x["dist_to_swing_low"] = (x["close"] - x["swing_low_20"]) / x["close"]
    x["trend_slope"] = x["close"].rolling(10).apply(lambda a: np.polyfit(np.arange(len(a)), a, 1)[0] / np.mean(a) if np.mean(a) else 0, raw=False)
    x["fib_range"] = x["swing_high_20"] - x["swing_low_20"]
    x["fib_position"] = (x["close"] - x["swing_low_20"]) / x["fib_range"].replace(0, np.nan)
    x["ema12"] = x["close"].ewm(span=12, adjust=False).mean()
    x["ema26"] = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    x["bb_mid"] = x["close"].rolling(20).mean()
    x["bb_std"] = x["close"].rolling(20).std()
    x["bb_upper"] = x["bb_mid"] + 2*x["bb_std"]
    x["bb_lower"] = x["bb_mid"] - 2*x["bb_std"]
    x["bb_position"] = (x["close"] - x["bb_lower"]) / (x["bb_upper"] - x["bb_lower"]).replace(0, np.nan)
    if ohlc:
        pc = x["close"].shift(1)
        tr = pd.concat([x["high"]-x["low"], (x["high"]-pc).abs(), (x["low"]-pc).abs()], axis=1).max(axis=1)
        x["atr"] = tr.rolling(14).mean()
    else:
        x["atr"] = x["close"].diff().abs().rolling(14).mean()
    return x.dropna().reset_index(drop=True)

def train_model(df):
    if df is None:
        return None
    d = df.dropna(subset=FEATURES+["target"]).copy()
    if len(d) < 160 or d["target"].nunique() < 2:
        return None
    split = int(len(d)*0.7)
    train, test = d.iloc[:split], d.iloc[split:]
    model = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=4, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["target"])
    probs = model.predict_proba(test[FEATURES])[:,1]
    wins = losses = 0
    for i in range(len(test)-1):
        p = probs[i]
        if p >= .55: direction = 1
        elif p <= .45: direction = -1
        else: continue
        change = (test["close"].iloc[i+1]-test["close"].iloc[i]) / test["close"].iloc[i]
        if (direction == 1 and change > 0) or (direction == -1 and change < 0): wins += 1
        else: losses += 1
    trades = wins+losses
    oos = wins/trades*100 if trades else 0
    final = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=4, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    final.fit(d[FEATURES], d["target"])
    p = float(final.predict_proba(d.iloc[[-1]][FEATURES])[0,1])
    return {"buy": round(p*100,1), "sell": round((1-p)*100,1), "direction": "BUY" if p >= .5 else "SELL", "oos": round(oos,2), "wins": wins, "losses": losses}

# ===================== NEWS PERCENTAGES =====================
@st.cache_data(ttl=900)
def get_news():
    base = {p:{"buy":50.0,"sell":50.0,"status":"NEWS UNAVAILABLE"} for p in PAIRS}
    d = get_json("https://newsapi.org/v2/everything", {
        "q": '"USD/JPY" OR "GBP/USD" OR "USD/CAD" OR gold OR "Bank of Japan" OR "Federal Reserve" OR "Bank of England" OR "Bank of Canada" OR inflation OR "interest rate" OR oil OR geopolitical',
        "language":"en", "sortBy":"publishedAt", "pageSize":20, "apiKey":NEWS_API_KEY
    })
    if not d or d.get("status") != "ok": return base, []
    headlines = [a.get("title") for a in d.get("articles",[]) if a.get("title")][:20]
    prompt = """Analyze the headlines as directional evidence. Return exactly four lines in this format, with BUY+SELL=100 for each pair:\nUSD/JPY BUY: number SELL: number\nGBP/USD BUY: number SELL: number\nUSD/CAD BUY: number SELL: number\nXAU/USD BUY: number SELL: number\nDo not add explanations. Headlines:\n""" + "\n".join("- "+h for h in headlines)
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"}, json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],"temperature":0}, timeout=30)
        text = r.json()["choices"][0]["message"]["content"]
        for pair in PAIRS:
            m = re.search(re.escape(pair)+r"\s+BUY:\s*(\d+(?:\.\d+)?)\s+SELL:\s*(\d+(?:\.\d+)?)", text, re.I)
            if m:
                b,s=float(m.group(1)),float(m.group(2)); total=b+s
                if total>0:
                    b=round(b/total*100,1); base[pair]={"buy":b,"sell":round(100-b,1),"status":"OK"}
    except Exception:
        pass
    return base, headlines

# ===================== TIMEFRAME / COT =====================
def tf_direction(df):
    if df is None or len(df)<50: return "UNKNOWN",50
    ma20=df.close.rolling(20).mean().iloc[-1]; ma50=df.close.rolling(50).mean().iloc[-1]; price=df.close.iloc[-1]
    if price>ma20 and ma20>=ma50: return "BUY",70
    if price<ma20 and ma20<=ma50: return "SELL",70
    if price>ma20: return "BUY",60
    if price<ma20: return "SELL",60
    return "NEUTRAL",50

@st.cache_data(ttl=600)
def timeframe_evidence(pair):
    ds={}
    for interval,label in [("4h","4H"),("1h","1H"),("30min","30M"),("15min","15M")]:
        ds[label]=tf_direction(fetch_intraday(SYMBOLS[pair],interval,150))[0]
    b=sum(v=="BUY" for v in ds.values()); s=sum(v=="SELL" for v in ds.values())
    if b>s: return "BUY", round(50+12.5*b,1), ds
    if s>b: return "SELL", round(50+12.5*s,1), ds
    return "NEUTRAL",50,ds

COT = {
    "USD/JPY": ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE","https://publicreporting.cftc.gov/resource/gpe5-46if.json","lev_money_positions",""),
    "GBP/USD": ("BRITISH POUND - CHICAGO MERCANTILE EXCHANGE","https://publicreporting.cftc.gov/resource/gpe5-46if.json","lev_money_positions",""),
    "USD/CAD": ("CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE","https://publicreporting.cftc.gov/resource/gpe5-46if.json","lev_money_positions",""),
    "XAU/USD": ("GOLD - COMMODITY EXCHANGE INC.","https://publicreporting.cftc.gov/resource/kh3c-gbw2.json","m_money_positions","_all")
}

@st.cache_data(ttl=21600)
def cot_position(pair):
    contract,url,prefix,suffix=COT[pair]
    d=get_json(url,{"$where":f"market_and_exchange_names = '{contract}'","$order":"report_date_as_yyyy_mm_dd DESC","$limit":1})
    if not d:return "UNKNOWN",50
    row=d[0]
    try: net=int(row.get(f"{prefix}_long{suffix}",0))-int(row.get(f"{prefix}_short{suffix}",0))
    except Exception:return "UNKNOWN",50
    return ("BUY",65) if net>0 else (("SELL",65) if net<0 else ("NEUTRAL",50))

def pivot_direction(df,ohlc=True):
    if df is None or len(df)<2:return "NEUTRAL"
    r=df.iloc[-2]; c=r.close
    h=r.high if ohlc else c*1.005; l=r.low if ohlc else c*.995
    return "BUY" if df.close.iloc[-1]>(h+l+c)/3 else "SELL"

# ===================== STRICT DECISION =====================
def combine(ml,news,tf_dir,tf_conf,cot_dir,cot_conf,pivot, oos):
    mb,ms=ml["buy"],ml["sell"]
    nb,ns=news["buy"],news["sell"]
    tb=tf_conf if tf_dir=="BUY" else (100-tf_conf if tf_dir=="SELL" else 50); ts=100-tb
    cb=cot_conf if cot_dir=="BUY" else (100-cot_conf if cot_dir=="SELL" else 50); cs=100-cb
    pb=62 if pivot=="BUY" else (38 if pivot=="SELL" else 50); ps=100-pb
    buy=mb*.30+nb*.20+tb*.20+cb*.15+pb*.15
    sell=ms*.30+ns*.20+ts*.20+cs*.15+ps*.15
    if oos<55:
        if buy>sell: buy=min(buy,69); sell=100-buy
        else: sell=min(sell,69); buy=100-sell
    decision="BUY" if buy>=70 and buy>sell else ("SELL" if sell>=70 and sell>buy else "WAIT")
    return round(buy,1),round(sell,1),decision

def trade_levels(direction,price,atr,high,low):
    if not all(pd.notna(v) for v in [price,atr,high,low]) or atr<=0:return None,None
    if direction=="BUY":
        sl=low-atr*.3 if 0<price-low<atr*3 else price-atr*1.5
        tp=high-atr*.2 if high>price else price+atr*2
        if tp<=price:tp=price+atr*2
    else:
        sl=high+atr*.3 if 0<high-price<atr*3 else price+atr*1.5
        tp=low+atr*.2 if low<price else price-atr*2
        if tp>=price:tp=price-atr*2
    if abs(tp-price)<abs(price-sl)*.8:
        tp=price+(atr*2 if direction=="BUY" else -atr*2)
    dec=3 if price>=1000 else 5
    return round(sl,dec),round(tp,dec)

# ===================== FIRESTORE PAPER TRADES =====================
def has_open_trade(pair,tf):
    try:return any(True for _ in db.collection("trade_history").where("pair","==",pair).where("timeframe","==",tf).where("outcome","==","OPEN").stream())
    except Exception:return False

def save_trades(rows):
    if not MARKET_OPEN:return
    now=datetime.datetime.utcnow(); stamp=now.strftime("%Y-%m-%d %H:%M UTC")
    for row in rows:
        if row.get("DECISION") not in ["BUY","SELL"]:continue
        pair=row["Pair"]; tf=row.get("Timeframe_Label","Daily")
        if has_open_trade(pair,tf):continue
        doc=f"{now:%Y%m%d_%H%M%S}_{pair.replace('/','')}_{tf}"
        db.collection("trade_history").document(doc).set({"date":now.strftime("%Y-%m-%d"),"pair":pair,"timeframe":tf,"decision":row["DECISION"],"entry_price":row["Price"],"stop_loss":row["Stop Loss"],"take_profit":row["Take Profit"],"outcome":"OPEN","opened_at":stamp,"closed_at":"","buy_conf":row.get("Buy Conf",""),"sell_conf":row.get("Sell Conf","")})

def trade_history():
    out=[]
    try:docs=db.collection("trade_history").stream()
    except Exception:return out
    for doc in docs:
        t=doc.to_dict(); pair=t.get("pair"); outcome=t.get("outcome","OPEN")
        if not pair:continue
        # NEVER close trades from stale weekend prices.
        if outcome=="OPEN" and MARKET_OPEN:
            try:
               if"XAU/USD": df=fetch_gold_history()
else:
    a,b=CURRENCY_PAIRS[pair]; df=fetch_daily_history(a,b)
                price=float(df.close.iloc[-1]) if df is not None else None
                sl=float(t["stop_loss"]); tp=float(t["take_profit"]); d=t["decision"]
                if price is not None:
                    if d=="BUY":
                        if price>=tp:outcome="WIN"
                        elif price<=sl:outcome="LOSS"
                    else:
                        if price<=tp:outcome="WIN"
                        elif price>=sl:outcome="LOSS"
                    if outcome!="OPEN":
                        closed=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                        db.collection("trade_history").document(doc.id).update({"outcome":outcome,"closed_at":closed,"exit_price":round(price,5)})
                        t["closed_at"]=closed;t["exit_price"]=round(price,5)
                    else:t["exit_price"]=round(price,5)
            except Exception:pass
        out.append({"Pair":pair,"Timeframe":t.get("timeframe","Daily"),"Decision":t.get("decision","WAIT"),"Entry":t.get("entry_price","-"),"Exit/Current":t.get("exit_price","-"),"Opened":t.get("opened_at",t.get("date","-")),"Closed":t.get("closed_at","OPEN") if outcome!="OPEN" else "OPEN","Status":outcome})
    return out

# ===================== INTRADAY STRICT =====================

def adx(df, period=14):
    if df is None or len(df) < period * 2:
        return 0

    h = df.high
    l = df.low
    c = df.close

    up = h.diff()
    down = -l.diff()

    plus = up.where((up > down) & (up > 0), 0)
    minus = down.where((down > up) & (down > 0), 0)

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().replace(0, np.nan)

    p = 100 * plus.rolling(period).mean() / atr
    m = 100 * minus.rolling(period).mean() / atr

    dx = 100 * (p - m).abs() / (p + m).replace(0, np.nan)

    value = dx.rolling(period).mean().iloc[-1]

    return float(value) if pd.notna(value) else 0


def intraday_signal(pair, interval, daily_dir, news):

    df = fetch_intraday(
        SYMBOLS[pair],
        interval,
        300
    )

    if df is None or df.empty:
        return "WAIT", None, None, None, 0

    f = build_features(df, True)

    if f is None or f.empty:
        return "WAIT", None, None, None, 0

    price = float(f.close.iloc[-1])
    atrv = float(f.atr.iloc[-1])
    atravg = float(f.atr.tail(20).mean())

    ma5 = f.close.rolling(5).mean().iloc[-1]
    ma20 = f.close.rolling(20).mean().iloc[-1]

    rsi = float(f.rsi.iloc[-1])
    mh = float(f.macd_hist.iloc[-1])

    candidate = "BUY" if ma5 > ma20 else "SELL"

    a = adx(f)

    sweep = False
    retest = False

    if len(f) >= 30:
        recent = f.iloc[-25:-1]
        cur = f.iloc[-1]

        ph = recent.high.iloc[:-1].max()
        pl = recent.low.iloc[:-1].min()

        sweep = (
            (recent.high.iloc[-1] > ph and cur.close < ph)
            or
            (recent.low.iloc[-1] < pl and cur.close > pl)
        )

        avg = f.close.diff().abs().tail(20).mean()

        retest = (
            avg > 0
            and
            (f.close.tail(5).max() - f.close.tail(5).min()) < avg * 2
        )

    if candidate == "BUY":
        fvg = f.high.iloc[-3] < f.low.iloc[-1]
    else:
        fvg = f.low.iloc[-3] > f.high.iloc[-1]

    swingh = float(f.swing_high_20.iloc[-1])
    swingl = float(f.swing_low_20.iloc[-1])

    bos = (
        (candidate == "BUY" and price > swingh * 0.999)
        or
        (candidate == "SELL" and price < swingl * 1.001)
    )

    score = 0

    if daily_dir == candidate:
        score += 15

    if bos:
        score += 15

    if retest or fvg:
        score += 15

    if atrv > atravg:
        score += 10

    if (
        (candidate == "BUY" and mh > 0)
        or
        (candidate == "SELL" and mh < 0)
    ):
        score += 10

    if 45 <= rsi <= 65:
        score += 10

    if a >= 20:
        score += 10

    utc_hour = datetime.datetime.utcnow().hour

    if (
        7 <= utc_hour < 16
        or
        12 <= utc_hour < 21
    ):
        score += 5

    if (
        (candidate == "BUY" and news["buy"] > news["sell"])
        or
        (candidate == "SELL" and news["sell"] > news["buy"])
    ):
        score += 10

    decision = candidate if score >= 70 else "WAIT"

    return (
        decision,
        price,
        atrv,
        (swingh, swingl),
        score
    )


# ===================== BACKTEST - SIDEBAR ONLY =====================
def backtest(pair,interval):
    df=fetch_intraday(SYMBOLS[pair],interval,5000);f=build_features(df,True)
    if f is None or len(f)<200:return None
    d=f.dropna(subset=FEATURES+["target"]).reset_index(drop=True); start=max(120,int(len(d)*.6));wins=losses=skipped=0
    for i in range(start,len(d)-1):
        tr=d.iloc[:i]
        if tr.target.nunique()<2:skipped+=1;continue
        m=RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=4,class_weight="balanced_subsample",random_state=42,n_jobs=-1);m.fit(tr[FEATURES],tr.target);p=m.predict_proba(d.iloc[[i]][FEATURES])[0,1]
        if p>=.55:direction=1
        elif p<=.45:direction=-1
        else:skipped+=1;continue
        ch=(d.close.iloc[i+1]-d.close.iloc[i])/d.close.iloc[i]
        if (direction==1 and ch>0) or (direction==-1 and ch<0):wins+=1
        else:losses+=1
    trades=wins+losses
    return {"pair":pair,"tf":interval,"candles":len(d),"wins":wins,"losses":losses,"trades":trades,"skipped":skipped,"winrate":wins/trades*100 if trades else 0}

with st.sidebar.expander("🧪 Validation",expanded=False):
    st.caption("Backtest is separate from live/paper analysis.")
    if st.button("RUN WALK-FORWARD BACKTEST",use_container_width=True):
        results=[]
        with st.spinner("Running historical validation..."):
            for pair in PAIRS:
                for interval,label in [("1h","1H"),("4h","4H")]:
                    x=backtest(pair,interval)
                    results.append((pair,label,x))
        st.session_state.backtest_results=results
    if st.session_state.backtest_results:
        st.write("### Results")
        for pair,label,x in st.session_state.backtest_results:
            if x is None:st.error(f"{pair} {label}: data unavailable")
            else:st.write(f"**{pair} {label}: {x['winrate']:.2f}%** — {x['wins']}W / {x['losses']}L — {x['trades']} trades")

# ===================== ADMIN =====================
if st.session_state.user_email==ADMIN_EMAIL:
    with st.sidebar.expander("🔑 Admin Panel"):
        try:
            for p in db.collection("pending_users").where("approved","==",False).stream():
                d=p.to_dict();email=d.get("email","");st.write(f"**{email}** | Age: {d.get('age','-')} | Location: {d.get('location','-')}")
                if st.button(f"Approve {email}",key=f"approve_{p.id}"):
                    db.collection("pending_users").document(p.id).update({"approved":True});st.rerun()
        except Exception as e:st.warning(f"Admin panel unavailable: {str(e)[:80]}")

# ===================== LIVE ANALYSIS =====================
st.title("📈 YonKing - Forex Analysis Dashboard")
st.caption("AI-assisted price prediction, news percentages, multi-timeframe confirmation and paper-trading validation.")

if MARKET_OPEN:st.success("🟢 Market open — strict analysis active.")
else:st.warning("🔴 Market closed — no new paper trades; stale weekend prices cannot close trades.")

def run_analysis():
    news,headlines=get_news();rows=[];daily_dir={}
    for pair in PAIRS:
        ni=news[pair]
        try:
            if pair=="XAU/USD":raw=fetch_gold_history();ohlc=False
            else:a,b=CURRENCY_PAIRS[pair];raw=fetch_daily_history(a,b);ohlc=True
            if raw is None or len(raw)<100:
                rows.append({"Pair":pair,"Price":"-","Price Model":"DATA ERROR","News":f"BUY {ni['buy']:.1f}% | SELL {ni['sell']:.1f}%","Timeframe":"DATA ERROR","COT":"DATA ERROR","Pivot":"DATA ERROR","Buy Conf":"0%","Sell Conf":"0%","DECISION":"WAIT","Stop Loss":"-","Take Profit":"-","Timeframe_Label":"Daily"});continue
            f=build_features(raw,ohlc);ml=train_model(f)
            if ml is None:
                rows.append({"Pair":pair,"Price":round(float(raw.close.iloc[-1]),5),"Price Model":"WAIT (validation unavailable)","News":f"BUY {ni['buy']:.1f}% | SELL {ni['sell']:.1f}%","Timeframe":"WAIT","COT":"UNKNOWN","Pivot":pivot_direction(raw,ohlc),"Buy Conf":"0%","Sell Conf":"0%","DECISION":"WAIT","Stop Loss":"-","Take Profit":"-","Timeframe_Label":"Daily"});continue
            daily_dir[pair]=ml["direction"];tfdir,tfconf,tfs=timeframe_evidence(pair);cotdir,cotconf=cot_position(pair);pivot=pivot_direction(raw,ohlc);bc,sc,decision=combine(ml,ni,tfdir,tfconf,cotdir,cotconf,pivot,ml["oos"]);price=float(raw.close.iloc[-1]);sl=tp="-"
            if decision in ["BUY","SELL"]:
                sl,tp=trade_levels(decision,price,float(f.atr.iloc[-1]),float(f.swing_high_20.iloc[-1]),float(f.swing_low_20.iloc[-1]))
            rows.append({"Pair":pair,"Price":round(price,5),"Price Model":f"{ml['direction']} | OOS {ml['oos']:.1f}%","News":f"BUY {ni['buy']:.1f}% | SELL {ni['sell']:.1f}%","Timeframe":f"{tfdir} | 4H:{tfs['4H']} 1H:{tfs['1H']} 30M:{tfs['30M']} 15M:{tfs['15M']}","COT":cotdir,"Pivot":pivot,"Buy Conf":f"{bc:.1f}%","Sell Conf":f"{sc:.1f}%","DECISION":decision,"Stop Loss":sl,"Take Profit":tp,"Timeframe_Label":"Daily"})
        except Exception:
            rows.append({"Pair":pair,"Price":"-","Price Model":"DATA ERROR","News":f"BUY {ni['buy']:.1f}% | SELL {ni['sell']:.1f}%","Timeframe":"DATA ERROR","COT":"ERROR","Pivot":"ERROR","Buy Conf":"0%","Sell Conf":"0%","DECISION":"WAIT","Stop Loss":"-","Take Profit":"-","Timeframe_Label":"Daily"})
    # Intraday additions; XAU/USD remains included.
    plan={"1H":["USD/JPY","USD/CAD","XAU/USD"],"4H":["GBP/USD"]}
    for label,interval in [("1H","1h"),("4H","4h")]:
        for pair in plan[label]:
            ni=news[pair];direction,price,atrv,levels,score=intraday_signal(pair,interval,daily_dir.get(pair,"WAIT"),ni);sl=tp="-"
            if direction in ["BUY","SELL"] and price and atrv and levels:sl,tp=trade_levels(direction,price,atrv,levels[0],levels[1])
            rows.append({"Pair":pair,"Price":round(price,5) if price else "-","Price Model":f"Intraday strict score {score}/100","News":f"BUY {ni['buy']:.1f}% | SELL {ni['sell']:.1f}%","Timeframe":label,"COT":"-","Pivot":"-","Buy Conf":f"{score}%" if direction=="BUY" else "0%","Sell Conf":f"{score}%" if direction=="SELL" else "0%","DECISION":direction,"Stop Loss":sl,"Take Profit":tp,"Timeframe_Label":label})
    st.session_state.results_table=rows;st.session_state.headlines=headlines;st.session_state.news_data=news
    save_trades(rows)

if st.session_state.results_table is not None:
    st.divider()
    st.subheader("🎯 YonKing Current Decisions")
    st.dataframe(pd.DataFrame(st.session_state.results_table), use_container_width=True)

    st.info(
        "Only decisions at or above the 70% evidence threshold are eligible "
        "for BUY/SELL. 70% is not a guaranteed future win rate."
    )

    st.divider()
    st.subheader("📰 News Evidence")

    for pair in PAIRS:
        n = st.session_state.news_data.get(
            pair, {"buy": 50, "sell": 50}
        )
        st.write(
            f"**{pair}: BUY {n['buy']:.1f}% | SELL {n['sell']:.1f}%**"
        )

    st.divider()
    st.subheader("📊 YonKing's Calculated Chart View")
    with st.spinner("Running YonKing analysis..."):run_analysis()
if st.button("🔄 Refresh Analysis",use_container_width=True):
    with st.spinner("Refreshing..."):run_analysis()

# ===================== RESULTS =====================
st.divider();st.subheader("📜 Trade History & Results")
history=trade_history()
if history:
    hdf=pd.DataFrame(history)
    def status_style(row):
        if row["Status"]=="WIN":return ["background-color:#1a4d2e"]*len(row)
        if row["Status"]=="LOSS":return ["background-color:#5c1a1a"]*len(row)
        return [""]*len(row)
    st.dataframe(hdf.style.apply(status_style,axis=1),use_container_width=True)
else:st.write("No paper-trade history yet.")
if st.session_state.results_table is not None:
    st.divider();st.subheader("🎯 YonKing Current Decisions")
    st.dataframe(pd.DataFrame(st.session_state.results_table),use_container_width=True)
    st.info("Only decisions at or above the 70% evidence threshold are eligible for BUY/SELL. 70% is not a guaranteed future win rate.")
    st.divider();st.subheader("📰 News Evidence")
    for pair in PAIRS:
        n=st.session_state.news_data.get(pair,{"buy":50,"sell":50});st.write(f"**{pair}: BUY {n['buy']:.1f}% | SELL {n['sell']:.1f}%**")

    st.divider();st.subheader("📊 YonKing's Calculated Chart View")
    chart_pairs={"USD/JPY":("USD","JPY",True),"GBP/USD":("GBP","USD",True),"USD/CAD":("USD","CAD",True),"XAU/USD":(None,None,False)}
    selected=st.selectbox("Select a pair to view its chart:",list(chart_pairs))
    a,b,ohlc=chart_pairs[selected];chart=fetch_daily_history(a,b) if ohlc else fetch_gold_history()
    if chart is None or chart.empty:st.error(f"No chart data available for {selected}.")
    else:
        chart=chart.tail(60).reset_index(drop=True);fig=go.Figure()
        if ohlc:
            fig.add_trace(go.Candlestick(x=chart.date,open=chart.open,high=chart.high,low=chart.low,close=chart.close,name=selected));hi=chart.high.rolling(20,min_periods=1).max().iloc[-1];lo=chart.low.rolling(20,min_periods=1).min().iloc[-1]
        else:
            fig.add_trace(go.Scatter(x=chart.date,y=chart.close,mode="lines",name=selected));hi=chart.close.rolling(20,min_periods=1).max().iloc[-1];lo=chart.close.rolling(20,min_periods=1).min().iloc[-1]
        fig.add_hline(y=hi,line_dash="dash",annotation_text=f"Resistance: {round(hi,4)}");fig.add_hline(y=lo,line_dash="dash",annotation_text=f"Support: {round(lo,4)}")
        if len(chart)>=2:
            p=chart.iloc[-2];c=p.close;h=p.high if ohlc else c*1.005;l=p.low if ohlc else c*.995;pv=(h+l+c)/3;fig.add_hline(y=pv,line_dash="dot",annotation_text=f"Pivot: {round(pv,4)}")
        fig.update_layout(title=f"{selected} - Price with Calculated Levels",template="plotly_dark",height=500,xaxis_rangeslider_visible=False);st.plotly_chart(fig,use_container_width=True)
    with st.expander("Headlines used for news analysis"):
        for h in st.session_state.headlines:st.write(f"- {h}")
