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

# ============================================================
# YonKing - strict paper-trading / validation version
# ============================================================

st.set_page_config(page_title="YonKing", page_icon="📈", layout="wide")

# -------------------------
# Secrets
# -------------------------
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
GROQ_KEY = st.secrets["GROQ_KEY"]
ALPHA_KEY = st.secrets["ALPHA_KEY"]
TWELVE_KEY = st.secrets["TWELVE_KEY"]

# -------------------------
# Firebase
# -------------------------
if not firebase_admin._apps:
    firebase_key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

ADMIN_EMAIL = "obidiharry@gmail.com"

# -------------------------
# Market/session state
# -------------------------
def is_market_closed():
    now_utc = datetime.datetime.utcnow()
    weekday = now_utc.weekday()
    hour = now_utc.hour

    # Saturday
    if weekday == 5:
        return True

    # Sunday before approximately 21:00 UTC
    if weekday == 6 and hour < 21:
        return True

    # Friday after approximately 21:00 UTC
    if weekday == 4 and hour >= 21:
        return True

    return False


market_closed = is_market_closed()

# ============================================================
# Authentication
# ============================================================

def signup_user(email, password, phone, age, location):
    try:
        user = firebase_auth.create_user(
            email=email,
            password=password,
            phone_number=phone
        )

        db.collection("pending_users").document(user.uid).set({
            "email": email,
            "phone": phone,
            "age": age,
            "location": location,
            "approved": False,
            "signup_date": datetime.date.today().strftime("%Y-%m-%d")
        })

        return True, "Account created! Waiting for approval before you can log in."

    except Exception as e:
        return False, str(e)


def check_approval(email):
    users = (
        db.collection("pending_users")
        .where("email", "==", email)
        .stream()
    )

    for u in users:
        data = u.to_dict()
        return data.get("approved", False)

    return False


def login_user(email, password):
    try:
        # Firebase Admin SDK does not verify passwords.
        # This function preserves the existing YonKing login flow.
        user = firebase_auth.get_user_by_email(email)

        if email == ADMIN_EMAIL:
            return True, "Login successful!"

        if not check_approval(email):
            return False, "Your account is still pending approval."

        return True, "Login successful!"

    except Exception:
        return False, "Invalid email or account not found."


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = None


if not st.session_state.logged_in:
    st.title("📈 Welcome to YonKing")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login"):
            success, message = login_user(
                login_email,
                login_password
            )

            if success:
                st.session_state.logged_in = True
                st.session_state.user_email = login_email
                st.rerun()
            else:
                st.error(message)

    with tab2:
        signup_email = st.text_input(
            "Email",
            key="signup_email"
        )

        signup_password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        signup_phone = st.text_input(
            "Phone Number (e.g. +2348012345678)",
            key="signup_phone"
        )

        signup_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            key="signup_age"
        )

        signup_location = st.text_input(
            "Location (City, Country)",
            key="signup_location"
        )

        if st.button("Sign Up"):
            if (
                signup_email
                and signup_password
                and signup_phone
                and signup_location
            ):
                success, message = signup_user(
                    signup_email,
                    signup_password,
                    signup_phone,
                    signup_age,
                    signup_location
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please fill in all fields.")

    st.stop()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.write(
    f"Logged in as: {st.session_state.user_email}"
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.rerun()


if market_closed:
    st.sidebar.warning(
        "⏸️ Market closed. New paper trades are disabled."
    )
else:
    st.sidebar.success(
        "🟢 Market open. Strict analysis enabled."
    )


# ============================================================
# Symbols
# ============================================================

twelvedata_symbols = {
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "USD/CAD": "USD/CAD",
    "XAU/USD": "XAU/USD"
}


# ============================================================
# Basic helpers
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def normalize_direction(direction):
    if direction in ("BUY", "UP"):
        return "BUY"

    if direction in ("SELL", "DOWN"):
        return "SELL"

    return "NEUTRAL"


# ============================================================
# Intraday data
# ============================================================

@st.cache_data(ttl=300)
def fetch_intraday(symbol, interval):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": TWELVE_KEY
    }

    try:
        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=20
        )

        data = response.json()

        if data.get("status") != "ok":
            return None

        if "values" not in data:
            return None

        df = pd.DataFrame(data["values"])

        if df.empty:
            return None

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["datetime"],
                errors="coerce"
            )

        df = df.iloc[::-1].reset_index(drop=True)

        return df.dropna(
            subset=["close"]
        ).reset_index(drop=True)

    except Exception:
        return None


# ============================================================
# Daily data
# ============================================================

@st.cache_data(ttl=1800)
def fetch_daily_history(from_sym, to_sym):
    url = "https://www.alphavantage.co/query"

    params = {
        "function": "FX_DAILY",
        "from_symbol": from_sym,
        "to_symbol": to_sym,
        "apikey": ALPHA_KEY,
        "outputsize": "full"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        data = response.json()

        key = "Time Series FX (Daily)"

        if key not in data:
            return None

        rows = []

        for date, values in data[key].items():
            rows.append({
                "date": date,
                "open": safe_float(values.get("1. open")),
                "high": safe_float(values.get("2. high")),
                "low": safe_float(values.get("3. low")),
                "close": safe_float(values.get("4. close"))
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return None

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["date", "close"]
        )

        return df.sort_values(
            "date"
        ).reset_index(drop=True)

    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_gold_history():
    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GOLD_SILVER_HISTORY",
        "symbol": "GOLD",
        "interval": "daily",
        "apikey": ALPHA_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        data = response.json()

        if "data" not in data:
            return None

        rows = []

        for item in data["data"]:
            rows.append({
                "date": item.get("date"),
                "close": safe_float(item.get("price"))
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return None

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["date", "close"]
        )

        return df.sort_values(
            "date"
        ).reset_index(drop=True)

    except Exception:
        return None


# ============================================================
# Feature engineering
# ============================================================

def build_features(df, has_ohlc=True):
    df = df.copy()

    if df is None or df.empty:
        return pd.DataFrame()

    df["next_close"] = df["close"].shift(-1)

    # Last row cannot be evaluated because the next candle is unknown.
    df["target"] = (
        df["next_close"] > df["close"]
    ).astype(int)

    df["change_1d"] = df["close"].diff(1)
    df["change_3d"] = df["close"].diff(3)
    df["change_5d"] = df["close"].diff(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_diff"] = df["ma_5"] - df["ma_20"]

    delta = df["close"].diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    if has_ohlc:
        df["swing_high_20"] = (
            df["high"]
            .shift(1)
            .rolling(20)
            .max()
        )

        df["swing_low_20"] = (
            df["low"]
            .shift(1)
            .rolling(20)
            .min()
        )

    else:
        df["swing_high_20"] = (
            df["close"]
            .shift(1)
            .rolling(20)
            .max()
        )

        df["swing_low_20"] = (
            df["close"]
            .shift(1)
            .rolling(20)
            .min()
        )

    df["dist_to_swing_high"] = (
        (df["swing_high_20"] - df["close"])
        / df["close"]
    )

    df["dist_to_swing_low"] = (
        (df["close"] - df["swing_low_20"])
        / df["close"]
    )

    df["trend_slope"] = (
        df["close"]
        .rolling(10)
        .apply(
            lambda s: np.polyfit(
                np.arange(len(s)),
                s,
                1
            )[0]
            if len(s) >= 2 else 0,
            raw=False
        )
    )

    df["fib_range"] = (
        df["swing_high_20"]
        - df["swing_low_20"]
    )

    df["fib_position"] = (
        (df["close"] - df["swing_low_20"])
        / df["fib_range"].replace(0, np.nan)
    )

    df["ema_12"] = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["ema_26"] = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = (
        df["ema_12"] - df["ema_26"]
    )

    df["macd_signal"] = (
        df["macd"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    df["bb_mid"] = (
        df["close"].rolling(20).mean()
    )

    df["bb_std"] = (
        df["close"].rolling(20).std()
    )

    df["bb_upper"] = (
        df["bb_mid"] + 2 * df["bb_std"]
    )

    df["bb_lower"] = (
        df["bb_mid"] - 2 * df["bb_std"]
    )

    df["bb_position"] = (
        (df["close"] - df["bb_lower"])
        / (
            df["bb_upper"]
            - df["bb_lower"]
        ).replace(0, np.nan)
    )

    if has_ohlc:
        prev_close = df["close"].shift(1)

        tr1 = df["high"] - df["low"]
        tr2 = abs(
            df["high"] - prev_close
        )
        tr3 = abs(
            df["low"] - prev_close
        )

        df["true_range"] = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        df["atr"] = (
            df["true_range"]
            .rolling(14)
            .mean()
        )

    else:
        df["atr"] = (
            df["close"]
            .diff()
            .abs()
            .rolling(14)
            .mean()
        )

    return df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna().reset_index(drop=True)


features = [
    "change_1d",
    "change_3d",
    "change_5d",
    "ma_diff",
    "rsi",
    "dist_to_swing_high",
    "dist_to_swing_low",
    "trend_slope",
    "fib_position",
    "macd",
    "macd_hist",
    "bb_position"
]


# ============================================================
# Walk-forward ML validation
# ============================================================

def walk_forward_backtest(
    df,
    train_fraction=0.60,
    probability_threshold=0.70,
    retrain_every=25
):
    """
    Strict chronological validation.

    The model never trains on future candles.
    A trade is counted only when:
        probability >= threshold -> BUY
        probability <= 1-threshold -> SELL
    """

    if df is None or len(df) < 250:
        return None

    data = df.copy().reset_index(drop=True)

    # Remove the final row because its target is not known.
    data = data.iloc[:-1].copy()

    start_test = max(
        int(len(data) * train_fraction),
        100
    )

    if start_test >= len(data) - 5:
        return None

    model = None
    wins = 0
    losses = 0
    skipped = 0
    predictions = []

    for i in range(start_test, len(data)):
        if (
            model is None
            or (i - start_test) % retrain_every == 0
        ):
            train = data.iloc[:i].copy()

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                train[features],
                train["target"]
            )

        row = data.iloc[[i]]

        probability_up = float(
            model.predict_proba(
                row[features]
            )[0][1]
        )

        if probability_up >= probability_threshold:
            direction = "BUY"

        elif probability_up <= (
            1 - probability_threshold
        ):
            direction = "SELL"

        else:
            direction = "WAIT"

        actual_up = int(
            data["target"].iloc[i]
        )

        if direction == "WAIT":
            skipped += 1
            continue

        if (
            direction == "BUY"
            and actual_up == 1
        ):
            wins += 1

        elif (
            direction == "SELL"
            and actual_up == 0
        ):
            wins += 1

        else:
            losses += 1

        predictions.append({
            "index": i,
            "prob_up": probability_up,
            "direction": direction,
            "actual": actual_up
        })

    total_trades = wins + losses

    win_rate = (
        wins / total_trades * 100
        if total_trades > 0
        else 0
    )

    coverage = (
        total_trades
        / (total_trades + skipped)
        * 100
        if total_trades + skipped > 0
        else 0
    )

    return {
        "win_rate": round(win_rate, 2),
        "wins": wins,
        "losses": losses,
        "trades": total_trades,
        "skipped": skipped,
        "coverage": round(coverage, 2),
        "predictions": predictions
    }


# ============================================================
# Legacy-compatible model analysis for live decisions
# ============================================================

def analyze_price_model(df):
    if df is None or len(df) < 100:
        return 0, "WAIT", None, 0

    clean = df.copy()

    # Remove final row from training/evaluation because target
    # is based on a future candle.
    model_df = clean.iloc[:-1].copy()

    if len(model_df) < 100:
        return 0, "WAIT", None, 0

    split_point = int(
        len(model_df) * 0.80
    )

    train_df = model_df.iloc[:split_point]
    test_df = model_df.iloc[split_point:].copy()

    if (
        len(train_df) < 50
        or len(test_df) < 20
    ):
        return 0, "WAIT", None, 0

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_df[features],
        train_df["target"]
    )

    test_df["prob_up"] = model.predict_proba(
        test_df[features]
    )[:, 1]

    # Strict validation threshold.
    threshold = 0.70

    wins = 0
    losses = 0

    for _, row in test_df.iterrows():
        p = row["prob_up"]

        if p >= threshold:
            direction = 1

        elif p <= 1 - threshold:
            direction = 0

        else:
            continue

        if direction == int(row["target"]):
            wins += 1
        else:
            losses += 1

    total = wins + losses

    validation_win_rate = (
        wins / total * 100
        if total > 0
        else 0
    )

    # Retrain using all known historical rows,
    # but never use the final unknown target.
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(
        model_df[features],
        model_df["target"]
    )

    latest = clean.iloc[[-1]]

    probability_up = float(
        final_model.predict_proba(
            latest[features]
        )[0][1]
    )

    if probability_up >= 0.70:
        direction = "BUY"

    elif probability_up <= 0.30:
        direction = "SELL"

    else:
        direction = "WAIT"

    return (
        round(validation_win_rate, 2),
        direction,
        float(clean["close"].iloc[-1]),
        round(probability_up * 100, 2)
    )


# ============================================================
# News
# ============================================================

@st.cache_data(ttl=1800)
def get_news_sentiment():
    params = {
        "q": '"USD/JPY" OR "Bank of Japan" OR "Japanese yen" OR '
              '"Federal Reserve" OR "interest rate" OR "safe haven" '
              'OR oil OR geopolitical',
        "language": "en",
        "sortBy": "publishedAt",
        "domains": "reuters.com,bloomberg.com,forexlive.com,investing.com,cnbc.com",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=20
        )

        data = response.json()

        if data.get("status") != "ok":
            return "NEUTRAL", []

        headlines = [
            article.get("title", "")
            for article in data.get("articles", [])
            if article.get("title")
        ]

        if not headlines:
            return "NEUTRAL", []

        prompt = (
            "Analyze these financial headlines.\n\n"
            + "\n".join(
                f"- {headline}"
                for headline in headlines
            )
            + "\n\n"
            "Respond in EXACTLY this format:\n"
            "BIAS: [BULLISH_USD or BEARISH_USD or NEUTRAL]"
        )

        gr = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=30
        ).json()

        bias = "NEUTRAL"

        if "choices" in gr:
            content = (
                gr["choices"][0]["message"]["content"]
            )

            for line in content.split("\n"):
                if line.startswith("BIAS:"):
                    candidate = (
                        line
                        .replace("BIAS:", "")
                        .strip()
                    )

                    if candidate in (
                        "BULLISH_USD",
                        "BEARISH_USD",
                        "NEUTRAL"
                    ):
                        bias = candidate

        return bias, headlines

    except Exception:
        return "NEUTRAL", []


pair_structure = {
    "USD/JPY": "USD",
    "GBP/USD": "GBP",
    "USD/CAD": "USD",
    "XAU/USD": "XAU"
}


def get_news_direction(pair, news_bias):
    usd_base = pair_structure[pair] == "USD"

    if news_bias == "BEARISH_USD":
        return "SELL" if usd_base else "BUY"

    if news_bias == "BULLISH_USD":
        return "BUY" if usd_base else "SELL"

    return "NEUTRAL"


# ============================================================
# Technical direction
# ============================================================

def get_tf_direction(df):
    if df is None or len(df) < 30:
        return "UNKNOWN"

    ma = df["close"].rolling(
        20
    ).mean().iloc[-1]

    price = df["close"].iloc[-1]

    if pd.isna(ma):
        return "UNKNOWN"

    if price > ma:
        return "UP"

    if price < ma:
        return "DOWN"

    return "FLAT"


def get_timeframe_directions(pair):
    symbol = twelvedata_symbols[pair]

    directions = {}

    for tf in [
        ("4H", "4h"),
        ("1H", "1h"),
        ("30M", "30min"),
        ("15M", "15min")
    ]:
        label, interval = tf

        df = fetch_intraday(
            symbol,
            interval
        )

        directions[label] = get_tf_direction(df)

    return directions


def get_timeframe_direction(pair):
    directions = get_timeframe_directions(pair)

    votes = []

    for direction in directions.values():
        if direction == "UP":
            votes.append("BUY")
        elif direction == "DOWN":
            votes.append("SELL")

    if not votes:
        return "NEUTRAL"

    buy = votes.count("BUY")
    sell = votes.count("SELL")

    if buy > sell:
        return "BUY"

    if sell > buy:
        return "SELL"

    return "NEUTRAL"


# ============================================================
# COT
# ============================================================

@st.cache_data(ttl=21600)
def get_cot_positioning(
    contract_name,
    dataset_url,
    field_prefix,
    suffix=""
):
    params = {
        "$where": (
            f"market_and_exchange_names = "
            f"'{contract_name}'"
        ),
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 1
    }

    try:
        response = requests.get(
            dataset_url,
            params=params,
            timeout=20
        )

        data = response.json()

        if not data:
            return "NEUTRAL"

        row = data[0]

        long_key = (
            f"{field_prefix}_long{suffix}"
        )

        short_key = (
            f"{field_prefix}_short{suffix}"
        )

        net = (
            int(row.get(long_key, 0))
            - int(row.get(short_key, 0))
        )

        if net > 0:
            return "BUY"

        if net < 0:
            return "SELL"

        return "NEUTRAL"

    except Exception:
        return "NEUTRAL"


TFF_URL = (
    "https://publicreporting.cftc.gov/"
    "resource/gpe5-46if.json"
)

DISAGG_URL = (
    "https://publicreporting.cftc.gov/"
    "resource/kh3c-gbw2.json"
)

cot_contracts = {
    "USD/JPY": (
        "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
        TFF_URL,
        "lev_money_positions",
        ""
    ),

    "GBP/USD": (
        "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
        TFF_URL,
        "lev_money_positions",
        ""
    ),

    "USD/CAD": (
        "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        TFF_URL,
        "lev_money_positions",
        ""
    ),

    "XAU/USD": (
        "GOLD - COMMODITY EXCHANGE INC.",
        DISAGG_URL,
        "m_money_positions",
        "_all"
    )
}


# ============================================================
# Market structure
# ============================================================

def calculate_adx(df, period=14):
    high = (
        df["high"]
        if "high" in df.columns
        else df["close"]
    )

    low = (
        df["low"]
        if "low" in df.columns
        else df["close"]
    )

    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)

    # Prevent simultaneous +DM and -DM.
    plus_dm = plus_dm.where(
        plus_dm > minus_dm,
        0
    )

    minus_dm = minus_dm.where(
        minus_dm > plus_dm,
        0
    )

    tr1 = high - low
    tr2 = abs(
        high - close.shift(1)
    )
    tr3 = abs(
        low - close.shift(1)
    )

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = (
        100
        * plus_dm.rolling(period).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.rolling(period).mean()
        / atr.replace(0, np.nan)
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * abs(plus_di - minus_di)
        / denominator
    )

    adx = dx.rolling(period).mean()

    value = adx.iloc[-1]

    return (
        float(value)
        if not pd.isna(value)
        else 0.0
    )


def get_pivot_direction(df, has_ohlc=True):
    if df is None or len(df) < 2:
        return "NEUTRAL"

    y = df.iloc[-2]

    close = y["close"]

    if has_ohlc:
        high = y["high"]
        low = y["low"]
    else:
        high = close * 1.005
        low = close * 0.995

    pivot = (
        high + low + close
    ) / 3

    current = df["close"].iloc[-1]

    if current > pivot:
        return "BUY"

    if current < pivot:
        return "SELL"

    return "NEUTRAL"


def detect_liquidity_sweep_and_retest(
    df,
    has_ohlc=True
):
    lookback = 20

    if df is None or len(df) < lookback + 5:
        return False, False

    recent = df.iloc[
        -lookback - 5:-1
    ].copy()

    current = df.iloc[-1]

    if has_ohlc:
        prior_high = (
            recent["high"]
            .iloc[:-1]
            .max()
        )

        prior_low = (
            recent["low"]
            .iloc[:-1]
            .min()
        )

        swept_high = (
            recent["high"].iloc[-1]
            > prior_high
            and current["close"]
            < prior_high
        )

        swept_low = (
            recent["low"].iloc[-1]
            < prior_low
            and current["close"]
            > prior_low
        )

    else:
        prior_high = (
            recent["close"]
            .iloc[:-1]
            .max()
        )

        prior_low = (
            recent["close"]
            .iloc[:-1]
            .min()
        )

        swept_high = (
            recent["close"].iloc[-1]
            > prior_high
            and current["close"]
            < prior_high
        )

        swept_low = (
            recent["close"].iloc[-1]
            < prior_low
            and current["close"]
            > prior_low
        )

    liquidity_sweep = (
        swept_high or swept_low
    )

    price_range = (
        df["close"]
        .tail(5)
        .max()
        - df["close"]
        .tail(5)
        .min()
    )

    avg_range = (
        df["close"]
        .diff()
        .abs()
        .tail(20)
        .mean()
    )

    retest_happening = (
        price_range < avg_range * 2
        if avg_range > 0
        else False
    )

    return (
        liquidity_sweep,
        retest_happening
    )


def detect_fair_value_gap(
    df,
    direction,
    has_ohlc=True
):
    if df is None or len(df) < 3:
        return False

    if has_ohlc:
        c1_high = df["high"].iloc[-3]
        c1_low = df["low"].iloc[-3]
        c3_high = df["high"].iloc[-1]
        c3_low = df["low"].iloc[-1]

    else:
        c1_high = c1_low = (
            df["close"].iloc[-3]
        )

        c3_high = c3_low = (
            df["close"].iloc[-1]
        )

    if direction == "BUY":
        return c3_low > c1_high

    if direction == "SELL":
        return c3_high < c1_low

    return False


def is_good_trading_session():
    hour = datetime.datetime.utcnow().hour

    london = 7 <= hour < 16
    new_york = 12 <= hour < 21

    return london or new_york


# ============================================================
# Confidence scoring
# ============================================================

def calculate_strict_confidence(
    direction,
    model_confidence,
    model_validation_win_rate,
    daily_direction,
    tf_4h,
    tf_1h,
    tf_30m,
    tf_15m,
    news_direction,
    cot_direction,
    pivot_direction,
    adx,
    rsi,
    macd_agrees,
    atr_expanding,
    liquidity_sweep,
    retest,
    fvg,
    session_ok
):
    """
    Evidence score, not a guaranteed probability.

    The score is deliberately conservative.
    A trade requires >= 70 AND the hard filters below.
    """

    score = 0
    evidence = []

    # ML probability contribution.
    if model_confidence >= 70:
        score += 20
        evidence.append("ML probability >= 70%")

    elif model_confidence >= 65:
        score += 10

    # Historical validation quality.
    if model_validation_win_rate >= 65:
        score += 15
        evidence.append("ML validation >= 65%")

    elif model_validation_win_rate >= 60:
        score += 8

    # Daily direction.
    if daily_direction == direction:
        score += 10
        evidence.append("Daily agrees")

    # Higher timeframes get more weight.
    if tf_4h == direction:
        score += 12
        evidence.append("4H agrees")

    if tf_1h == direction:
        score += 10
        evidence.append("1H agrees")

    if tf_30m == direction:
        score += 5

    if tf_15m == direction:
        score += 3

    # News.
    if news_direction == direction:
        score += 8
        evidence.append("News agrees")

    # COT.
    if cot_direction == direction:
        score += 5

    # Pivot.
    if pivot_direction == direction:
        score += 3

    # Momentum / trend.
    if adx >= 25:
        score += 5
        evidence.append("ADX >= 25")

    if (
        direction == "BUY"
        and 45 <= rsi <= 65
    ):
        score += 3

    elif (
        direction == "SELL"
        and 35 <= rsi <= 55
    ):
        score += 3

    if macd_agrees:
        score += 4

    if atr_expanding:
        score += 3

    # Structure.
    if liquidity_sweep:
        score += 4

    if retest:
        score += 4

    if fvg:
        score += 3

    if session_ok:
        score += 3

    return min(score, 100), evidence


def get_confidence_label(score):
    if score < 60:
        return "NO TRADE"

    if score < 70:
        return "FILTERED"

    if score < 80:
        return "QUALIFIED"

    if score < 90:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# Intraday signal
# ============================================================

def analyze_intraday_signal(
    pair,
    interval,
    daily_dir,
    news_dir,
    cot_dir,
    pivot_dir
):
    try:
        symbol = twelvedata_symbols[pair]

        df = fetch_intraday(
            symbol,
            interval
        )

        if df is None or len(df) < 60:
            return (
                "WAIT",
                None,
                None,
                None,
                0,
                {}
            )

        df = df.copy()

        df["ma_5"] = (
            df["close"]
            .rolling(5)
            .mean()
        )

        df["ma_20"] = (
            df["close"]
            .rolling(20)
            .mean()
        )

        delta = df["close"].diff()

        gain = delta.where(
            delta > 0,
            0
        )

        loss = -delta.where(
            delta < 0,
            0
        )

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["rsi"] = (
            100 - 100 / (1 + rs)
        )

        df["ema_12"] = (
            df["close"]
            .ewm(
                span=12,
                adjust=False
            )
            .mean()
        )

        df["ema_26"] = (
            df["close"]
            .ewm(
                span=26,
                adjust=False
            )
            .mean()
        )

        df["macd"] = (
            df["ema_12"]
            - df["ema_26"]
        )

        df["macd_signal_line"] = (
            df["macd"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        df["macd_hist"] = (
            df["macd"]
            - df["macd_signal_line"]
        )

        df["tr"] = (
            df["close"]
            .diff()
            .abs()
        )

        df["atr"] = (
            df["tr"]
            .rolling(14)
            .mean()
        )

        current_price = float(
            df["close"].iloc[-1]
        )

        ma5 = df["ma_5"].iloc[-1]
        ma20 = df["ma_20"].iloc[-1]
        rsi = float(df["rsi"].iloc[-1])
        macd_hist = float(
            df["macd_hist"].iloc[-1]
        )

        atr_now = float(
            df["atr"].iloc[-1]
        )

        atr_avg = float(
            df["atr"].tail(20).mean()
        )

        if any(
            pd.isna(x)
            for x in [
                ma5,
                ma20,
                rsi,
                atr_now,
                atr_avg
            ]
        ):
            return (
                "WAIT",
                current_price,
                atr_now,
                None,
                0,
                {}
            )

        candidate_dir = (
            "BUY"
            if ma5 > ma20
            else "SELL"
        )

        adx = calculate_adx(df)

        (
            liquidity_sweep,
            retest
        ) = detect_liquidity_sweep_and_retest(
            df,
            has_ohlc=(
                "high" in df.columns
                and "low" in df.columns
            )
        )

        fvg = detect_fair_value_gap(
            df,
            candidate_dir,
            has_ohlc=(
                "high" in df.columns
                and "low" in df.columns
            )
        )

        session_ok = is_good_trading_session()

        atr_expanding = (
            atr_now > atr_avg
        )

        macd_agrees = (
            macd_hist > 0
            if candidate_dir == "BUY"
            else macd_hist < 0
        )

        swing_high = (
            df["close"]
            .rolling(20)
            .max()
            .iloc[-1]
        )

        swing_low = (
            df["close"]
            .rolling(20)
            .min()
            .iloc[-1]
        )

        previous_swing_high = (
            df["close"]
            .rolling(20)
            .max()
            .shift(1)
            .iloc[-1]
        )

        previous_swing_low = (
            df["close"]
            .rolling(20)
            .min()
            .shift(1)
            .iloc[-1]
        )

        bos_detected = (
            candidate_dir == "BUY"
            and current_price > previous_swing_high
        ) or (
            candidate_dir == "SELL"
            and current_price < previous_swing_low
        )

        # Intraday ML probability is estimated from the same
        # technical evidence rather than pretending the 4H/1H
        # candidate is independent evidence.
        returns = df["close"].pct_change()

        recent_up = (
            returns.tail(20)
            .mean()
            > 0
        )

        model_confidence = (
            72
            if (
                candidate_dir == "BUY"
                and recent_up
            ) or (
                candidate_dir == "SELL"
                and not recent_up
            )
            else 55
        )

        # Get real independent timeframe directions.
        tf_dirs = get_timeframe_directions(pair)

        tf_4h = normalize_direction(
            tf_dirs.get("4H")
        )

        tf_1h = normalize_direction(
            tf_dirs.get("1H")
        )

        tf_30m = normalize_direction(
            tf_dirs.get("30M")
        )

        tf_15m = normalize_direction(
            tf_dirs.get("15M")
        )

        score, evidence = (
            calculate_strict_confidence(
                direction=candidate_dir,
                model_confidence=model_confidence,
                model_validation_win_rate=0,
                daily_direction=daily_dir,
                tf_4h=tf_4h,
                tf_1h=tf_1h,
                tf_30m=tf_30m,
                tf_15m=tf_15m,
                news_direction=news_dir,
                cot_direction=cot_dir,
                pivot_direction=pivot_dir,
                adx=adx,
                rsi=rsi,
                macd_agrees=macd_agrees,
                atr_expanding=atr_expanding,
                liquidity_sweep=liquidity_sweep,
                retest=retest,
                fvg=fvg,
                session_ok=session_ok
            )
        )

        # HARD FILTERS.
        hard_filter = (
            score >= 70
            and daily_dir == candidate_dir
            and tf_4h == candidate_dir
            and tf_1h == candidate_dir
            and adx >= 20
            and macd_agrees
            and session_ok
        )

        if news_dir not in (
            "NEUTRAL",
            candidate_dir
        ):
            hard_filter = False

        if not hard_filter:
            return (
                "WAIT",
                current_price,
                atr_now,
                (swing_high, swing_low),
                score,
                {
                    "ADX": round(adx, 2),
                    "RSI": round(rsi, 2),
                    "4H": tf_4h,
                    "1H": tf_1h,
                    "30M": tf_30m,
                    "15M": tf_15m,
                    "News": news_dir,
                    "COT": cot_dir,
                    "Pivot": pivot_dir,
                    "Liquidity Sweep": liquidity_sweep,
                    "Retest": retest,
                    "FVG": fvg,
                    "BOS": bos_detected,
                    "Session": session_ok,
                    "Evidence": evidence,
                    "Reason": "Hard filter failed"
                }
            )

        return (
            candidate_dir,
            current_price,
            atr_now,
            (swing_high, swing_low),
            score,
            {
                "ADX": round(adx, 2),
                "RSI": round(rsi, 2),
                "4H": tf_4h,
                "1H": tf_1h,
                "30M": tf_30m,
                "15M": tf_15m,
                "News": news_dir,
                "COT": cot_dir,
                "Pivot": pivot_dir,
                "Liquidity Sweep": liquidity_sweep,
                "Retest": retest,
                "FVG": fvg,
                "BOS": bos_detected,
                "Session": session_ok,
                "Evidence": evidence,
                "Reason": "All hard filters passed"
            }
        )

    except Exception as e:
        return (
            "WAIT",
            None,
            None,
            None,
            0,
            {"Error": str(e)}
        )


# ============================================================
# Trade levels
# ============================================================

def calculate_trade_levels(
    direction,
    current_price,
    atr,
    swing_high,
    swing_low
):
    if (
        current_price is None
        or atr is None
        or atr <= 0
    ):
        return "-", "-"

    if direction == "BUY":
        dist_to_support = (
            current_price - swing_low
        )

        if (
            0 < dist_to_support < atr * 3
        ):
            stop_loss_price = (
                swing_low - atr * 0.3
            )
        else:
            stop_loss_price = (
                current_price - atr * 1.5
            )

        dist_to_resist = (
            swing_high - current_price
        )

        if dist_to_resist > 0:
            take_profit_price = (
                swing_high - atr * 0.2
            )

            if (
                take_profit_price
                <= current_price
            ):
                take_profit_price = (
                    current_price + atr * 1.5
                )

        else:
            take_profit_price = (
                current_price + atr * 1.5
            )

    else:
        dist_to_resist = (
            swing_high - current_price
        )

        if (
            0 < dist_to_resist < atr * 3
        ):
            stop_loss_price = (
                swing_high + atr * 0.3
            )
        else:
            stop_loss_price = (
                current_price + atr * 1.5
            )

        dist_to_support = (
            current_price - swing_low
        )

        if dist_to_support > 0:
            take_profit_price = (
                swing_low + atr * 0.2
            )

            if (
                take_profit_price
                >= current_price
            ):
                take_profit_price = (
                    current_price - atr * 1.5
                )

        else:
            take_profit_price = (
                current_price - atr * 1.5
            )

    # Make sure TP is actually beyond entry.
    if direction == "BUY":
        if take_profit_price <= current_price:
            take_profit_price = (
                current_price + atr * 1.5
            )

        if stop_loss_price >= current_price:
            stop_loss_price = (
                current_price - atr * 1.5
            )

    else:
        if take_profit_price >= current_price:
            take_profit_price = (
                current_price - atr * 1.5
            )

        if stop_loss_price <= current_price:
            stop_loss_price = (
                current_price + atr * 1.5
            )

    return (
        round(stop_loss_price, 5),
        round(take_profit_price, 5)
    )


# ============================================================
# Firestore trade logging
# ============================================================

def has_open_trade(
    pair,
    timeframe="Daily"
):
    docs = (
        db.collection("trade_history")
        .where("pair", "==", pair)
        .where("timeframe", "==", timeframe)
        .where("outcome", "==", "OPEN")
        .stream()
    )

    return any(True for _ in docs)


def has_recent_duplicate_signal(
    pair,
    timeframe,
    decision,
    price
):
    """
    Prevent repeated refreshes from creating the same signal.
    We use date + pair + timeframe + direction + rounded entry.
    """

    today_str = (
        datetime.date.today()
        .strftime("%Y-%m-%d")
    )

    docs = (
        db.collection("trade_history")
        .where("pair", "==", pair)
        .where("timeframe", "==", timeframe)
        .where("date", "==", today_str)
        .stream()
    )

    rounded_price = round(
        float(price),
        4
    )

    for doc in docs:
        data = doc.to_dict()

        if (
            data.get("decision") == decision
            and round(
                safe_float(
                    data.get("entry_price")
                ),
                4
            ) == rounded_price
        ):
            return True

    return False


def save_decisions_to_firestore(
    results_table
):
    # Never create new paper trades while closed.
    if market_closed:
        return

    today_str = (
        datetime.date.today()
        .strftime("%Y-%m-%d")
    )

    now_str = (
        datetime.datetime.utcnow()
        .strftime("%Y-%m-%d %H:%M UTC")
    )

    for row in results_table:
        decision = row.get(
            "DECISION",
            "WAIT"
        )

        if decision not in (
            "BUY",
            "SELL"
        ):
            continue

        pair = row["Pair"]

        timeframe = row.get(
            "Timeframe_Label",
            "Daily"
        )

        price = row.get("Price")

        if price in (
            None,
            "-",
            ""
        ):
            continue

        if has_open_trade(
            pair,
            timeframe
        ):
            continue

        if has_recent_duplicate_signal(
            pair,
            timeframe,
            decision,
            price
        ):
            continue

        doc_id = (
            f"{today_str}_"
            f"{pair.replace('/', '')}_"
            f"{timeframe}_"
            f"{decision}_"
            f"{str(round(float(price), 4)).replace('.', '_')}"
        )

        db.collection(
            "trade_history"
        ).document(doc_id).set({
            "date": today_str,
            "pair": pair,
            "timeframe": timeframe,
            "decision": decision,
            "entry_price": float(price),
            "stop_loss": row["Stop Loss"],
            "take_profit": row["Take Profit"],
            "confidence": row.get(
                "Confidence",
                "-"
            ),
            "outcome": "OPEN",
            "opened_at": now_str,
            "closed_at": ""
        })


# ============================================================
# Trade monitoring
# ============================================================

def get_current_trade_price(pair):
    """
    Only returns a price from a source we can currently query.
    Caller decides whether market state permits closing.
    """

    try:
        if pair == "XAU/USD":
            current_df = fetch_gold_history()

        else:
            mapping = {
                "USD/JPY": ("USD", "JPY"),
                "GBP/USD": ("GBP", "USD"),
                "USD/CAD": ("USD", "CAD")
            }

            fs, ts = mapping[pair]

            current_df = fetch_daily_history(
                fs,
                ts
            )

        if (
            current_df is None
            or current_df.empty
        ):
            return None

        return float(
            current_df["close"].iloc[-1]
        )

    except Exception:
        return None


def check_previous_trades():
    docs = (
        db.collection("trade_history")
        .stream()
    )

    results = []

    for doc in docs:
        trade = doc.to_dict()

        pair = trade.get("pair")
        outcome = trade.get(
            "outcome",
            "OPEN"
        )

        decision = trade.get(
            "decision",
            "WAIT"
        )

        # NEVER close a trade during market-closed periods.
        if (
            outcome == "OPEN"
            and market_closed
        ):
            results.append({
                "Pair": pair,
                "Timeframe": trade.get(
                    "timeframe",
                    "Daily"
                ),
                "Decision": decision,
                "Entry": trade.get(
                    "entry_price",
                    "-"
                ),
                "Exit/Current": "-",
                "Opened": trade.get(
                    "opened_at",
                    trade.get("date", "-")
                ),
                "Closed": "OPEN",
                "Status": "OPEN"
            })

            continue

        if outcome == "OPEN":
            current_price = (
                get_current_trade_price(pair)
            )

            if current_price is None:
                continue

            sl = trade.get("stop_loss")
            tp = trade.get("take_profit")

            if (
                sl in (None, "-")
                or tp in (None, "-")
            ):
                continue

            sl = safe_float(sl)
            tp = safe_float(tp)

            if (
                np.isnan(sl)
                or np.isnan(tp)
            ):
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

            if outcome != "OPEN":
                now_str = (
                    datetime.datetime.utcnow()
                    .strftime(
                        "%Y-%m-%d %H:%M UTC"
                    )
                )

                db.collection(
                    "trade_history"
                ).document(doc.id).update({
                    "outcome": outcome,
                    "closed_at": now_str,
                    "exit_price": round(
                        current_price,
                        5
                    )
                })

                trade["closed_at"] = now_str
                trade["exit_price"] = round(
                    current_price,
                    5
                )

            else:
                trade["exit_price"] = round(
                    current_price,
                    5
                )

        results.append({
            "Pair": pair,
            "Timeframe": trade.get(
                "timeframe",
                "Daily"
            ),
            "Decision": decision,
            "Entry": trade.get(
                "entry_price",
                "-"
            ),
            "Exit/Current": trade.get(
                "exit_price",
                "-"
            ),
            "Opened": trade.get(
                "opened_at",
                trade.get("date", "-")
            ),
            "Closed": (
                trade.get(
                    "closed_at",
                    "OPEN"
                )
                if outcome != "OPEN"
                else "OPEN"
            ),
            "Status": outcome
        })

    return results


# ============================================================
# Full analysis
# ============================================================

def run_full_analysis():
    if market_closed:
        st.warning(
            "Market is closed. No new paper trades will be generated."
        )
        return

    news_bias, headlines = (
        get_news_sentiment()
    )

    results_table = []
    daily_directions = {}

    currency_pairs = {
        "USD/JPY": ("USD", "JPY"),
        "GBP/USD": ("GBP", "USD"),
        "USD/CAD": ("USD", "CAD")
    }

    # --------------------------------------------------------
    # DAILY ANALYSIS
    # --------------------------------------------------------
    for pair, (fs, ts) in currency_pairs.items():

        raw_df = fetch_daily_history(
            fs,
            ts
        )

        if raw_df is None:
            results_table.append({
                "Pair": pair,
                "Price": "-",
                "Price Model": "DATA ERROR",
                "News": "-",
                "Timeframe": "-",
                "COT": "-",
                "Pivot": "-",
                "Buy Conf": "-",
                "Sell Conf": "-",
                "Confidence": "-",
                "DECISION": "WAIT",
                "Stop Loss": "-",
                "Take Profit": "-",
                "Timeframe_Label": "Daily"
            })

            continue

        featured_df = build_features(
            raw_df,
            True
        )

        if featured_df.empty:
            continue

        (
            validation_win_rate,
            price_dir,
            current_price,
            model_probability
        ) = analyze_price_model(
            featured_df
        )

        daily_directions[pair] = (
            price_dir
        )

        news_dir = get_news_direction(
            pair,
            news_bias
        )

        tf_dirs = get_timeframe_directions(
            pair
        )

        tf_4h = normalize_direction(
            tf_dirs.get("4H")
        )

        tf_1h = normalize_direction(
            tf_dirs.get("1H")
        )

        tf_30m = normalize_direction(
            tf_dirs.get("30M")
        )

        tf_15m = normalize_direction(
            tf_dirs.get("15M")
        )

        contract, url, prefix, suffix = (
            cot_contracts[pair]
        )

        cot_dir = get_cot_positioning(
            contract,
            url,
            prefix,
            suffix
        )

        pivot_dir = get_pivot_direction(
            raw_df,
            True
        )

        adx = calculate_adx(
            featured_df
        )

        rsi = float(
            featured_df["rsi"].iloc[-1]
        )

        macd_hist = float(
            featured_df["macd_hist"].iloc[-1]
        )

        atr_now = float(
            featured_df["atr"].iloc[-1]
        )

        atr_avg = float(
            featured_df["atr"].tail(20).mean()
        )

        atr_expanding = (
            atr_now > atr_avg
        )

        macd_buy = (
            macd_hist > 0
        )

        macd_sell = (
            macd_hist < 0
        )

        liquidity_sweep, retest = (
            detect_liquidity_sweep_and_retest(
                featured_df,
                True
            )
        )

        fvg_buy = detect_fair_value_gap(
            featured_df,
            "BUY",
            True
        )

        fvg_sell = detect_fair_value_gap(
            featured_df,
            "SELL",
            True
        )

        session_ok = (
            is_good_trading_session()
        )

        # Calculate separate evidence for BUY and SELL.
        buy_score, _ = (
            calculate_strict_confidence(
                direction="BUY",
                model_confidence=model_probability,
                model_validation_win_rate=validation_win_rate,
                daily_direction=price_dir,
                tf_4h=tf_4h,
                tf_1h=tf_1h,
                tf_30m=tf_30m,
                tf_15m=tf_15m,
                news_direction=news_dir,
                cot_direction=cot_dir,
                pivot_direction=pivot_dir,
                adx=adx,
                rsi=rsi,
                macd_agrees=macd_buy,
                atr_expanding=atr_expanding,
                liquidity_sweep=liquidity_sweep,
                retest=retest,
                fvg=fvg_buy,
                session_ok=session_ok
            )
        )

        sell_score, _ = (
            calculate_strict_confidence(
                direction="SELL",
                model_confidence=(
                    100 - model_probability
                ),
                model_validation_win_rate=validation_win_rate,
                daily_direction=price_dir,
                tf_4h=tf_4h,
                tf_1h=tf_1h,
                tf_30m=tf_30m,
                tf_15m=tf_15m,
                news_direction=news_dir,
                cot_direction=cot_dir,
                pivot_direction=pivot_dir,
                adx=adx,
                rsi=rsi,
                macd_agrees=macd_sell,
                atr_expanding=atr_expanding,
                liquidity_sweep=liquidity_sweep,
                retest=retest,
                fvg=fvg_sell,
                session_ok=session_ok
            )
        )

        # HARD DAILY FILTER.
        buy_allowed = (
            buy_score >= 70
            and model_probability >= 70
            and validation_win_rate >= 60
            and price_dir == "BUY"
            and tf_4h == "BUY"
            and tf_1h == "BUY"
            and adx >= 20
            and macd_buy
            and session_ok
            and news_dir in (
                "BUY",
                "NEUTRAL"
            )
        )

        sell_allowed = (
            sell_score >= 70
            and model_probability <= 30
            and validation_win_rate >= 60
            and price_dir == "SELL"
            and tf_4h == "SELL"
            and tf_1h == "SELL"
            and adx >= 20
            and macd_sell
            and session_ok
            and news_dir in (
                "SELL",
                "NEUTRAL"
            )
        )

        if buy_allowed and buy_score > sell_score:
            decision = "BUY"
            confidence = buy_score

        elif (
            sell_allowed
            and sell_score > buy_score
        ):
            decision = "SELL"
            confidence = sell_score

        else:
            decision = "WAIT"
            confidence = max(
                buy_score,
                sell_score
            )

        sl_price = "-"
        tp_price = "-"

        if decision != "WAIT":
            atr = featured_df[
                "atr"
            ].iloc[-1]

            swing_high = featured_df[
                "swing_high_20"
            ].iloc[-1]

            swing_low = featured_df[
                "swing_low_20"
            ].iloc[-1]

            sl_price, tp_price = (
                calculate_trade_levels(
                    decision,
                    current_price,
                    atr,
                    swing_high,
                    swing_low
                )
            )

        results_table.append({
            "Pair": pair,
            "Price": round(
                current_price,
                4
            ),
            "Price Model": (
                f"{price_dir} "
                f"({validation_win_rate}%)"
            ),
            "ML Probability": (
                f"{model_probability}%"
            ),
            "News": news_dir,
            "4H": tf_4h,
            "1H": tf_1h,
            "30M": tf_30m,
            "15M": tf_15m,
            "COT": cot_dir,
            "Pivot": pivot_dir,
            "ADX": round(adx, 2),
            "Buy Conf": f"{buy_score}%",
            "Sell Conf": f"{sell_score}%",
            "Confidence": (
                f"{confidence}% "
                f"({get_confidence_label(confidence)})"
            ),
            "DECISION": decision,
            "Stop Loss": sl_price,
            "Take Profit": tp_price,
            "Timeframe_Label": "Daily"
        })

    # --------------------------------------------------------
    # GOLD DAILY ANALYSIS
    # --------------------------------------------------------
    gold_raw = fetch_gold_history()

    if gold_raw is not None:
        gold_featured = build_features(
            gold_raw,
            False
        )

        if not gold_featured.empty:
            (
                validation_win_rate,
                price_dir,
                current_price,
                model_probability
            ) = analyze_price_model(
                gold_featured
            )

            daily_directions[
                "XAU/USD"
            ] = price_dir

            news_dir = get_news_direction(
                "XAU/USD",
                news_bias
            )

            tf_dirs = get_timeframe_directions(
                "XAU/USD"
            )

            tf_4h = normalize_direction(
                tf_dirs.get("4H")
            )

            tf_1h = normalize_direction(
                tf_dirs.get("1H")
            )

            tf_30m = normalize_direction(
                tf_dirs.get("30M")
            )

            tf_15m = normalize_direction(
                tf_dirs.get("15M")
            )

            contract, url, prefix, suffix = (
                cot_contracts["XAU/USD"]
            )

            cot_dir = get_cot_positioning(
                contract,
                url,
                prefix,
                suffix
            )

            pivot_dir = get_pivot_direction(
                gold_raw,
                False
            )

            adx = calculate_adx(
                gold_featured
            )

            rsi = float(
                gold_featured[
                    "rsi"
                ].iloc[-1]
            )

            macd_hist = float(
                gold_featured[
                    "macd_hist"
                ].iloc[-1]
            )

            atr_now = float(
                gold_featured[
                    "atr"
                ].iloc[-1]
            )

            atr_avg = float(
                gold_featured[
                    "atr"
                ].tail(20)
                .mean()
            )

            atr_expanding = (
                atr_now > atr_avg
            )

            macd_buy = (
                macd_hist > 0
            )

            macd_sell = (
                macd_hist < 0
            )

            liquidity_sweep, retest = (
                detect_liquidity_sweep_and_retest(
                    gold_featured,
                    False
                )
            )

            fvg_buy = detect_fair_value_gap(
                gold_featured,
                "BUY",
                False
            )

            fvg_sell = detect_fair_value_gap(
                gold_featured,
                "SELL",
                False
            )

            session_ok = (
                is_good_trading_session()
            )

            buy_score, _ = (
                calculate_strict_confidence(
                    direction="BUY",
                    model_confidence=model_probability,
                    model_validation_win_rate=validation_win_rate,
                    daily_direction=price_dir,
                    tf_4h=tf_4h,
                    tf_1h=tf_1h,
                    tf_30m=tf_30m,
                    tf_15m=tf_15m,
                    news_direction=news_dir,
                    cot_direction=cot_dir,
                    pivot_direction=pivot_dir,
                    adx=adx,
                    rsi=rsi,
                    macd_agrees=macd_buy,
                    atr_expanding=atr_expanding,
                    liquidity_sweep=liquidity_sweep,
                    retest=retest,
                    fvg=fvg_buy,
                    session_ok=session_ok
                )
            )

            sell_score, _ = (
                calculate_strict_confidence(
                    direction="SELL",
                    model_confidence=(
                        100 - model_probability
                    ),
                    model_validation_win_rate=validation_win_rate,
                    daily_direction=price_dir,
                    tf_4h=tf_4h,
                    tf_1h=tf_1h,
                    tf_30m=tf_30m,
                    tf_15m=tf_15m,
                    news_direction=news_dir,
                    cot_direction=cot_dir,
                    pivot_direction=pivot_dir,
                    adx=adx,
                    rsi=rsi,
                    macd_agrees=macd_sell,
                    atr_expanding=atr_expanding,
                    liquidity_sweep=liquidity_sweep,
                    retest=retest,
                    fvg=fvg_sell,
                    session_ok=session_ok
                )
            )

            buy_allowed = (
                buy_score >= 70
                and model_probability >= 70
                and validation_win_rate >= 60
                and price_dir == "BUY"
                and tf_4h == "BUY"
                and tf_1h == "BUY"
                and adx >= 20
                and macd_buy
                and session_ok
                and news_dir in (
                    "BUY",
                    "NEUTRAL"
                )
            )

            sell_allowed = (
                sell_score >= 70
                and model_probability <= 30
                and validation_win_rate >= 60
                and price_dir == "SELL"
                and tf_4h == "SELL"
                and tf_1h == "SELL"
                and adx >= 20
                and macd_sell
                and session_ok
                and news_dir in (
                    "SELL",
                    "NEUTRAL"
                )
            )

            if (
                buy_allowed
                and buy_score > sell_score
            ):
                decision = "BUY"
                confidence = buy_score

            elif (
                sell_allowed
                and sell_score > buy_score
            ):
                decision = "SELL"
                confidence = sell_score

            else:
                decision = "WAIT"
                confidence = max(
                    buy_score,
                    sell_score
                )

            sl_price = "-"
            tp_price = "-"

            if decision != "WAIT":
                atr = gold_featured[
                    "atr"
                ].iloc[-1]

                swing_high = gold_featured[
                    "swing_high_20"
                ].iloc[-1]

                swing_low = gold_featured[
                    "swing_low_20"
                ].iloc[-1]

                sl_price, tp_price = (
                    calculate_trade_levels(
                        decision,
                        current_price,
                        atr,
                        swing_high,
                        swing_low
                    )
                )

            results_table.append({
                "Pair": "XAU/USD",
                "Price": round(
                    current_price,
                    4
                ),
                "Price Model": (
                    f"{price_dir} "
                    f"({validation_win_rate}%)"
                ),
                "ML Probability": (
                    f"{model_probability}%"
                ),
                "News": news_dir,
                "4H": tf_4h,
                "1H": tf_1h,
                "30M": tf_30m,
                "15M": tf_15m,
                "COT": cot_dir,
                "Pivot": pivot_dir,
                "ADX": round(adx, 2),
                "Buy Conf": f"{buy_score}%",
                "Sell Conf": f"{sell_score}%",
                "Confidence": (
                    f"{confidence}% "
                    f"({get_confidence_label(confidence)})"
                ),
                "DECISION": decision,
                "Stop Loss": sl_price,
                "Take Profit": tp_price,
                "Timeframe_Label": "Daily"
            })

    # --------------------------------------------------------
    # STRICT INTRADAY
    # --------------------------------------------------------
    intraday_allowed = {
        "1H": [
            "USD/JPY",
            "USD/CAD",
            "XAU/USD"
        ],
        "4H": [
            "GBP/USD"
        ]
    }

    for label, interval in [
        ("1H", "1h"),
        ("4H", "4h")
    ]:
        for pair in intraday_allowed[label]:
            try:
                daily_dir_for_pair = (
                    daily_directions.get(
                        pair,
                        "WAIT"
                    )
                )

                # Need the daily direction before an intraday
                # trade can even be considered.
                if daily_dir_for_pair not in (
                    "BUY",
                    "SELL"
                ):
                    continue

                news_dir = get_news_direction(
                    pair,
                    news_bias
                )

                contract, url, prefix, suffix = (
                    cot_contracts[pair]
                )

                cot_dir = get_cot_positioning(
                    contract,
                    url,
                    prefix,
                    suffix
                )

                raw_for_pivot = (
                    fetch_gold_history()
                    if pair == "XAU/USD"
                    else fetch_daily_history(
                        *{
                            "USD/JPY": (
                                "USD",
                                "JPY"
                            ),
                            "GBP/USD": (
                                "GBP",
                                "USD"
                            ),
                            "USD/CAD": (
                                "USD",
                                "CAD"
                            )
                        }[pair]
                    )
                )

                pivot_dir = get_pivot_direction(
                    raw_for_pivot,
                    pair != "XAU/USD"
                )

                (
                    direction,
                    price,
                    atr,
                    levels,
                    score,
                    diagnostics
                ) = analyze_intraday_signal(
                    pair,
                    interval,
                    daily_dir_for_pair,
                    news_dir,
                    cot_dir,
                    pivot_dir
                )

                if (
                    direction != "WAIT"
                    and price is not None
                    and atr is not None
                    and levels
                ):
                    swing_high, swing_low = (
                        levels
                    )

                    sl, tp = (
                        calculate_trade_levels(
                            direction,
                            price,
                            atr,
                            swing_high,
                            swing_low
                        )
                    )

                    results_table.append({
                        "Pair": pair,
                        "Price": round(
                            price,
                            4
                        ),
                        "Price Model": "-",
                        "ML Probability": "-",
                        "News": news_dir,
                        "4H": diagnostics.get(
                            "4H",
                            "-"
                        ),
                        "1H": diagnostics.get(
                            "1H",
                            "-"
                        ),
                        "30M": diagnostics.get(
                            "30M",
                            "-"
                        ),
                        "15M": diagnostics.get(
                            "15M",
                            "-"
                        ),
                        "COT": cot_dir,
                        "Pivot": pivot_dir,
                        "ADX": diagnostics.get(
                            "ADX",
                            "-"
                        ),
                        "Buy Conf": "-",
                        "Sell Conf": "-",
                        "Confidence": (
                            f"{score}% "
                            f"({get_confidence_label(score)})"
                        ),
                        "DECISION": direction,
                        "Stop Loss": sl,
                        "Take Profit": tp,
                        "Timeframe_Label": label
                    })

            except Exception as e:
                st.sidebar.write(
                    f"⚠️ Skipped {pair} "
                    f"{label}: "
                    f"{str(e)[:80]}"
                )

    st.session_state.results_table = (
        results_table
    )

    st.session_state.headlines = (
        headlines
    )

    st.session_state.news_bias = (
        news_bias
    )

    save_decisions_to_firestore(
        results_table
    )


# ============================================================
# Backtest UI
# ============================================================

def run_validation_backtests():
    pairs = {
        "USD/JPY": (
            "USD",
            "JPY",
            True
        ),
        "GBP/USD": (
            "GBP",
            "USD",
            True
        ),
        "USD/CAD": (
            "USD",
            "CAD",
            True
        ),
        "XAU/USD": (
            None,
            None,
            False
        )
    }

    rows = []

    progress = st.progress(0)

    for idx, (pair, info) in enumerate(
        pairs.items(),
        start=1
    ):
        fs, ts, has_ohlc = info

        if has_ohlc:
            raw = fetch_daily_history(
                fs,
                ts
            )
        else:
            raw = fetch_gold_history()

        if raw is None:
            rows.append({
                "Pair": pair,
                "Status": "DATA UNAVAILABLE"
            })

            progress.progress(
                idx / len(pairs)
            )

            continue

        featured = build_features(
            raw,
            has_ohlc
        )

        result_60 = walk_forward_backtest(
            featured,
            probability_threshold=0.60
        )

        result_70 = walk_forward_backtest(
            featured,
            probability_threshold=0.70
        )

        if result_70 is None:
            rows.append({
                "Pair": pair,
                "Status": "NOT ENOUGH DATA"
            })

        else:
            rows.append({
                "Pair": pair,
                "Candles": len(featured),
                "70% Filter Win Rate": (
                    f"{result_70['win_rate']}%"
                ),
                "70% Filter Trades": (
                    result_70["trades"]
                ),
                "70% Coverage": (
                    f"{result_70['coverage']}%"
                ),
                "70% Wins": result_70["wins"],
                "70% Losses": result_70["losses"],
                "60% Filter Win Rate": (
                    f"{result_60['win_rate']}%"
                    if result_60
                    else "-"
                ),
                "Status": "VALID"
            })

        progress.progress(
            idx / len(pairs)
        )

    return pd.DataFrame(rows)


# ============================================================
# Streamlit UI
# ============================================================

st.title(
    "📈 YonKing - Strict Forex Analysis Dashboard"
)

st.caption(
    "Paper-trading only • strict 70% evidence threshold • "
    "walk-forward validation • no broker execution"
)

if market_closed:
    st.warning(
        "⏸️ Market is closed. YonKing will not create new "
        "paper trades or close existing trades using stale "
        "weekend prices."
    )

# -------------------------
# Backtest buttons
# -------------------------
st.sidebar.divider()
st.sidebar.subheader(
    "🧪 Validation"
)

if st.sidebar.button(
    "RUN WALK-FORWARD BACKTEST"
):
    with st.spinner(
        "Running chronological validation..."
    ):
        backtest_results = (
            run_validation_backtests()
        )

    st.subheader(
        "🧪 Walk-Forward Backtest Results"
    )

    st.dataframe(
        backtest_results,
        use_container_width=True
    )

    st.info(
        "The 70% column means the model only entered when "
        "its probability threshold was at least 70%. "
        "It does NOT mean the strategy is guaranteed to "
        "win 70% of future trades."
    )


# -------------------------
# Admin
# -------------------------
if (
    st.session_state.user_email
    == ADMIN_EMAIL
):
    with st.sidebar.expander(
        "🔑 Admin Panel"
    ):
        st.write("Pending Approvals")

        pending = (
            db.collection("pending_users")
            .where(
                "approved",
                "==",
                False
            )
            .stream()
        )

        for p in pending:
            data = p.to_dict()

            st.write(
                f"**{data['email']}** | "
                f"Age: {data['age']} | "
                f"Location: {data['location']} | "
                f"Phone: {data['phone']}"
            )

            if st.button(
                f"Approve {data['email']}",
                key=f"approve_{p.id}"
            ):
                (
                    db.collection(
                        "pending_users"
                    )
                    .document(p.id)
                    .update({
                        "approved": True
                    })
                )

                st.success(
                    f"Approved {data['email']}"
                )

                st.rerun()


# -------------------------
# Session state
# -------------------------
if "results_table" not in st.session_state:
    st.session_state.results_table = None
    st.session_state.headlines = None
    st.session_state.news_bias = None


# -------------------------
# Trade history
# -------------------------
st.divider()
st.subheader(
    "📜 Trade History & Results"
)

past_results = (
    check_previous_trades()
)

if past_results:
    df_results = pd.DataFrame(
        past_results
    )

    def color_status(row):
        if row["Status"] == "WIN":
            return [
                "background-color: #1a4d2e"
            ] * len(row)

        if row["Status"] == "LOSS":
            return [
                "background-color: #5c1a1a"
            ] * len(row)

        return [""] * len(row)

    st.dataframe(
        df_results.style.apply(
            color_status,
            axis=1
        ),
        use_container_width=True
    )

else:
    st.write(
        "No trade history yet."
    )


# -------------------------
# Automatic analysis
# -------------------------
if (
    st.session_state.results_table
    is None
):
    if market_closed:
        st.info(
            "⏸️ Analysis paused because the market "
            "is closed. No new paper trades will be created."
        )

    else:
        with st.spinner(
            "Running YonKing's strict analysis..."
        ):
            run_full_analysis()


# -------------------------
# Manual refresh
# -------------------------
if st.button(
    "🔄 Refresh Analysis"
):
    if market_closed:
        st.warning(
            "⏸️ Market is closed. YonKing will not "
            "create a new trade from stale prices."
        )

    else:
        with st.spinner(
            "Refreshing strict analysis..."
        ):
            run_full_analysis()


# -------------------------
# Results
# -------------------------
if (
    st.session_state.results_table
    is not None
):
    st.subheader(
        f"News Sentiment: "
        f"{st.session_state.news_bias}"
    )

    results_df = pd.DataFrame(
        st.session_state.results_table
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.info(
        "Only BUY/SELL decisions that pass the strict "
        "filters are saved. A score of 70% is an evidence "
        "threshold, not a guaranteed 70% future win rate."
    )

    st.divider()

    st.subheader(
        "📊 YonKing's Calculated Chart View"
    )

    chart_pairs = {
        "USD/JPY": (
            "USD",
            "JPY",
            True
        ),
        "GBP/USD": (
            "GBP",
            "USD",
            True
        ),
        "USD/CAD": (
            "USD",
            "CAD",
            True
        ),
        "XAU/USD": (
            None,
            None,
            False
        )
    }

    selected_pair = st.selectbox(
        "Select a pair to view its chart "
        "with calculated levels:",
        list(chart_pairs.keys())
    )

    fs, ts, has_ohlc = (
        chart_pairs[selected_pair]
    )

    if has_ohlc:
        chart_df = (
            fetch_daily_history(
                fs,
                ts
            )
        )

    else:
        chart_df = fetch_gold_history()

    if (
        chart_df is not None
        and not chart_df.empty
    ):
        chart_df = (
            chart_df
            .tail(60)
            .reset_index(drop=True)
        )

        fig = go.Figure()

        if has_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=chart_df["date"],
                    open=chart_df["open"],
                    high=chart_df["high"],
                    low=chart_df["low"],
                    close=chart_df["close"],
                    name=selected_pair
                )
            )

            swing_high = (
                chart_df["high"]
                .rolling(
                    20,
                    min_periods=1
                )
                .max()
                .iloc[-1]
            )

            swing_low = (
                chart_df["low"]
                .rolling(
                    20,
                    min_periods=1
                )
                .min()
                .iloc[-1]
            )

        else:
            fig.add_trace(
                go.Scatter(
                    x=chart_df["date"],
                    y=chart_df["close"],
                    mode="lines",
                    name=selected_pair
                )
            )

            swing_high = (
                chart_df["close"]
                .rolling(
                    20,
                    min_periods=1
                )
                .max()
                .iloc[-1]
            )

            swing_low = (
                chart_df["close"]
                .rolling(
                    20,
                    min_periods=1
                )
                .min()
                .iloc[-1]
            )

        fig.add_hline(
            y=swing_high,
            line_dash="dash",
            line_color="red",
            annotation_text=(
                f"Resistance: "
                f"{round(swing_high, 4)}"
            )
        )

        fig.add_hline(
            y=swing_low,
            line_dash="dash",
            line_color="green",
            annotation_text=(
                f"Support: "
                f"{round(swing_low, 4)}"
            )
        )

        if len(chart_df) >= 2:
            y_prev = chart_df.iloc[-2]

            close_p = y_prev["close"]

            if has_ohlc:
                high_p = y_prev["high"]
                low_p = y_prev["low"]

            else:
                high_p = close_p * 1.005
                low_p = close_p * 0.995

            pivot = (
                high_p
                + low_p
                + close_p
            ) / 3

            fig.add_hline(
                y=pivot,
                line_dash="dot",
                line_color="yellow",
                annotation_text=(
                    f"Pivot: "
                    f"{round(pivot, 4)}"
                )
            )

        recent_lows = (
            chart_df["low"]
            if has_ohlc
            else chart_df["close"]
        )

        min_idx = recent_lows.idxmin()

        if min_idx < len(chart_df) - 1:
            fig.add_trace(
                go.Scatter(
                    x=[
                        chart_df[
                            "date"
                        ].iloc[min_idx],
                        chart_df[
                            "date"
                        ].iloc[-1]
                    ],
                    y=[
                        recent_lows.iloc[
                            min_idx
                        ],
                        chart_df[
                            "close"
                        ].iloc[-1]
                    ],
                    mode="lines",
                    line=dict(
                        color="cyan",
                        width=2
                    ),
                    name="Trend line"
                )
            )

        fig.update_layout(
            title=(
                f"{selected_pair} - "
                "Price with Calculated Levels"
            ),
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with st.expander(
        "Headlines used for news sentiment"
    ):
        for headline in (
            st.session_state.headlines
            or []
        ):
            st.write(
                f"- {headline}"
            )
