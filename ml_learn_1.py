import os
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
import time
import requests
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

warnings.filterwarnings("ignore")

# --- CLOUD CONFIGURATION ---
# Pulling from Railway Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 8080)) # Railway provides this

TICKER = "GLD"
DXY_TICKER = "UUP"
INTERVAL = "15m"
PERIOD = "59d"
HORIZON = 12
CONF_THRESHOLD = 0.55

# --- DUMMY SERVER FOR RAILWAY HEALTH CHECKS ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is Running")

def run_health_server():
    server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
    server.serve_forever()

# --- NOTIFICATION ENGINE ---
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing in Environment Variables!")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# --- AI LOGIC (Indicators & Analysis) ---
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / (loss + 0.001))))

def get_atr(df, period=14):
    h_l = df['High'] - df['Low']
    h_c = np.abs(df['High'] - df['Close'].shift())
    l_c = np.abs(df['Low'] - df['Close'].shift())
    return pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(period).mean()

def run_analysis(last_signal):
    print(f"🕒 Analyzing: {datetime.now().strftime('%H:%M:%S')}")
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    dxy = yf.download(DXY_TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    
    if df.empty or len(df) < 200:
        df = yf.download(TICKER, period="730d", interval="1h", progress=False, auto_adjust=True)
        dxy = yf.download(DXY_TICKER, period="730d", interval="1h", progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    # Features
    df["dxy_ret"] = dxy["Close"].pct_change(5).ffill().fillna(0)
    df["RSI"] = get_rsi(df["Close"])
    df["ATR_pct"] = get_atr(df) / df["Close"]
    df["EMA"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["dist_ema"] = (df["Close"] - df["EMA"]) / df["EMA"]
    df["is_ny"] = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
    feat_data = df.dropna()

    # Model
    future_change = (feat_data["Close"].shift(-HORIZON) - feat_data["Close"]) / feat_data["Close"]
    feat_data["target"] = 1
    feat_data.loc[future_change > (feat_data["ATR_pct"] * 0.5), "target"] = 2
    feat_data.loc[future_change < -(feat_data["ATR_pct"] * 0.5), "target"] = 0
    train_df = feat_data.dropna()

    features = ["RSI", "ATR_pct", "dist_ema", "dxy_ret", "is_ny"]
    X, y = train_df[features], train_df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0)
    model.fit(X_train, y_train)

    # Signal Generation
    last_row = feat_data.iloc[-1:]
    probs = model.predict_proba(last_row[features])[0]
    price = last_row['Close'].values[0]
    trend_up = last_row['dist_ema'].values[0] > 0
    
    current_signal = "HOLD"
    if probs[2] >= CONF_THRESHOLD and trend_up: current_signal = "BUY"
    elif probs[0] >= CONF_THRESHOLD and not trend_up: current_signal = "SELL"

    if current_signal != last_signal:
        msg = f"🔔 *GOLD AI ALERT*\nSignal: *{current_signal}*\nPrice: ${price:.2f}"
        send_telegram(msg)
    
    return current_signal

# --- MAIN LOOP ---
def main():
    # Start the dummy web server in the background for Railway
    threading.Thread(target=run_health_server, daemon=True).start()
    
    print("🚀 Bot deployed. Monitoring markets...")
    last_signal = "HOLD"
    while True:
        try:
            last_signal = run_analysis(last_signal)
            time.sleep(900) # Wait 15 mins
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
