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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 8080)) 

# --- TICKER SET TO GOLD FUTURES ($4,495+) ---
TICKER = "GC=F" 
DXY_TICKER = "UUP"
INTERVAL = "15m"
PERIOD = "59d"
HORIZON = 12
CONF_THRESHOLD = 0.55
EMA_PERIOD = 100

# Global storage for your custom price alerts
target_price = None
last_update_id = 0

# --- DUMMY SERVER FOR RAILWAY ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Gold Bot is Active")

def run_health_server():
    server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
    server.serve_forever()

# --- TELEGRAM INTERACTION ---
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload)
    except Exception as e: print(f"Telegram Error: {e}")

def check_remote_commands():
    """Check Telegram for /set XXXX commands"""
    global target_price, last_update_id
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?offset={last_update_id + 1}"
        updates = requests.get(url).json()
        if updates.get("result"):
            for upd in updates["result"]:
                last_update_id = upd["update_id"]
                msg = upd.get("message", {}).get("text", "")
                if msg.startswith("/set"):
                    try:
                        val = float(msg.split(" ")[1])
                        target_price = val
                        send_telegram(f"🎯 Target set! I'll alert you at **${target_price}**")
                    except:
                        send_telegram("❌ Use: `/set 4500`")
                elif msg == "/status":
                    send_telegram(f"🤖 Bot is running.\nTarget: {target_price if target_price else 'None'}")
    except: pass

# --- INDICATORS ---
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

# --- ENGINE ---
def run_analysis(last_signal):
    global target_price
    check_remote_commands()
    
    print(f"🕒 Analyzing Gold Futures: {datetime.now().strftime('%H:%M:%S')}")
    
    # Fetch actual Gold Futures price
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    dxy = yf.download(DXY_TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    
    if df.empty or len(df) < 150:
        print("⚠️ Waiting for more market data...")
        return last_signal

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    # Feature Logic
    df["dxy_ret"] = dxy["Close"].pct_change(5).ffill().fillna(0)
    df["RSI"] = get_rsi(df["Close"])
    df["ATR_pct"] = get_atr(df) / df["Close"]
    df["EMA"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["dist_ema"] = (df["Close"] - df["EMA"]) / df["EMA"]
    df["is_ny"] = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
    feat_data = df.dropna()

    # Target Labeling
    future_change = (feat_data["Close"].shift(-HORIZON) - feat_data["Close"]) / feat_data["Close"]
    feat_data["target"] = 1
    feat_data.loc[future_change > (feat_data["ATR_pct"] * 0.5), "target"] = 2
    feat_data.loc[future_change < -(feat_data["ATR_pct"] * 0.5), "target"] = 0
    train_df = feat_data.dropna()

    if len(train_df) < 50 or len(np.unique(train_df["target"])) < 2:
        print("⚠️ Data too flat to train. Holding...")
        return last_signal

    # AI Training
    features = ["RSI", "ATR_pct", "dist_ema", "dxy_ret", "is_ny"]
    X, y = train_df[features], train_df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0)
    model.fit(X_train, y_train)

    # Current Stats
    last_row = feat_data.iloc[-1:]
    probs = model.predict_proba(last_row[features])[0]
    current_price = last_row['Close'].values[0]
    
    # Check manual target
    if target_price:
        if current_price >= target_price:
            send_telegram(f"🚀 **TARGET REACHED!**\nGold is now **${current_price:.2f}**")
            target_price = None

    # Determine Signal
    trend_up = last_row['dist_ema'].values[0] > 0
    current_signal = "HOLD"
    if probs[2] >= CONF_THRESHOLD and trend_up: current_signal = "BUY"
    elif probs[0] >= CONF_THRESHOLD and not trend_up: current_signal = "SELL"

    if current_signal != last_signal:
        msg = f"🔔 *NEW SIGNAL*\nPrice: **${current_price:.2f}**\nAction: **{current_signal}**"
        send_telegram(msg)
    
    return current_signal

def main():
    threading.Thread(target=run_health_server, daemon=True).start()
    print(f"🤖 Monitoring {TICKER} at ${INTERVAL} intervals...")
    last_signal = "HOLD"
    while True:
        try:
            last_signal = run_analysis(last_signal)
            time.sleep(900) # Check every 15 mins
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
