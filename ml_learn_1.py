import os
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
import time
import requests
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

warnings.filterwarnings("ignore")

# --- CONFIG ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 8080)) 

TICKER = "GC=F" 
# The typical difference between Futures and Spot (~$29.30 currently)
# This allows the AI to use high-quality Futures data while showing you Spot prices.
BASIS_DIFF = 29.30 

INTERVAL = "15m"
PERIOD = "59d"
HORIZON = 12
CONF_THRESHOLD = 0.55

target_price = None
last_update_id = 0

# --- TELEGRAM CORE ---
def send_telegram(message):
    if not TELEGRAM_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
    except: pass

def command_listener():
    global target_price, last_update_id
    print("📡 Telegram Listener Active...")
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=10"
            response = requests.get(url, timeout=15).json()
            if response.get("result"):
                for upd in response["result"]:
                    last_update_id = upd["update_id"]
                    msg = upd.get("message", {}).get("text", "").strip().lower()
                    
                    if msg.startswith("/set"):
                        try:
                            val = float(msg.split(" ")[1])
                            target_price = val
                            send_telegram(f"🎯 **Target Set!**\nAlert at Spot Price: **${target_price}**")
                        except:
                            send_telegram("❌ Use: `/set 4495`")
                    
                    elif msg == "/stop":
                        target_price = None
                        send_telegram("🛑 **Target Cleared.**")
                    
                    elif msg == "/price":
                        data = yf.download(TICKER, period="1d", interval="1m", progress=False)
                        if data.empty:
                            hist = yf.download(TICKER, period="5d", interval="1d", progress=False)
                            if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
                            spot_est = hist['Close'].iloc[-1] - BASIS_DIFF
                            send_telegram(f"😴 **Market Closed.**\nEst. Spot: **${float(spot_est):.2f}**")
                        else:
                            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                            spot_price = data['Close'].iloc[-1] - BASIS_DIFF
                            send_telegram(f"💰 **Current Gold Spot:** ${float(spot_price):.2f}")

                    elif msg == "/status":
                        send_telegram(f"🤖 **Bot Status**\nTarget: `${target_price if target_price else 'None'}`")
            time.sleep(2)
        except: time.sleep(10)

# --- AI & ANALYSIS ---
def run_analysis(last_signal):
    global target_price
    print(f"🕒 Market Check: {datetime.now().strftime('%H:%M:%S')}")
    
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    if df is None or df.empty or len(df) < 100:
        return last_signal

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Convert all prices to Spot for the user/alerts
    df['Spot_Close'] = df['Close'] - BASIS_DIFF
    current_spot = df['Spot_Close'].iloc[-1]
    
    # RSI & EMA using Spot prices
    df["RSI"] = 100 - (100 / (1 + (df['Spot_Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / 
                                   df['Spot_Close'].diff().where(lambda x: x<0, 0).abs().rolling(14).mean())))
    df["EMA"] = df["Spot_Close"].ewm(span=100, adjust=False).mean()
    df["ATR"] = pd.concat([df['High']-df['Low'], (df['High']-df['Spot_Close'].shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    feat_data = df.dropna()

    # Manual Price Alert Check
    if target_price:
        if current_spot >= target_price:
            send_telegram(f"🚀 **SPOT TARGET REACHED!**\nGold is at **${current_spot:.2f}**")
            target_price = None

    # ML Training
    future_change = (feat_data["Spot_Close"].shift(-HORIZON) - feat_data["Spot_Close"])
    feat_data["target"] = 1
    feat_data.loc[future_change > (feat_data["ATR"] * 0.5), "target"] = 2
    feat_data.loc[future_change < -(feat_data["ATR"] * 0.5), "target"] = 0
    train_df = feat_data.dropna()

    if len(train_df) < 50 or len(np.unique(train_df["target"])) < 2:
        return last_signal

    X, y = train_df[["RSI", "EMA"]], train_df["target"]
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0)
    model.fit(X, y)

    probs = model.predict_proba(feat_data[["RSI", "EMA"]].iloc[-1:])[0]
    trend_up = current_spot > feat_data["EMA"].iloc[-1]
    
    current_signal = "HOLD"
    if probs[2] >= CONF_THRESHOLD and trend_up: current_signal = "BUY"
    elif probs[0] >= CONF_THRESHOLD and not trend_up: current_signal = "SELL"

    if current_signal != last_signal:
        send_telegram(f"🔔 **AI SIGNAL**\nSpot Price: **${current_spot:.2f}**\nAction: **{current_signal}**")
    
    return current_signal

# --- SERVER ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"Bot Alive")

def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', PORT), HealthCheckHandler).serve_forever, daemon=True).start()
    threading.Thread(target=command_listener, daemon=True).start()
    
    print("🤖 AI Gold Bot (Spot-Adjusted) Ready.")
    last_signal = "HOLD"
    while True:
        try:
            last_signal = run_analysis(last_signal)
            time.sleep(900)
        except Exception as e:
            print(f"Main Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
