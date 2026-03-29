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

TICKER = "GC=F" # Gold Futures ($4,495 range)
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
    """Checks Telegram every 2 seconds. Works even when market is closed."""
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
                            send_telegram(f"🎯 **Target Set!**\nAlert at: **${target_price}**")
                        except:
                            send_telegram("❌ Use: `/set 4500`")
                    
                    elif msg == "/stop":
                        target_price = None
                        send_telegram("🛑 **Target Cleared.**")
                    
                    elif msg == "/price":
                        # Attempt to get the latest 1-minute price
                        data = yf.download(TICKER, period="1d", interval="1m", progress=False)
                        if data.empty or len(data) == 0:
                            # Market is likely closed, get the last daily close instead
                            hist = yf.download(TICKER, period="5d", interval="1d", progress=False)
                            price = hist['Close'].iloc[-1]
                            send_telegram(f"😴 **Market Closed.**\nLast Close: **${price:.2f}**\n*Opens Sunday 6PM ET*")
                        else:
                            price = data['Close'].iloc[-1]
                            send_telegram(f"💰 **Live Gold:** ${price:.2f}")

                    elif msg == "/status":
                        status = f"Target: `${target_price if target_price else 'None'}`"
                        send_telegram(f"🤖 **Bot Status**\n{status}")
            
            time.sleep(2)
        except Exception as e:
            print(f"Listener Error: {e}")
            time.sleep(10)

# --- AI & ANALYSIS ---
def run_analysis(last_signal):
    global target_price
    print(f"🕒 Market Check: {datetime.now().strftime('%H:%M:%S')}")
    
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    if df.empty or len(df) < 100:
        print("⚠️ Waiting for Market Open (Sunday 6PM ET)...")
        return last_signal

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicators
    df["RSI"] = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / 
                                   df['Close'].diff().where(lambda x: x<0, 0).abs().rolling(14).mean())))
    df["EMA"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["ATR"] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    feat_data = df.dropna()

    current_price = feat_data['Close'].iloc[-1]
    
    # Target Alert
    if target_price and current_price >= target_price:
        send_telegram(f"🚀 **TARGET HIT!**\nGold is at **${current_price:.2f}**")
        target_price = None

    # AI Prediction
    future_change = (feat_data["Close"].shift(-HORIZON) - feat_data["Close"])
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
    trend_up = current_price > feat_data["EMA"].iloc[-1]
    
    current_signal = "HOLD"
    if probs[2] >= CONF_THRESHOLD and trend_up: current_signal = "BUY"
    elif probs[0] >= CONF_THRESHOLD and not trend_up: current_signal = "SELL"

    if current_signal != last_signal:
        send_telegram(f"🔔 **SIGNAL CHANGE**\nAction: **{current_signal}**\nPrice: ${current_price:.2f}")
    
    return current_signal

# --- RAILWAY SERVER ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"Bot Alive")

def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', PORT), HealthCheckHandler).serve_forever, daemon=True).start()
    threading.Thread(target=command_listener, daemon=True).start()
    
    print("🤖 AI Gold Bot Ready.")
    last_signal = "HOLD"
    while True:
        try:
            last_signal = run_analysis(last_signal)
            time.sleep(900)
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
