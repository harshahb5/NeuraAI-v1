import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import finnhub

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY") or "YOUR_API_KEY_HERE"
SYMBOLS = ['XAUUSD', 'BTCUSD', 'EURUSD', 'NIFTY']
TIMEFRAMES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}
TRADE_HISTORY_CSV = "trade_history.csv"
RF_MODEL = None

def get_finnhub_candles(symbol, tf="M15", limit=300):
    client = finnhub.Client(api_key=FINNHUB_KEY)
    tf_min = TIMEFRAMES.get(tf, 15)
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(minutes=limit * tf_min)).timestamp())
    res = client.crypto_candles(f'BINANCE:{symbol}', str(tf_min), start, end)
    if res.get('s') != 'ok':
        return pd.DataFrame()
    df = pd.DataFrame(res)[['t', 'o', 'h', 'l', 'c', 'v']]
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA15'] = df['close'].ewm(span=15).mean()
    return df

def load_history():
    if not os.path.exists(TRADE_HISTORY_CSV):
        return pd.DataFrame(columns=["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "pnl_usd"])
    return pd.read_csv(TRADE_HISTORY_CSV)

def save_history(df):
    df.to_csv(TRADE_HISTORY_CSV, index=False)

def train_model():
    candles = get_finnhub_candles('XAUUSD', 'M15', 100)
    if candles.empty:
        return RandomForestClassifier()
    candles['ema_gap'] = candles['EMA9'] - candles['EMA15']
    candles['candle_body'] = candles['close'] - candles['open']
    candles['direction'] = (candles['candle_body'] > 0).astype(int)
    X = candles[['EMA9', 'EMA15', 'ema_gap', 'candle_body']].fillna(0)
    y = candles['direction']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

@app.on_event("startup")
def startup_event():
    global RF_MODEL
    RF_MODEL = train_model()

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "symbols": SYMBOLS})

@app.get("/api/candles")
def api_candles(symbol: str = "XAUUSD", timeframe: str = "M15"):
    df = get_finnhub_candles(symbol, timeframe, 300)
    if df.empty:
        return {"data": []}
    records = df.to_dict('records')
    for r in records:
        r['time'] = r['time'].strftime('%Y-%m-%d %H:%M')
    return {"data": records}

@app.get("/api/predict")
def api_predict(symbol: str = "XAUUSD", timeframe: str = "M15"):
    df = get_finnhub_candles(symbol, timeframe, 30)
    if df.empty:
        return {"suggestion": None}
    latest = df.iloc[-1]
    ema_gap = latest['EMA9'] - latest['EMA15']
    candle_body = latest['close'] - latest['open']
    X_pred = np.array([[latest['EMA9'], latest['EMA15'], ema_gap, candle_body]])
    pred = RF_MODEL.predict(X_pred)[0] if RF_MODEL else 1
    conf = RF_MODEL.predict_proba(X_pred)[0, int(pred)] * 100 if hasattr(RF_MODEL, "predict_proba") else 50
    direction = 'Buy' if pred == 1 else 'Sell'
    entry = float(latest['close'])
    suggestion = dict(
        direction=direction,
        entry_price=round(entry,2),
        sl=round(entry - 10,2) if direction == 'Buy' else round(entry + 10,2),
        tp=round(entry + 20,2) if direction == 'Buy' else round(entry - 20,2),
        ai_prob=round(conf,2)
    )
    return {"suggestion": suggestion}

@app.get("/api/history")
def api_history(start: str = None, end: str = None):
    df = load_history()
    if start:
        df = df[df['entry_time'] >= start]
    if end:
        df = df[df['exit_time'] <= end]
    return {"data": df.to_dict('records')}

@app.post("/api/retrain")
def api_retrain():
    global RF_MODEL
    RF_MODEL = train_model()
    return {"status": "ok"}

@app.get("/api/export")
def api_export():
    return FileResponse(TRADE_HISTORY_CSV, media_type='text/csv', filename='trade_history.csv')
