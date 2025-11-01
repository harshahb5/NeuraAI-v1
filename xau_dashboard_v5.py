# xau_dashboard_v5.py
"""
Neura.AI v1.1 backend (FastAPI)
- Cloud-friendly: uses Finnhub fallback
- Optional MT5 bridge: set BRIDGE_URL & BRIDGE_API_KEY
- SQLite trade_history.db + trade_history.csv sync
- Manual retrain (RandomForest)
- Endpoints for frontend (dashboard.html)
"""

import os
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Config
APP_NAME = "Neura.AI v1.1"
DB_FILE = "trade_history.db"
CSV_FILE = "trade_history.csv"
MODEL_FILE = "model.pkl"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_SYMBOL_MAP = {
    "XAUUSD": "OANDA:XAU_USD",
    "BTCUSD": "BINANCE:BTCUSDT",
    "EURUSD": "OANDA:EUR_USD",
    "GBPUSD": "OANDA:GBP_USD",
    "USDJPY": "OANDA:USD_JPY",
    # add as needed
}

BRIDGE_URL = os.getenv("BRIDGE_URL", "")  # e.g., https://<ngrok>.io
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "neura_bridge_token_2025")

# Exness / broker contract size mapping (points -> USD per 0.01 lot)
CONTRACT_POINT_VALUE = {
    # approximate example values, adjust per broker/instrument
    "XAUUSD": 0.10,   # USD per 0.01 lot per point — user may tweak for Exness
    "BTCUSD": 0.01,
    "EURUSD": 0.10,
}

# Strategy constants
EMA_SHORT = 9
EMA_LONG = 15
RR_RATIO = 2.0
SL_POINTS_DEFAULT = 2.5   # stop-loss in points
LOT_SIZE_DEFAULT = 0.01

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neura_ai")

# App init
app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# -------------------- Database --------------------
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl_points REAL,
            pnl_usd REAL,
            rr_ratio REAL,
            ai_confidence REAL,
            setup TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_trade_db(trade: Dict[str, Any]):
    ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (symbol,direction,entry_time,exit_time,entry_price,exit_price,
                            pnl_points,pnl_usd,rr_ratio,ai_confidence,setup)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        trade.get("symbol"),
        trade.get("direction"),
        trade.get("entry_time"),
        trade.get("exit_time"),
        trade.get("entry_price"),
        trade.get("exit_price"),
        trade.get("pnl_points"),
        trade.get("pnl_usd"),
        trade.get("rr_ratio"),
        trade.get("ai_confidence"),
        trade.get("setup")
    ))
    conn.commit()
    conn.close()
    # sync CSV
    df = pd.read_sql_query("SELECT * FROM trades", sqlite3.connect(DB_FILE))
    df.to_csv(CSV_FILE, index=False)


def query_trades(symbol: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict]:
    ensure_db()
    conn = sqlite3.connect(DB_FILE)
    q = "SELECT * FROM trades WHERE 1=1"
    params = []
    if symbol:
        q += " AND symbol=?"; params.append(symbol)
    if start:
        q += " AND datetime(entry_time) >= datetime(?)"; params.append(start)
    if end:
        q += " AND datetime(entry_time) <= datetime(?)"; params.append(end)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df.to_dict(orient="records")


# -------------------- Data fetching --------------------
def fetch_from_bridge(symbol: str, timeframe: str, count: int = 500):
    """Attempt to fetch candles from local bridge (if configured)."""
    if not BRIDGE_URL:
        raise RuntimeError("Bridge URL not configured")
    url = f"{BRIDGE_URL.rstrip('/')}/candles"
    params = {"symbol": symbol, "timeframe": timeframe, "count": count}
    headers = {"X-API-KEY": BRIDGE_API_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    # expecting list of {time, open, high, low, close, volume}
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    return df


def fetch_from_finnhub(symbol: str, resolution: str = "15", count: int = 500):
    """Fetch candles from Finnhub (fallback for cloud)."""
    # symbol is mapped
    mapped = FINNHUB_SYMBOL_MAP.get(symbol, symbol)
    url = "https://finnhub.io/api/v1/forex/candle"
    params = {"symbol": mapped, "resolution": resolution, "count": count, "token": FINNHUB_API_KEY}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub error: {data}")
    df = pd.DataFrame({
        "time": pd.to_datetime(data["t"], unit="s"),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data.get("v", [0]*len(data["c"]))
    })
    return df


def get_candles(symbol: str = "XAUUSD", timeframe: str = "M15", count: int = 500) -> pd.DataFrame:
    """Unified candle fetch: prefer bridge, else Finnhub fallback."""
    tf_map = {"M1": "1", "M5": "5", "M15": "15", "M30": "30", "H1": "60"}
    resolution = tf_map.get(timeframe, "15")
    # try bridge first
    if BRIDGE_URL:
        try:
            df = fetch_from_bridge(symbol, timeframe, count)
            logger.info("Fetched %d candles from bridge for %s", len(df), symbol)
            df = df.sort_values('time').reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning("Bridge fetch failed: %s — falling back to Finnhub", e)
    # fallback to Finnhub
    df = fetch_from_finnhub(symbol, resolution, count)
    df = df.sort_values('time').reset_index(drop=True)
    logger.info("Fetched %d candles from Finnhub for %s", len(df), symbol)
    return df


# -------------------- Indicators & signals --------------------
def compute_emas(df: pd.DataFrame):
    df['EMA9'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['EMA15'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    return df


def detect_crosses(df: pd.DataFrame):
    df = compute_emas(df.copy())
    df['cross'] = None
    for i in range(1, len(df)):
        prev_s, prev_l = df.loc[i-1,'EMA9'], df.loc[i-1,'EMA15']
        curr_s, curr_l = df.loc[i,'EMA9'], df.loc[i,'EMA15']
        if prev_s <= prev_l and curr_s > curr_l:
            df.at[i,'cross'] = 'bull'
        elif prev_s >= prev_l and curr_s < curr_l:
            df.at[i,'cross'] = 'bear'
    return df


def compute_sl_tp(entry_price: float, direction: str, sl_points: float = SL_POINTS_DEFAULT, rr: float = RR_RATIO):
    if direction == 'bull' or direction == 'BUY':
        sl = entry_price - sl_points
        tp = entry_price + sl_points * rr
    else:
        sl = entry_price + sl_points
        tp = entry_price - sl_points * rr
    return float(sl), float(tp)


def calc_pnl_points(entry: float, exit: float, direction: str):
    return float((exit - entry) if direction.upper() == "BUY" else (entry - exit))


def calc_pnl_usd(symbol: str, pnl_points: float, lot: float = LOT_SIZE_DEFAULT):
    per_point = CONTRACT_POINT_VALUE.get(symbol, 0.1)
    # assumed per_point is USD per 0.01 lot per point; scale by lot/0.01
    factor = lot / 0.01
    return round(pnl_points * per_point * factor, 2)


# -------------------- ML: RandomForest --------------------
def train_rf_model(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """Train RF using historical candles (last N) — stores to MODEL_FILE."""
    df = get_candles(symbol, "M15", count=2000)
    df = compute_emas(df)
    # simple supervised target: next candle direction
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    # features
    X = df[['EMA9','EMA15','open','high','low','close']].fillna(0)
    y = df['target'].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    joblib.dump(model, MODEL_FILE)
    logger.info("Trained RF model; score=%.4f", score)
    return {"score": float(score), "model_path": MODEL_FILE}


def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None


def predict_with_model(df: pd.DataFrame):
    model = load_model()
    if model is None:
        return None, None
    X = df[['EMA9','EMA15','open','high','low','close']].tail(1).fillna(0)
    pred = model.predict(X)[0]
    prob = max(model.predict_proba(X)[0]) if hasattr(model, "predict_proba") else None
    return int(pred), float(prob) if prob is not None else None


# -------------------- API Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "symbols": list(FINNHUB_SYMBOL_MAP.keys())})


@app.get("/api/candles")
async def api_candles(symbol: str = "XAUUSD", timeframe: str = "M15", count: int = 500):
    try:
        df = get_candles(symbol, timeframe, count)
        df = compute_emas(df)
        return JSONResponse({"data": df.tail(100).to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict")
async def api_predict(symbol: str = "XAUUSD", timeframe: str = "M15"):
    try:
        df = get_candles(symbol, timeframe, 500)
        df = compute_emas(df)
        # model prediction
        pred, prob = predict_with_model(df)
        # EMA cross suggestion (latest cross within last 6 candles)
        cross_df = detect_crosses(df)
        latest_cross = None
        for i in range(len(cross_df)-1, -1, -1):
            if cross_df.loc[i, 'cross'] is not None:
                latest_cross = (i, cross_df.loc[i,'cross'])
                break
        suggestion = None
        if latest_cross:
            idx, direction = latest_cross
            # find a retest: search next 12 bars for price touching EMA and closing back
            ret = None
            for j in range(idx+1, min(idx+12, len(df))):
                for ema_col in ['EMA9','EMA15']:
                    ema_val = float(df.loc[j, ema_col])
                    if direction == 'bull':
                        if float(df.loc[j,'low']) <= ema_val <= float(df.loc[j,'close']):
                            ret = (j,float(df.loc[j,'close']), ema_col)
                            break
                    else:
                        if float(df.loc[j,'high']) >= ema_val >= float(df.loc[j,'close']):
                            ret = (j,float(df.loc[j,'close']), ema_col)
                            break
                if ret:
                    break
            if ret:
                entry_idx, entry_price, used_ema = ret
                dir_text = 'BUY' if direction=='bull' else 'SELL'
                sl, tp = compute_sl_tp(entry_price, direction)
                suggestion = {
                    "direction": dir_text,
                    "entry_time": str(df.loc[entry_idx,'time']),
                    "entry_price": round(float(entry_price), 4),
                    "sl": round(sl,4),
                    "tp": round(tp,4),
                    "ai_pred": int(pred) if pred is not None else None,
                    "ai_prob": round(prob*100,2) if prob is not None else None,
                    "setup": "EMA9/EMA15 + RF",
                    "rr": f"1:{RR_RATIO}"
                }
        return JSONResponse({"suggestion": suggestion, "model_pred": int(pred) if pred is not None else None, "model_prob": round(prob*100,2) if prob is not None else None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaveTradePayload(BaseModel):
    symbol: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    rr_ratio: float
    ai_confidence: float
    setup: str


@app.post("/api/save_trade")
async def api_save_trade(payload: SaveTradePayload):
    pnl_points = calc_pnl_points(payload.entry_price, payload.exit_price, payload.direction)
    pnl_usd = calc_pnl_usd(payload.symbol, pnl_points, LOT_SIZE_DEFAULT)
    trade = {
        "symbol": payload.symbol,
        "direction": payload.direction,
        "entry_time": payload.entry_time,
        "exit_time": payload.exit_time,
        "entry_price": payload.entry_price,
        "exit_price": payload.exit_price,
        "pnl_points": pnl_points,
        "pnl_usd": pnl_usd,
        "rr_ratio": payload.rr_ratio,
        "ai_confidence": payload.ai_confidence,
        "setup": payload.setup
    }
    save_trade_db(trade)
    return JSONResponse({"status": "ok", "trade": trade})


@app.get("/api/history")
async def api_history(symbol: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None):
    data = query_trades(symbol, start, end)
    return JSONResponse({"data": data})


@app.get("/api/export")
async def api_export():
    ensure_db()
    if not os.path.exists(CSV_FILE):
        df = pd.read_sql_query("SELECT * FROM trades", sqlite3.connect(DB_FILE))
        df.to_csv(CSV_FILE, index=False)
    return FileResponse(CSV_FILE, media_type='text/csv', filename='trade_history.csv')


@app.post("/api/retrain")
async def api_retrain(symbol: str = Form("XAUUSD")):
    res = train_rf_model(symbol)
    return JSONResponse({"status": "ok", "score": res.get("score")})


@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "app": APP_NAME,
        "finnhub": bool(FINNHUB_API_KEY),
        "bridge": bool(BRIDGE_URL),
        "model_exists": os.path.exists(MODEL_FILE),
        "db_exists": os.path.exists(DB_FILE)
    })


# Startup
@app.on_event("startup")
def startup_event():
    ensure_db()
    logger.info("Neura.AI v1.1 backend started")


if __name__ == "__main__":
    ensure_db()
    import uvicorn
    uvicorn.run("xau_dashboard_v5:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
