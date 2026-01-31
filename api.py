from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from pathlib import Path
import os
import traceback
import tensorflow as tf
import keras

from tensorflow.keras.models import load_model

app = FastAPI(title="Crypto + Stock Predictor", version="1.0")

@app.get("/debug")
def debug():
    def list_dir(p):
        try:
            return sorted(os.listdir(p))
        except Exception as e:
            return [f"ERROR: {e}"]

    return {
        "python": os.sys.version,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "models_dir": str(MODELS_DIR),
        "data_dir": str(DATA_DIR),
        "models_files": list_dir(MODELS_DIR),
        "data_files": list_dir(DATA_DIR),
    }


# -------- Config --------
LOOKBACK = 60

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# -------- Utilities --------
def safe_name(asset: str) -> str:
    # BTC-USD -> btc-usd
    return asset.lower().replace("/", "-")


def next_dates(last_date: pd.Timestamp, horizon: int, business_days: bool) -> pd.DatetimeIndex:
    if business_days:
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def load_series_for_sl20() -> pd.Series:
    csv_path = DATA_DIR / "sl20_synthetic.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "SL20_Synthetic" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="sl20_synthetic.csv must have columns: Date, SL20_Synthetic"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    series = df["SL20_Synthetic"].dropna()
    if len(series) < LOOKBACK + 10:
        raise HTTPException(status_code=500, detail="Not enough SL20 data for prediction.")
    return series


# ✅ UPDATED: more reliable on Render than yf.download()
def load_series_for_crypto(asset: str) -> pd.Series:
    t = yf.Ticker(asset)
    df = t.history(period="max", interval="1d", auto_adjust=False)

    if df is None or df.empty or "Close" not in df.columns:
        df = t.history(period="5y", interval="1d", auto_adjust=False)

    if df is None or df.empty or "Close" not in df.columns:
        raise HTTPException(status_code=503, detail=f"Yahoo data unavailable for {asset} right now")

    series = df["Close"].dropna()
    series.index = pd.to_datetime(series.index)
    if len(series) < LOOKBACK + 10:
        raise HTTPException(status_code=500, detail=f"Not enough {asset} data for prediction.")
    return series


def forecast_series(series: pd.Series, model_path: Path, scaler_path: Path, horizon: int, business_days: bool):
    try:
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing model: {model_path.name}")
        if not scaler_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing scaler: {scaler_path.name}")

        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        values = series.values.reshape(-1, 1)
        scaled = scaler.transform(values)

        window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)

        preds_scaled = []
        current = window.copy()

        for _ in range(horizon):
            pred = model.predict(current, verbose=0)[0, 0]
            preds_scaled.append(pred)
            current = np.append(current[0, 1:, 0], pred).reshape(1, LOOKBACK, 1)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

        last_date = pd.to_datetime(series.index[-1])
        dates = next_dates(last_date, horizon, business_days)

        # ✅ FIX: avoid pandas FutureWarning for float(Series)
        last_val = series.iloc[-1]
        last_val = float(last_val.item()) if hasattr(last_val, "item") else float(last_val)

        return {
            "last_date": last_date.strftime("%Y-%m-%d"),
            "last_value": last_val,
            "predictions": [{"date": d.strftime("%Y-%m-%d"), "yhat": float(y)} for d, y in zip(dates, preds)]
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")



# -------- Routes --------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(asset: str = "BTC-USD", horizon: int = 7):
    if horizon < 1 or horizon > 30:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 30")

    asset = asset.upper().strip()

    # ---- SL20 Synthetic (business days) ----
    if asset == "SL20_SYN":
        series = load_series_for_sl20()
        model_path = MODELS_DIR / "sl20_lstm.keras"
        scaler_path = MODELS_DIR / "sl20_scaler.pkl"

        out = forecast_series(series, model_path, scaler_path, horizon, business_days=True)
        out["asset"] = asset
        return out

    # ---- Crypto (calendar days) ----
    series = load_series_for_crypto(asset)

    safe = safe_name(asset)
    model_path = MODELS_DIR / f"{safe}_lstm.keras"
    scaler_path = MODELS_DIR / f"{safe}_scaler.pkl"

    out = forecast_series(series, model_path, scaler_path, horizon, business_days=False)
    out["asset"] = asset
    return out


# ✅ NEW: history endpoint for Streamlit "Actual + Forecast"
@app.get("/history")
def history(asset: str = "BTC-USD", period_days: int = 365):
    asset = asset.upper().strip()

    if period_days < 30 or period_days > 3650:
        raise HTTPException(status_code=400, detail="period_days must be 30..3650")

    if asset == "SL20_SYN":
        series = load_series_for_sl20().tail(period_days)
        out = [{"date": idx.strftime("%Y-%m-%d"), "close": float(val)} for idx, val in series.items()]
        return {"asset": asset, "history": out}

    series = load_series_for_crypto(asset).tail(period_days)
    out = [{"date": idx.strftime("%Y-%m-%d"), "close": float(val)} for idx, val in series.items()]
    return {"asset": asset, "history": out}