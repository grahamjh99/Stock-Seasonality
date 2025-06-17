"""Streamlit app: Stock price forecasting with two Random Forest models (t+1 and t+7)
-----------------------------------------------------------------------
Features expected by the models (column order):
    open, high, low, close, volume,
    overall_sentiment_score, overall_sentiment_label,
    ticker_relevance_score,  ticker_sentiment_score,  ticker_sentiment_label
The models are saved as:
    models/best_rf_model-1-day_1.pkl  # predicts close(t+1)
    models/best_rf_model.pkl  # predicts close(t+7)
They are full sklearn Pipelines (scaler + RandomForestRegressor).
"""

import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from pandas.tseries.offsets import BDay

###############################################################################
# Configuration & helpers
###############################################################################

load_dotenv("aplhavantage_api_key.env")
API_KEY = os.getenv("key")
BASE_URL = "https://www.alphavantage.co/query"

# Sentiment‑label mapping used during training
LABEL_MAP = {
    "Somewhat-Bullish": 4,
    "Neutral": 3,
    "Bullish": 5,
    "Somewhat-Bearish": 2,
    "Bearish": 1,
}

@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_daily_prices(symbol: str, full: bool = False) -> pd.DataFrame:
    """Pull TIME_SERIES_DAILY_ADJUSTED and return an OHLCV dataframe (newest first)."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol.upper(),
        "apikey": API_KEY,
        "outputsize": "full" if full else "compact",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "Time Series (Daily)" not in data:
        raise RuntimeError(data.get("Error Message", "Unexpected response from Alpha Vantage"))

    ts = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()  # oldest → newest
    ts = ts.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]]
    ts["volume"] = ts["volume"].astype(float)
    return ts

@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_news_sentiment(symbol: str, page_size: int = 200) -> pd.DataFrame:
    """Pull NEWS_SENTIMENT and aggregate daily averages for the required columns."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol.upper(),
        "sort": "LATEST",
        "limit": str(page_size),
        "apikey": API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "feed" not in data or not data["feed"]:
        # No news = return empty DataFrame with expected structure
        index = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="B")
        empty = pd.DataFrame(0.0, index=index, columns=[
            "overall_sentiment_score",
            "overall_sentiment_label",
            "ticker_relevance_score",
            "ticker_sentiment_score",
            "ticker_sentiment_label",
        ])
        return empty

    rows = []
    for item in data["feed"]:
        date = pd.to_datetime(item["time_published"]).date()
        overall_score = float(item["overall_sentiment_score"])
        overall_label = LABEL_MAP.get(item["overall_sentiment_label"], 3)
        # Each item might cover multiple tickers; find the entry for our symbol
        ticker_row = next((t for t in item["ticker_sentiment"] if t["ticker"] == symbol.upper()), None)
        if ticker_row:
            tick_rel = float(ticker_row["ticker_relevance_score"])
            tick_score = float(ticker_row["ticker_sentiment_score"])
            tick_label = LABEL_MAP.get(ticker_row["ticker_sentiment_label"], 3)
        else:
            tick_rel = tick_score = 0.0
            tick_label = 3
        rows.append(
            {
                "date": date,
                "overall_sentiment_score": overall_score,
                "overall_sentiment_label": overall_label,
                "ticker_relevance_score": tick_rel,
                "ticker_sentiment_score": tick_score,
                "ticker_sentiment_label": tick_label,
            }
        )

    df = pd.DataFrame(rows)
    # Aggregate by date (mean of scores / labels)
    df = (
        df.groupby("date").mean(numeric_only=True)
        .sort_index()
        .rename_axis(index="date")
    )
    return df

###############################################################################
# Feature engineering (match training pipeline)
###############################################################################

def build_feature_df(price_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OHLCV with sentiment and ensure column order matches training."""
    # Align on business days of price_df (sentiment may be sparse)
    sent_df = sent_df.reindex(price_df.index, method="ffill")
    feat = pd.concat([price_df, sent_df], axis=1)
    feat = feat.fillna(0.0)  # fill sentiment gaps with neutral/zero
    feat = feat[[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "overall_sentiment_score",
        "overall_sentiment_label",
        "ticker_relevance_score",
        "ticker_sentiment_score",
        "ticker_sentiment_label",
    ]]
    return feat.astype(float)

###############################################################################
# Load trained models
###############################################################################

MODEL_1_PATH = Path("models/best_rf_model-1-day_1.pkl")  # t+1
MODEL_7_PATH = Path("models/best_rf_model.pkl")  # t+7

if not MODEL_1_PATH.exists() or not MODEL_7_PATH.exists():
    st.stop()

model_1 = joblib.load(MODEL_1_PATH)
model_7 = joblib.load(MODEL_7_PATH)

###############################################################################
# Streamlit UI
###############################################################################

st.title("📈 Stock Price Forecast (1‑day & 7‑day)")

symbol = st.text_input("Enter a stock ticker (e.g., MSFT)", value="MSFT").upper().strip()

if symbol:
    try:
        with st.spinner("Fetching data…"):
            prices = get_daily_prices(symbol, full=False)
            news = get_news_sentiment(symbol)
            features = build_feature_df(prices, news)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    # Split features into in‑sample (for plotting) and the latest row (for forward forecasts)
    X_all = features.copy()
    X_latest = X_all.tail(1)

    # 1‑day ahead predictions for historical plot
    pred1_all = pd.Series(model_1.predict(X_all), index=X_all.index).shift(1)
    # Remove last NaN after shift
    pred1_all = pred1_all[:-1]

    # 7‑day ahead
    pred7_all_raw = model_7.predict(X_all)
    pred7_dates = X_all.index + BDay(7)
    pred7_all = pd.Series(pred7_all_raw, index=pred7_dates)
    # Keep only those within price_df range for plotting comparison
    pred7_hist = pred7_all[pred7_all.index <= prices.index.max()]

    # Forward forecasts (tomorrow & 7‑day future point)
    tomorrow_date = prices.index.max() + BDay(1)
    next7_date = prices.index.max() + BDay(7)

    tomorrow_pred = float(model_1.predict(X_latest))
    next7_pred = float(model_7.predict(X_latest))

    # ------------------ Plot ------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["close"], name="Actual Close"))
    fig.add_trace(go.Scatter(x=pred1_all.index, y=pred1_all, name="Predicted Close (t+1)", line=dict(dash="dot")))
    # Historical t+7 (optional)
    if not pred7_hist.empty:
        fig.add_trace(go.Scatter(x=pred7_hist.index, y=pred7_hist, name="Predicted Close (t+7 hist)", line=dict(dash="dash")))
    # Future 7‑day forecast point
    fig.add_trace(
        go.Scatter(
            x=[next7_date],
            y=[next7_pred],
            mode="markers",
            marker=dict(size=10, symbol="diamond"),
            name=f"t+7 Forecast ({next7_date.date()})",
        )
    )

    fig.update_layout(
        title=f"{symbol} – Actual vs. Predicted Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ Forecast metrics ------------------
    col1, col2 = st.columns(2)
    col1.metric(label=f"Predicted close for {tomorrow_date.date()}", value=f"${tomorrow_pred:,.2f}")
    col2.metric(label=f"Predicted close for {next7_date.date()}", value=f"${next7_pred:,.2f}")

    st.caption("Models: Random Forest (scaled) trained on OHLCV + Alpha Vantage news sentiment.  •  Weekends automatically skipped using business‑day offsets.")