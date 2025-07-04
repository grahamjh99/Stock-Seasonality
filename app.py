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
from sklearn.ensemble import RandomForestRegressor

###############################################################################
# Configuration & helpers
###############################################################################

load_dotenv("aplhavantage_api_key.env")
API_KEY = os.getenv("key")
BASE_URL = "https://www.alphavantage.co/query"

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

    feed_items = data['feed']
    flat_data = []

    for article in feed_items:
        base = {
            'title': article.get('title'),
            'time_published': article.get('time_published'),
            'authors': ", ".join(article.get('authors', [])),
            'summary': article.get('summary'),
            'source': article.get('source'),
            'overall_sentiment_score': article.get('overall_sentiment_score'),
            'overall_sentiment_label': article.get('overall_sentiment_label'),
        }

        # Topics as comma-separated string
        topics = article.get('topics', [])
        topic_names = [t['topic'] for t in topics]
        base['topics'] = ", ".join(topic_names)

        # Ticker sentiment - multiple tickers possible
        for ticker_info in article.get('ticker_sentiment', []):
            if ticker_info.get('ticker') == symbol.upper():
                flat_row = base.copy()
                flat_row['ticker'] = ticker_info.get('ticker')
                flat_row['ticker_relevance_score'] = ticker_info.get('relevance_score')
                flat_row['ticker_sentiment_score'] = ticker_info.get('ticker_sentiment_score')
                flat_row['ticker_sentiment_label'] = ticker_info.get('ticker_sentiment_label')
                flat_data.append(flat_row)
                break  # Only keep data for this symbol

    df = pd.DataFrame(flat_data)
    df['time_published'] = pd.to_datetime(df['time_published'].str[:8], format='%Y%m%d')
    df = df.set_index('time_published')

    label_map = {
        'Somewhat-Bullish': 4, 
        'Neutral': 3, 
        'Bullish': 5, 
        'Somewhat-Bearish': 2,
        'Bearish': 1
    }
    df['overall_sentiment_label'] = df['overall_sentiment_label'].map(label_map)
    df['ticker_sentiment_label'] = df['ticker_sentiment_label'].map(label_map)

    return df


###############################################################################
# Feature engineering (match training pipeline)
###############################################################################

def build_feature_df(price_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """Merge stock prices with sentiment and ensure column order matches training."""
    merged_df = price_df.merge(sent_df,how = 'left',left_index = True, right_index = True)
    merged_df = merged_df.drop(columns = ['title','topics','ticker','authors','summary','source'])
    merged_df = merged_df.fillna(0)
    return merged_df.astype(float)

###############################################################################
# Load trained models
###############################################################################

model_1_path = Path("models/best_rf_model-1-day_1.pkl")  # t+1
model_7_path = Path("models/best_rf_model.pkl")  # t+7

# Extract parameters
pipeline_1 = joblib.load(model_1_path)
pipeline_7 = joblib.load(model_7_path)

rf_1 = pipeline_1.named_steps["rf"]
rf_7 = pipeline_7.named_steps["rf"]

model_1_params = rf_1.get_params()
model_7_params = rf_7.get_params()

@st.cache_resource
def train_models_for_symbol(symbol, _params_t1, _params_t7):
    prices = get_daily_prices(symbol)
    news = get_news_sentiment(symbol)
    features = build_feature_df(prices, news)

    # Targets
    target_t1 = prices["close"].shift(-1).dropna()
    target_t7 = prices["close"].shift(-7).dropna()

    # Align features to target lengths
    features_t1 = features.iloc[: len(target_t1)]
    features_t7 = features.iloc[: len(target_t7)]

    # Train models
    model_t1 = RandomForestRegressor(**_params_t1).fit(features_t1, target_t1)
    model_t7 = RandomForestRegressor(**_params_t7).fit(features_t7, target_t7)

    return model_t1, model_t7, prices, features

###############################################################################
# Streamlit UI
###############################################################################

st.title("📈 Stock Price Forecast (1‑day & 7‑day)")

symbol = st.text_input("Enter a stock ticker (e.g., MSFT)", placeholder="MSFT").upper().strip()

if symbol:
    try:
        with st.spinner("Fetching & training models…"):
            model_1, model_7, prices, features = train_models_for_symbol(symbol, model_1_params, model_7_params)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Split features
    X_all = features.copy()
    X_latest = X_all.tail(1)

    # t+1 predictions
    pred1_all = pd.Series(model_1.predict(X_all), index=X_all.index).shift(1)[:-1]

    # t+7 predictions
    pred7_all_raw = model_7.predict(X_all)
    pred7_dates = X_all.index + BDay(7)
    pred7_all = pd.Series(pred7_all_raw, index=pred7_dates)
    pred7_hist = pred7_all[pred7_all.index <= prices.index.max()]

    # Forecast points
    tomorrow_date = prices.index.max() + BDay(1)
    next7_date = prices.index.max() + BDay(7)

    tomorrow_pred = float(model_1.predict(X_latest))
    next7_pred = float(model_7.predict(X_latest))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["close"], name="Actual Close"))
    fig.add_trace(go.Scatter(x=pred1_all.index, y=pred1_all, name="Predicted Close (t+1)", line=dict(dash="dot")))
    if not pred7_hist.empty:
        fig.add_trace(go.Scatter(x=pred7_hist.index, y=pred7_hist, name="Predicted Close (t+7 hist)", line=dict(dash="dash")))
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

    # Forecast metrics
    col1, col2 = st.columns(2)
    col1.metric(label=f"Predicted close for {tomorrow_date.date()}", value=f"${tomorrow_pred:,.2f}")
    col2.metric(label=f"Predicted close for {next7_date.date()}", value=f"${next7_pred:,.2f}")

    st.caption("Models: Random Forest (retrained per stock) using Alpha Vantage stock price and news sentiment.  •  Weekends skipped with business-day offsets.")
