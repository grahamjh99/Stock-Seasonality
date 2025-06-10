import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
API_KEY = os.getenv("key")

# Load trained model
model = joblib.load("stock_model.pkl")

st.title("📈 7-Day Stock Price Predictor (via Alpha Vantage)")

ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL").upper()

# Get stock data from Alpha Vantage
def get_alpha_vantage_data(symbol):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    if "Time Series (Daily)" not in data:
        return None
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={"5. adjusted close": "Adj Close"})
    df["Adj Close"] = df["Adj Close"].astype(float)
    df = df.sort_index()  # oldest to newest
    df.index = pd.to_datetime(df.index)
    return df[["Adj Close"]]

if ticker:
    df = get_alpha_vantage_data(ticker)
    if df is None or df.empty:
        st.error("Unable to retrieve stock data. Check ticker and API key.")
    else:
        st.subheader(f"{ticker} Adjusted Close (Last {len(df)} Days)")
        st.line_chart(df["Adj Close"])

        # Prepare data for prediction
        last_n = 60  # depends on your model
        if len(df) < last_n:
            st.warning(f"Need at least {last_n} days of data.")
        else:
            input_sequence = df["Adj Close"].values[-last_n:].reshape(1, -1)
            prediction = model.predict(input_sequence)

            # Generate next 7 trading days (skip weekends)
            last_date = df.index[-1]
            future_dates = []
            d = last_date + timedelta(days=1)
            while len(future_dates) < 7:
                if d.weekday() < 5:
                    future_dates.append(d)
                d += timedelta(days=1)

            pred_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Price": prediction.flatten()
            }).set_index("Date")

            st.subheader("📉 Predicted Prices (Next 7 Trading Days)")
            st.line_chart(pred_df)
            st.dataframe(pred_df)
