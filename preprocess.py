import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker='AAPL', period='5y'):
    df = yf.download(ticker, period=period)
    return df

def normalize_data(df, feature='Close'):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature]])
    return scaled_data, scaler

def create_sequences(data, time_step=60):
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i-time_step:i])
        y.append(data[i])
    return x, y
