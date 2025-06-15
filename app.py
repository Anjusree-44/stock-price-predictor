import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info logs

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import normalize_data, create_sequences
import numpy as np

st.title("📈 Stock Price Prediction with LSTM")
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Predict"):
    df = yf.download(ticker, period='5y', auto_adjust=True)
    data, scaler = normalize_data(df)
    x, y = create_sequences(data)
    x = np.array(x)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Load the model (make sure you have a .h5 or .keras file, not a directory)
    model_path = "lstm_stock_model.h5"  # Change to your actual model file
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please train and save your model as .h5 or .keras.")
    else:
        model = load_model(model_path)
        prediction = model.predict(x)
        prediction = scaler.inverse_transform(prediction)
        actual = scaler.inverse_transform(np.array(y).reshape(-1, 1))

        fig, ax = plt.subplots()
        ax.plot(actual, label='Actual')
        ax.plot(prediction, label='Predicted')
        ax.legend()
        st.pyplot(fig)
        #streamlit run "c:\Users\katta\OneDrive\Python-ai\myenv\stock price predictor\app.py"
