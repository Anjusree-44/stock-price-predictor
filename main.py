import numpy as np
import matplotlib.pyplot as plt
from preprocess import fetch_data, normalize_data, create_sequences
from model import build_model
from utils import add_indicators

import tensorflow as tf

# 1. Fetch and preprocess data
df = fetch_data('AAPL')
df = add_indicators(df)
scaled_data, scaler = normalize_data(df)

# 2. Sequence preparation
x, y = create_sequences(scaled_data)
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# 3. Train/test split
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# 4. Build and train model
model = build_model((x_train.shape[1], 1))
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5. Predict
predicted = model.predict(x_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 6. Plot
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# 7. Save model
model.save("lstm_stock_model.h5")

