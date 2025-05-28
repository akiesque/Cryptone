import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load Sample Data
df = pd.read_csv("crypto_prices.csv")
df['price'] = df['price'].astype(float)

# Normalize Prices
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['price'].values.reshape(-1,1))

# Prepare Data for LSTM
X, Y = [], []
for i in range(5, len(scaled_data)):
    X.append(scaled_data[i-5:i, 0])  # Last 5 days' prices
    Y.append(scaled_data[i, 0])      # Next day's price

# Convert to NumPy arrays
X, Y = np.array(X), np.array(Y)

# Handle empty dataset case
if X.shape[0] == 0:
    print("Error: Not enough data points for training!")
else:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Define LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, batch_size=1, epochs=3)

    # Save model
    model.save("lstm_model.h5")
    print("Model training complete and saved!")
