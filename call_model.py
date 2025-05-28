## keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

## load
def load_predmodel(symbol): ## change model based on selected coin
    return load_model(f"./model/{symbol}_lstm_model.h5")

# Load and preprocess data
def call_data(symbol):
    df = pd.read_csv(f"./dataset/{symbol}_sentiment_scored.csv")
    df['published_date'] = pd.to_datetime(df['published_date']).dt.normalize()
    df.sort_values('published_date', inplace=True)

    df['price_xrp'] = df[f'price_{symbol}'].astype(float)
    df['sentiment_score'] = df['sentiment'].map({
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }).fillna(0)

    daily_df = df.groupby('published_date').agg({
        'sentiment_score': 'mean',
        f'price_{symbol}': 'mean'
    }).reset_index()

    # Just simulate residuals here if ARIMA isn't in this file
    daily_df['residual'] = daily_df[f'price_{symbol}'].diff().fillna(0)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_df[['sentiment_score', 'residual']])

    return scaled_features, scaler, daily_df

# Predict next price
def predict_next_price(model, scaled_features):
    last_seq = scaled_features[-5:] 
    X_pred = np.reshape(last_seq, (1, 5, 2))
    prediction = model.predict(X_pred)
    return prediction[0][0]