import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import base64
import time

from sample_sentiment import get_mock_sentiment

from call_model import load_predmodel, call_data, predict_next_price
from emotion_model import get_sentiment

from binance.client import Client
from requests.exceptions import ConnectionError
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px

show = True

## Binance API 
API_KEY = st.secrets["BINANCE_API_KEY"]
API_SECRET = st.secrets["BINANCE_API_SECRET"]

try:
    client = Client(API_KEY, API_SECRET)
except ConnectionError:
    pass

## functions


def get_base64_img(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# st.button("Sample")

# Streamlit app title
st.title("Cryptone")

st.markdown("""
<style>
.button {
    display: inline-block;
    padding: 0.6em 1.2em;
    margin: 0.5em;
    font-size: 1.1em;
    color: white;
    background-color: #444;
    border-radius: 10px;
    text-decoration: none;
    transition: all 0.2s ease-in-out;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}
.button:hover {
    background-color: #666;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

## switch to 'BTCPHP'

# List of coins
coins = [
    {"name": "Bitcoin", "symbol": "BTC", "img": './static/bitcoin-btc-logo.png', "id": "BTCUSDT", "label": "BTC/USDT", "model": "btc"},
    {"name": "Ethereum", "symbol": "ETH", "img": './static/ethereum-eth-logo.png', "id": "ETHUSDT", "label": "ETH/USDT",  "model": "eth"},
    {"name": "XRP", "symbol": "XRP", "img": './static/xrp-xrp-logo.png', "id": "XRPUSDT", "label": "XRP/USDT",  "model": "xrp"},
    # {"name": "Tether", "symbol": "USDT", "img": './static/tether-usdt-logo.png', "id": "DOGEUSDT", "label": "USDT Conversion Rate"}
]

# Sidebar title
with st.sidebar:
    st.markdown("### Select Cryptocurrency")
    cols = st.columns(2)
    for i, coin in enumerate(coins):
        with cols[i % 2]:
            if st.button(coin["name"], key=coin["symbol"]):
                st.session_state.selected_coin = coin["symbol"]
            # if st.button:
            #     msg = st.empty()
            #     st.success(f"You selected: {coin['name']}")
            #     msg.empty()
            #     time.sleep(3)

## sets BTC as default.
selected_coin = st.session_state.get("selected_coin", coins[0].get('symbol'))

coin = next((c for c in coins if c["symbol"] == selected_coin), coins[0])

# st.header(coin.get("name"))

st.markdown("""
    <style>
    .coin-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding-top: 40px;
    }
    .coin-image {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .coin-image:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

selected_coin_data = next((coin for coin in coins if coin["symbol"] == selected_coin), coins[0])

## variables
scaled_features, scaler, df = call_data(f"{coin.get("model")}")
model = load_predmodel(f"{coin.get("model")}")

predicted = predict_next_price(model, scaled_features)

# Display current price
current_price = df[f'price_{coin.get("model")}'].values[-1]

col1, col2 = st.columns([1,3])

with col1:
    if coin:
        img_base64 = get_base64_img(coin["img"])
        st.markdown(
            f"""
            <div class='coin-container'>
                <img src="data:image/png;base64,{img_base64}" width="150" class="coin-image"/>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
     # Auto-refresh every 10 seconds
    st_autorefresh(interval=10000, key="refresh")

    if show:

        try:
            # # Auto-refresh every 10 seconds
            # st_autorefresh(interval=10000, key="refresh")

            # Get price
            price_data = client.get_symbol_ticker(symbol=coin.get("id"))
            current_price = float(price_data["price"])

            # Delta logic with session state
            if "last_price" not in st.session_state:
                st.session_state.last_price = {}

            last_price = st.session_state.last_price.get(selected_coin)
            price_change = current_price - last_price if last_price is not None else 0.0
            delta = f"{price_change:+.6f}" if last_price is not None else "↺"

            st.session_state.last_price[selected_coin] = current_price

            # Display stylized markdown for price + delta
            st.markdown(
                f"""
                <div style="
                    padding-bottom: 10px;">
                <div style="
                    background-color: #1e1e1e;
                    padding: 1rem;
                    border-radius: 1rem;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
                    color: white;
                ">
                    <h3 style="margin: 0;">{coin.get("name")}</h3>
                    <p style="margin: 0; font-size: 0.9rem; color: #888;">{coin.get("label")}</p>
                    <h1 style="margin: 0; color: #00FFAA; font-size: 2rem;">
                        {current_price:.6f}
                    </h1>
                    <p style="margin: 0; font-size: 1rem; color: {'#00ff00' if price_change >= 0 else '#ff5555'};">
                        {delta}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except ConnectionError:
            pass

        except Exception as e:
            st.error(f'''If you're seeing this message, that just means you're coding at school u sick deranged woman 
                    aren't u aware binance don't work outside cuz starlink diff? Well now you know.''')

    
# sentiment = get_mock_sentiment() ## bullish, high, low... etc
# predicted_price = predict_next_price()

# df = pd.read_csv("C:/Users/steph/Desktop/MIMI/Jupynotebooks/ThesisTrial/empathic - binance ver/dataset/xrp_sentiment_scored.csv")

# # Load LSTM Model
# model = tf.keras.models.load_model("C:/Users/steph/Desktop/MIMI/Jupynotebooks/ThesisTrial/empathic - binance ver/model/xrp_lstm_model.h5")

# # Predict Next Price (Basic Example)
# def predict_next_price():
#     last_prices = df['price_xrp'].values[-5:]
#     last_prices_scaled = np.array(last_prices).reshape(1, -1, 1)
#     prediction = model.predict(last_prices_scaled)
#     return round(float(prediction[0][0]), 2)

# sentiment = df["compound"].values[-1] 


# style for the card

st.markdown("""
<style>
.card {
    padding: 1px;
    margin-bottom: 20px;
    background-color: #0077AB;
    border-radius: 12px;
    text-align: center;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 1.2rem;
    color: #0001E;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Display metrics in columns
col1, col2, col3 = st.columns(3)

## stuffs
df = pd.read_csv(f"./dataset/{coin.get("model")}_sentiment_scored.csv")
sen_score = df['compound'].iloc[5] if not df.empty and 'compound' in df.columns else 0 
sentiment = df['sentiment'].iloc[-1] if not df.empty and 'sentiment' in df.columns else "Neutral" 

# Load predicted prices
pred_df = pd.read_csv(f"./dataset/{coin.get("model")}_predicted_prices.csv")
pred_df["date"] = pd.to_datetime(pred_df["date"])

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">📉 Sentiment Score</div>
        <div class="metric-value">{sen_score}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">🙎 Emotion</div>
        <div class="metric-value">{sentiment}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">🔮 Predicted Price</div>
        <div class="metric-value">{predicted:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# Load and prepare historical data
df = pd.read_csv(f"./dataset/{coin.get("model")}_price_series.csv")  # Update path
df["published_date"] = pd.to_datetime(df["published_date"])
df = df.sort_values("published_date")
df.set_index("published_date", inplace=True)

# Load and prepare predicted data
pred_df = pd.read_csv(f"./dataset/{coin.get("model")}_predicted_prices.csv")  # Should contain columns: date, predicted_price
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df = pred_df.sort_values("date")
pred_df.set_index("date", inplace=True)

# Load and prepare historically predicted data
hist_pred = pd.read_csv(f"./dataset/{coin.get("model")}_hist_pred.csv") # Should contain columns: date, predicted_price
hist_pred['date'] = pd.to_datetime(hist_pred['date'])
hist_pred = hist_pred.sort_values('date')
hist_pred.set_index("date", inplace=True)

# load the sentiment scored ones
histsent = pd.read_csv(f"./dataset/{coin.get("model")}_sentiment_scored.csv")  # Update path
histsent["published_date"] = pd.to_datetime(histsent["published_date"])
histsent = histsent.sort_values("published_date")
histsent.set_index("published_date", inplace=True)
histsent.drop(histsent[histsent['compound'] == 0].index, inplace = True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Forecast", f"💬 LIVE | {coin.get("name")} Sentiment Score", "📊 Raw Sentiment Counts", "🧾 Historical Forecast"])

with tab1:
    # Create the plot
    fig = go.Figure()

    # Add historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[f"price_{coin.get("model")}"],
        mode="lines",
        name="Historical Price",
        line=dict(color="royalblue")
    ))

    # Add predicted prices (make sure dates are after the historical range)
    fig.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df[f"{coin.get("model")}_predicted_price"],
        mode="lines",
        name="Predicted Price",
        line=dict(color="orange", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=hist_pred.index,
        y=hist_pred["model_predicted_price"],
        mode="lines",
        name="Model Prediction (Training)",
        line=dict(color="orange", dash="dot"),
        opacity= 0.5
    ))

    # Layout adjustments
    fig.update_layout(
        title=f"📈 {coin.get("name")} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (in USD)",
        hovermode="x unified",
        hoverlabel=dict(
            font_size=16,
            font_family="Arial"
        ),
        showlegend=True
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)

with tab2:

    # Initialize session state for sentiment log and history
    if "sentiment_log" not in st.session_state:
        st.session_state.sentiment_log = []

    if "sentiment_df" not in st.session_state:
        st.session_state.sentiment_df = pd.DataFrame(columns=["timestamp", "score", "coin"])

    # Fetch new data
    new_sentiments = get_sentiment(f"{coin.get("model")}")

    MAX_LOG_SIZE = 20

    existing_timestamps = set(st.session_state.sentiment_df["timestamp"].astype(str))

    for ts, title, score, emoji in new_sentiments:
        # timestamp = pd.to_datetime(ts).strftime("%H:%M:%S")
        ts = pd.to_datetime(ts) 
        ts += pd.to_timedelta(np.random.randint(1, 1000), unit='ms')  # add jitter

        timestamp = ts.strftime("%H:%M:%S:%f")[:-3]  # include milliseconds

          # Keep only last MAX_LOG_SIZE entries
        if len(st.session_state.sentiment_log) > MAX_LOG_SIZE:
            st.session_state.sentiment_log.pop(0)

         # Keep only latest 5 in log
        st.session_state.sentiment_log = st.session_state.sentiment_log[-MAX_LOG_SIZE:]

        # Add all 6 fields: ts, timestamp, title, score, emoji, coin
        st.session_state.sentiment_log.append((ts, timestamp, title, score, emoji, selected_coin))

        if str(ts) not in existing_timestamps:
            st.session_state.sentiment_df = pd.concat([
                st.session_state.sentiment_df,
                pd.DataFrame([[ts, score, selected_coin]], columns=["timestamp", "score", "coin"])
            ])

    # Display log
    # st.markdown(":violet-badge[:material/star: Sample]")
    st.subheader("🧠 Recent Sentiment Log")

    log_html = "<div style='max-height: 350px; overflow-y: auto; padding-right: 10px;'>"

    for ts, timestamp, title, score, emoji, coin in reversed(st.session_state.sentiment_log):
        log_html += f"""<div>
            <div style='margin-bottom: 8px; padding: 6px; border-radius: 8px; background-color: rgba(255,255,255,0.05);'>
                🕒 <b>{timestamp}</b> | <span style='font-size: 1.2em;'>{emoji}</span>
                (<span style='color: {"#00FFAA" if score > 0 else "#FF6B6B"};'>{score:.2f}</span>) — {title}
            </div>
        """

    log_html += "</div>"

    st.markdown(log_html, unsafe_allow_html=True)

    st.subheader(f"{selected_coin} Sentiment Over Time")
    # Filter chart data to selected coin only
    filtered_df = st.session_state.sentiment_df[
        st.session_state.sentiment_df["coin"] == selected_coin
    ]

    if not filtered_df.empty:
        chart_data = filtered_df.set_index("timestamp")[["score"]].reset_index()

    # Show last N entries
    # chart_data = chart_data[-:]

    # Line trace
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_data["timestamp"],
        y=chart_data["score"],
        mode="lines+markers",
        line=dict(color="grey", width=0.5, dash="dot"),
        marker=dict(
            size=10,
            color=chart_data["score"],  # Color by sentiment
            colorscale=["red", "orange", "green"],
            colorbar=dict(title="Sentiment"),
            line=dict(width=1, color="white")
        ),
        hovertemplate='Time: %{x}<br>Score: %{y:.2f}<extra></extra>',
        name="Sentiment"
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template="plotly_dark",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            font=dict(color="white"),
            align="right")
    )

    st.plotly_chart(fig, use_container_width=True)

    # new_sentiments = get_sentiment(f"{coin.get("symbol")}")

    # for ts, title, score, emoji in new_sentiments:
    #     st.write(f"🕒 {ts.strftime('%H:%M:%S')} | {emoji} ({score:.2f}) — {title}")



with tab3:
     st.write("Coming soon: Raw positive/neutral/negative counts if included in dataset!")

with tab4:
    st.subheader(f"{selected_coin} Historical Sentiment Forecast")
    # Line trace
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=histsent.index,
        y=histsent["compound"],
        mode="lines+markers",
        line=dict(color="grey", width=0.5, dash="dot"),
        marker=dict(
            size=10,
            color=histsent["compound"],  # Color by sentiment
            colorscale=["red", "orange", "green"],
            colorbar=dict(title="Sentiment"),
            line=dict(width=1, color="white")
        ),
        hovertemplate='Date: %{x}<br>Score: %{y:.2f}<extra></extra>',
        name="Sentiment"
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template="plotly_dark",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            font=dict(color="white"),
            align="right")
    )

    st.plotly_chart(fig, use_container_width=True)

    