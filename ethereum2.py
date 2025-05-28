from binance.client import Client
from datetime import datetime
from dateutil import parser
import pandas as pd
import re


# Set up the Binance client with your API Key and Secret
api_key = 'RqjwPkWHP6C5jfxrU4dPP6pjbTHeZOp1ICuhH8MgQDvTrky4jWgoTFyJ0zjWIZSI'
api_secret = 'ZgjsuMviGySm0yGmGehdsrDEMySBWlZQuR5fGM26azs3djV91bQlQGONTunp08r5'
client = Client(api_key, api_secret)

def safe_parse_date(date_str):
    try:
        # ISO format — unambiguous
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return datetime.strptime(date_str, '%Y-%m-%d')

        # MM/DD/YYYY format — assume U.S. style if slashes and ambiguous
        if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
            return datetime.strptime(date_str, '%m/%d/%Y')

        # Fallback to parser with month-first logic
        return parser.parse(date_str, dayfirst=False)

    except Exception as e:
        print(f"Failed to parse '{date_str}': {e}")
        return None

# ---- STEP 2: HISTORICAL PRICE FETCH ----
def get_eth_price_on_date(date_obj):
    formatted_date = date_obj.strftime('%d %b, %Y')
    klines = client.get_historical_klines('ETHUSDT', Client.KLINE_INTERVAL_1DAY, start_str=formatted_date, limit=1)

    if klines:
        close_price = float(klines[0][4])  # Close price
        return close_price
    else:
        print(f"No data found for {formatted_date}")
        return None

# ---- STEP 3: READ & CLEAN DATA ----
df = pd.read_csv("ethereum_sentiment(1).csv")

# Apply safe date parser
df['parsed_date'] = df['published_date'].apply(safe_parse_date)

# Drop rows that failed to parse
df = df.dropna(subset=['parsed_date'])

# ---- STEP 4: FETCH ETH PRICES ----
price_rec = []

for _, row in df.iterrows():
    date_obj = row['parsed_date']
    price = get_eth_price_on_date(date_obj)
    if price:
        price_rec.append({"Date": row['published_date'], "ETH_Price": price})

# ---- STEP 5: OUTPUT ----
price_df = pd.DataFrame(price_rec)

for index, row in price_df.iterrows():
    print(f"The price of ETH on {row['Date']} was ${row['ETH_Price']}")

price_df.to_csv("eth_prices(1).csv", index=False)

