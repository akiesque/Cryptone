# from binance.client import Client
# from datetime import datetime
# import pandas as pd
# import time
# import streamlit as st

# # Set up the Binance client with your API Key and Secret
# API_KEY = st.secrets["BINANCE_API_KEY"]
# API_SECRET = st.secrets["BINANCE_API_SECRET"]
# client = Client(API_KEY, API_SECRET)

# # Function to fetch historical data for XRP on a specific date
# def get_xrp_price_on_date(date_str):
#     # Convert date to datetime object
#     date = datetime.strptime(date_str, '%d/%m/%Y')
    
#     # Convert to human-readable format Binance expects
#     formatted_date = date.strftime('%d %b, %Y')

#     # Fetch a single day's worth of klines from that date onward
#     klines = client.get_historical_klines('XRPUSDT', Client.KLINE_INTERVAL_1DAY, start_str=formatted_date, limit=1)
    
#     if klines:
#         close_price = float(klines[0][4])  # Close price is at index 4 in the kline data
#         return close_price
#     else:
#         print(f"No data found for {date_str}")
#         return None


# df = pd.read_csv("xrp_sentiment(1).csv")


# # List of dates to check
# dates = df['published_date'].tolist()

# # # Output the collected prices
# # for date, price in prices.items():
# #     print(f"The price of XRP on {date} was ${price}")

# price_rec = []

# # Fetch and print the XRP price for each date
# prices = {}
# for date in dates:
#     price = get_xrp_price_on_date(date)
#     if price:
#         price_rec.append({"Date": date, "XRP_Price": price})

# price_df = pd.DataFrame(price_rec)

# # Print the collected prices
# for index, row in price_df.iterrows():
#     print(f"The price of XRP on {row['Date']} was ${row['XRP_Price']}")

# # Export the DataFrame to a CSV file
# price_df.to_csv("xrp_prices(1).csv", index=False)

