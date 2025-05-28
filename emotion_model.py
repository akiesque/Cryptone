import nltk
import requests
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
your_auth_token = ""

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

## KAOMOJI will do it later ehe
def to_emoji(score):
    if score > 0.01:
        return "😄"  # happy
    elif score < -0.1:
        return "😢"  # sad
    else:
        return "😐"  # neutral
    
# Function to fetch and score news
def get_sentiment(symbol):
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={your_auth_token}"
    response = requests.get(url)
    data = response.json()


    sentiments = []
    for post in data.get("results", []):
        title = post.get("title", "")
        published_at = post.get("published_at", "")

        ## base on selected coin so news are only to specified coin.
        if symbol.upper() in title.upper():
            score = sia.polarity_scores(title)["compound"]
            emoji = to_emoji(score)
            sentiments.append((published_at, title, score, emoji))
        else:
            continue

    return sentiments