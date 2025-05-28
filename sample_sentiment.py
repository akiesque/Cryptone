import random

def get_mock_sentiment():
    return round(random.uniform(-1, 1), 2)

# Test the function
print(get_mock_sentiment())
