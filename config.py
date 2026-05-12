"""
config.py
Central configuration — all settings in one place.
Import this in any notebook or module.
"""

# companies to track
COMPANIES = ["AAPL", "GOOGL", "META", "AMZN", "MSFT"]

# file paths
RAW_DATA_PATH       = "data/raw/all_news.csv"
PROCESSED_DATA_PATH = "data/processed/news_clean.csv"
FEATURES_PATH       = "data/features/model_input.csv"
MODEL_PATH          = "models/xgboost_sentiment.pkl"

# model settings
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
TARGET_COLUMN = "price_direction"
