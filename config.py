"""
config.py
────────────────────────────────────────────────────────────
Single source of truth for all project settings.

Why this pattern?
    Imagine you want to add TSLA to your company list.
    Without config.py: you hunt through 8 files changing
    the list in each one, probably missing one.
    With config.py: you change ONE line here. Done.

    This is called the Single Source of Truth principle.
    It is used in every production system at MAANG.

How to use in any file:
    from config import COMPANIES, RAW_DATA_PATH
────────────────────────────────────────────────────────────
"""

import os

# ── companies to track ────────────────────────────────────
# these are the 5 most-discussed stocks in financial news
# chosen because: high news volume, high liquidity,
# and they represent different sectors (ad-tech, e-commerce,
# cloud, consumer hardware, enterprise software)
COMPANIES = ["AAPL", "GOOGL", "META", "AMZN", "MSFT"]

# ── file paths ────────────────────────────────────────────
# all paths relative to project root
# os.path.join handles Windows backslash vs Mac/Linux
# forward slash automatically — cross-platform safe
RAW_DATA_PATH       = os.path.join("data", "raw",       "all_news.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "news_clean.csv")
FEATURES_PATH       = os.path.join("data", "features",  "model_input.csv")
MODEL_PATH          = os.path.join("models",             "xgboost_sentiment.pkl")
REPORTS_PATH        = os.path.join("reports", "figures")

# ── model settings ────────────────────────────────────────
RANDOM_STATE  = 42      # fixed seed = reproducible results every run
TEST_SIZE     = 0.2     # 80% train, 20% test — standard split
TARGET_COLUMN = "price_direction"   # 1 = up, 0 = down

# ── data collection settings ─────────────────────────────
MAX_ARTICLES_PER_COMPANY = 100     # cap per collection run
MIN_TITLE_WORDS          = 3       # reject titles shorter than this
NEWS_LOOKBACK_DAYS       = 30      # how far back to consider news fresh

# ── sentiment settings ────────────────────────────────────
SENTIMENT_MODEL    = "ProsusAI/finbert"   # HuggingFace model ID
SENTIMENT_BATCH_SIZE = 16                 # articles per FinBERT batch