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

# ── chart colors per company ──────────────────────────────
# defined once here — used by eda.py, sentiment.py,
# features.py, dashboard.py
# keeping colors consistent across all charts makes your
# project look professionally designed, not cobbled together
COLORS = {
    "AAPL" : "#5B8DB8",
    "GOOGL": "#E8834D",
    "META" : "#6BAF7A",
    "AMZN" : "#C9637A",
    "MSFT" : "#9B7EC8",
}
PALETTE = list(COLORS.values())

# ── file paths ────────────────────────────────────────────
# BASE_DIR is the absolute path to your project root
# Every other path is built from this
# Why: eliminates all relative path confusion permanently
# Whether code runs from notebooks/, src/, or root —
# all paths resolve to the same place every time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw",       "all_news.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "news_clean.csv")
FEATURES_PATH       = os.path.join(BASE_DIR, "data", "features",  "model_input.csv")
MODEL_PATH          = os.path.join(BASE_DIR, "models",             "xgboost_sentiment.pkl")
REPORTS_PATH        = os.path.join(BASE_DIR, "reports",            "figures")
SENTIMENT_SAVE_PATH  = os.path.join(BASE_DIR, "data", "processed", "news_sentiment.csv")

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
SENTIMENT_SAVE_PATH  = os.path.join(
    BASE_DIR, "data", "processed", "news_sentiment.csv"
)
SENTIMENT_FILTERED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "news_sentiment_filtered.csv"
)