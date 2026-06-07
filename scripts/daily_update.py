"""
daily_update.py
────────────────────────────────────────────────────────────
Runs every weekday at 10pm UTC via GitHub Actions.

What it does:
    1. Fetches today's financial news for all companies
    2. Scores sentiment with FinBERT
    3. Fetches today's stock prices
    4. Appends new data to existing CSV files
    5. Retrains XGBoost on the expanded dataset
    6. Saves updated model

After 30 days of running:
    - 30x more sentiment data
    - Model accuracy improves significantly
    - Sentiment features start dominating over price

This is what Bloomberg's data pipeline does every day.
────────────────────────────────────────────────────────────
"""

import os
import sys
import pandas as pd
from datetime import datetime

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from config import (
    COMPANIES,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SENTIMENT_FILTERED_PATH,
    FEATURES_PATH,
    BASE_DIR,
)

def run_daily_update():
    print(f"Daily update started: {datetime.now()}")
    print(f"Companies: {COMPANIES}")
    print()

    # step 1: collect new articles
    print("Step 1: Collecting news...")
    from src.data_collector import collect_all_companies
    df_new = collect_all_companies()
    print(f"  Collected: {len(df_new)} articles")

    # step 2: append to existing raw data
    raw_path = RAW_DATA_PATH
    if os.path.exists(raw_path):
        df_existing = pd.read_csv(raw_path)
        df_combined = pd.concat(
            [df_existing, df_new], ignore_index=True
        )
        df_combined = df_combined.drop_duplicates(
            subset=["title"]
        )
    else:
        df_combined = df_new

    df_combined.to_csv(raw_path, index=False)
    print(f"  Total raw articles: {len(df_combined)}")

    # step 3: run EDA cleaning
    print("\nStep 2: Cleaning data...")
    from src.eda import clean_and_save
    df_clean = clean_and_save(df_combined, PROCESSED_DATA_PATH)
    print(f"  Clean articles: {len(df_clean)}")

    # step 4: score sentiment
    print("\nStep 3: Scoring sentiment...")
    from src.sentiment import score_dataset, filter_relevant_articles
    from config import SENTIMENT_SAVE_PATH, SENTIMENT_FILTERED_PATH

    df_scored   = score_dataset(
        df_clean,
        text_column="title",
        save_path=SENTIMENT_SAVE_PATH,
    )
    df_filtered = filter_relevant_articles(df_scored)
    df_filtered.to_csv(SENTIMENT_FILTERED_PATH, index=False)
    print(f"  Scored and filtered: {len(df_filtered)} articles")

    # step 5: rebuild features
    print("\nStep 4: Engineering features...")
    from src.features import build_feature_matrix
    feature_matrix = build_feature_matrix(
        sentiment_path=SENTIMENT_FILTERED_PATH,
        save_path=FEATURES_PATH,
    )
    print(f"  Feature matrix: {feature_matrix.shape}")

    # step 6: retrain model
    print("\nStep 5: Retraining model...")
    from src.model import run_full_pipeline
    model, metrics, feature_names = run_full_pipeline(
        features_path=FEATURES_PATH,
    )
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")

    print(f"\nDaily update complete: {datetime.now()}")
    print(f"Model accuracy: {metrics['accuracy']*100:.1f}%")

if __name__ == "__main__":
    run_daily_update()