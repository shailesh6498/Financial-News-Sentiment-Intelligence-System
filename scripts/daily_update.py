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
────────────────────────────────────────────────────────────
"""

import os
import sys
import pandas as pd
from datetime import datetime

# add project root to path so imports work from scripts/ folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from config import (
    COMPANIES,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SENTIMENT_SAVE_PATH,
    SENTIMENT_FILTERED_PATH,
    FEATURES_PATH,
    BASE_DIR,
)


def add_date_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds date_clean column to any DataFrame that is missing it.

    WHY this function exists:
        collect_all_companies() saves articles with a 'date'
        column formatted as "11 May 2026, 02:03 PM UTC".
        It does NOT add date_clean (YYYY-MM-DD format) —
        that is done by Phase 2 EDA cleaning.

        But daily_update.py needs to merge old and new data
        BEFORE running EDA cleaning, so we need to derive
        date_clean here ourselves.

    WHY we handle both old and new format:
        Old files (from Phase 1 notebooks) may have date
        in different formats. We try the known format first,
        then fall back to pandas auto-parsing.
    """
    if 'date_clean' in df.columns:
        return df  # already has it, nothing to do

    if 'date' not in df.columns:
        df['date_clean'] = 'unknown'
        return df

    def parse_one_date(d):
        if pd.isna(d) or d == '':
            return 'unknown'
        # try the standard format our collector produces
        try:
            return datetime.strptime(
                str(d), "%d %B %Y, %I:%M %p UTC"
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass
        # try pandas auto-parse as fallback
        try:
            return pd.to_datetime(str(d)).strftime("%Y-%m-%d")
        except Exception:
            return 'unknown'

    df = df.copy()
    df['date_clean'] = df['date'].apply(parse_one_date)
    return df


def run_daily_update():
    print(f"Daily update started: {datetime.now()}")
    print(f"Companies: {COMPANIES}")
    print()

    raw_path = RAW_DATA_PATH

    # ── step 0: load existing data before collection wipes it ──
    # CRITICAL: collect_all_companies() overwrites RAW_DATA_PATH
    # with only today's 50 articles. We must read existing history
    # into memory FIRST before calling it.
    if os.path.exists(raw_path):
        df_existing = pd.read_csv(raw_path)
        df_existing = add_date_clean(df_existing)
        df_existing = df_existing[
            df_existing['date_clean'] != 'unknown'
        ].copy()
        print(f"Existing articles before collection : {len(df_existing)}")
        if len(df_existing) > 0:
            print(
                f"Existing dates                      : "
                f"{sorted(df_existing['date_clean'].unique())}"
            )
    else:
        df_existing = pd.DataFrame()
        print("No existing data found — starting fresh")

    # ── step 1: collect new articles ───────────────────────────
    # This OVERWRITES raw_path with today's 50 articles only.
    # df_existing above already captured the full history safely.
    print("\nStep 1: Collecting news...")
    from src.data_collector import collect_all_companies
    df_new = collect_all_companies()

    # add date_clean to new articles as well
    df_new = add_date_clean(df_new)
    df_new = df_new[df_new['date_clean'] != 'unknown'].copy()
    print(f"  New articles collected : {len(df_new)}")
    print(
        f"  New dates              : "
        f"{sorted(df_new['date_clean'].unique())}"
    )

    # ── step 2: merge old history with new articles ─────────────
    if not df_existing.empty:
        df_combined = pd.concat(
            [df_existing, df_new], ignore_index=True
        )
    else:
        df_combined = df_new.copy()

    # deduplicate — same article fetched on multiple days = keep once
    df_combined = df_combined.drop_duplicates(
        subset=["title"], keep="first"
    )
    df_combined = df_combined.sort_values(
        ["ticker", "date_clean"]
    ).reset_index(drop=True)

    # save the TRUE combined dataset back to raw path
    # this overwrites the today-only file from collect_all_companies
    df_combined.to_csv(raw_path, index=False)
    print(f"\nCombined total : {len(df_combined)} articles")
    print(
        f"Combined dates : "
        f"{sorted(df_combined['date_clean'].unique())}"
    )

    # ── step 3: clean ───────────────────────────────────────────
    print("\nStep 2: Cleaning data...")
    from src.eda import clean_and_save
    df_clean = clean_and_save(df_combined, PROCESSED_DATA_PATH)
    print(f"  Clean articles: {len(df_clean)}")

    # ── step 4: score sentiment ─────────────────────────────────
    print("\nStep 3: Scoring sentiment with FinBERT...")
    from src.sentiment import score_dataset, filter_relevant_articles

    df_scored = score_dataset(
        df_clean,
        text_column="title",
        save_path=SENTIMENT_SAVE_PATH,
    )
    df_filtered = filter_relevant_articles(df_scored)
    df_filtered.to_csv(SENTIMENT_FILTERED_PATH, index=False)
    print(
        f"  Scored: {len(df_scored)} | "
        f"Filtered: {len(df_filtered)}"
    )
    print(
        f"  Sentiment dates: "
        f"{sorted(df_filtered['date_clean'].unique())}"
    )

    # ── step 5: build features ──────────────────────────────────
    print("\nStep 4: Engineering features...")
    import gc
    import torch
    # free FinBERT memory before feature engineering + training
    # WHY: FinBERT (~500MB) + XGBoost training together can
    # exceed GitHub Actions 7GB RAM limit
    del df_scored
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass  # no GPU — fine, just skip

    from src.features import build_feature_matrix

    # delete old feature matrix so it rebuilds completely fresh
    if os.path.exists(FEATURES_PATH):
        os.remove(FEATURES_PATH)

    feature_matrix = build_feature_matrix(
        sentiment_path=SENTIMENT_FILTERED_PATH,
        save_path=FEATURES_PATH,
    )
    real_sentiment_rows = int(
        (feature_matrix['daily_avg_sentiment'] != 0).sum()
    )
    print(f"  Feature matrix shape     : {feature_matrix.shape}")
    print(f"  Rows with real sentiment : {real_sentiment_rows}")

    # ── step 6: retrain model ───────────────────────────────────
    print("\nStep 5: Retraining XGBoost...")
    from src.model import run_full_pipeline

    model, metrics, feature_names = run_full_pipeline(
        features_path=FEATURES_PATH,
    )
    print(f"  Accuracy : {metrics['accuracy']*100:.1f}%")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

    # ── done ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Daily update complete : {datetime.now()}")
    print(f"Total articles        : {len(df_combined)}")
    print(f"Model accuracy        : {metrics['accuracy']*100:.1f}%")
    print(f"AUC-ROC               : {metrics['auc_roc']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_daily_update()