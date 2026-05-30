"""
features.py
────────────────────────────────────────────────────────────
Feature engineering for the Financial Sentiment System.

WHAT this file does:
    Takes sentiment scores + stock prices and builds a
    feature matrix ready for ML model training.

WHY feature engineering matters:
    Raw data tells you WHAT happened.
    Features tell you WHY it matters.

    Example:
    Raw: sentiment_score = +0.85 on 2026-05-11
    Feature: sentiment_3day_rolling_avg = +0.62
             (sentiment has been positive for 3 days)

    The rolling average is more predictive than the single
    day score because markets respond to sustained sentiment
    trends, not single articles.

    This is the core insight of quantitative finance —
    signals need context to become predictions.

REAL WORLD CONTEXT:
    Two Sigma (hedge fund, $60B AUM) employs 300+ data
    scientists who spend most of their time on feature
    engineering, not model selection. At MAANG, the DS
    teams that build recommendation and ad systems spend
    60-70% of their time engineering features.
    A great feature with a simple model beats a simple
    feature with a complex model almost every time.

Used by:
    notebooks/04_feature_engineering.ipynb
    app/dashboard.py (Phase 6)
────────────────────────────────────────────────────────────
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import (
    COMPANIES,
    SENTIMENT_FILTERED_PATH,
    FEATURES_PATH,
    PRICE_LOOKBACK_DAYS,
    LAG_DAYS,
    ROLLING_WINDOW,
    TARGET_COLUMN,
    BASE_DIR,
)
from src.utils import get_logger, ensure_dirs

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# STEP 1 — FETCH STOCK PRICE DATA
# ══════════════════════════════════════════════════════════

def fetch_stock_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetches daily OHLCV stock prices from Yahoo Finance.

    WHAT is OHLCV?
        O = Open  — price when market opened that day
        H = High  — highest price reached during the day
        L = Low   — lowest price reached during the day
        C = Close — price when market closed that day
        V = Volume — how many shares were traded

    WHY do we need all five?
        Close price = what the stock was worth end of day
        Volume = how many people were trading (high volume
                 on a down day = strong selling conviction)
        High-Low range = how volatile the day was

    WHY Yahoo Finance via yfinance?
        Free, reliable, returns clean pandas DataFrames,
        covers all major exchanges, goes back 50+ years.
        Bloomberg costs $24,000/year for the same data.
        For a portfolio project, yfinance is industry standard.

    Args:
        tickers: list of stock tickers e.g. ["AAPL", "GOOGL"]
        start_date: "YYYY-MM-DD" format
        end_date: "YYYY-MM-DD" format

    Returns:
        DataFrame with columns: ticker, date, open, high,
        low, close, volume
    """
    import yfinance as yf

    logger.info(f"Fetching prices for {tickers}")
    logger.info(f"Date range: {start_date} to {end_date}")

    all_prices = []

    for ticker in tickers:
        try:
            # fetch raw price data
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,   # suppress download bar
                auto_adjust=True, # adjust for stock splits/dividends
            )

            if raw.empty:
                logger.warning(f"No price data returned for {ticker}")
                continue

            # flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # standardise column names to lowercase
            raw.columns = [c.lower() for c in raw.columns]

            # add ticker column and clean date
            raw["ticker"]     = ticker
            raw["date_clean"] = raw.index.strftime("%Y-%m-%d")
            raw               = raw.reset_index(drop=True)

            # keep only what we need
            cols_to_keep = [
                "ticker", "date_clean",
                "open", "high", "low", "close", "volume"
            ]
            available = [c for c in cols_to_keep if c in raw.columns]
            raw = raw[available]

            # final verification before appending
            if "ticker" not in raw.columns:
                logger.error(
                    f"ticker column missing for {ticker} "
                    f"— adding explicitly"
                )
                raw["ticker"] = ticker

            if "date_clean" not in raw.columns:
                logger.error(
                    f"date_clean missing for {ticker}"
                )
                continue

            logger.info(
                f"  {ticker}: {len(raw)} days | "
                f"columns: {raw.columns.tolist()}"
            )
            all_prices.append(raw)

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            continue

    if not all_prices:
        raise ValueError(
            "No price data fetched for any ticker. "
            "Check your internet connection and date range."
        )

    prices_df = pd.concat(all_prices, ignore_index=True)
    logger.info(
        f"Total price rows fetched: {len(prices_df)} "
        f"across {prices_df['ticker'].nunique()} companies"
    )

    return prices_df


# ══════════════════════════════════════════════════════════
# STEP 2 — ENGINEER PRICE FEATURES
# ══════════════════════════════════════════════════════════

def engineer_price_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from raw OHLCV price data.

    WHAT features we create and WHY each matters:

    1. daily_return
       Formula: (close - prev_close) / prev_close * 100
       Why: absolute price is meaningless across companies.
       Apple at $189 and Google at $141 are incomparable.
       Return % is comparable — both up 1% means both
       gained equally relative to their starting point.

    2. price_range
       Formula: (high - low) / close * 100
       Why: measures daily volatility. A stock with high
       range is moving a lot — risky but potentially
       responsive to news. Used by traders as a
       volatility proxy.

    3. volume_change
       Formula: (volume - prev_volume) / prev_volume * 100
       Why: unusually high volume on a down day = strong
       selling conviction. High volume on an up day =
       strong buying conviction. Volume confirms price.

    4. price_vs_5day_avg
       Formula: close / 5_day_rolling_mean - 1
       Why: is today's price above or below recent trend?
       If price is 5% above its 5-day average, it may
       be overextended and due for a pullback. This is
       called mean reversion — a core quant finance concept.

    5. TARGET: price_direction
       Formula: 1 if tomorrow's close > today's close else 0
       Why: this is what we predict. Not the exact price
       (too hard, too noisy) but the DIRECTION (up or down).
       Binary classification is more tractable than
       regression for noisy financial data.

    Args:
        prices_df: DataFrame from fetch_stock_prices()

    Returns:
        DataFrame with engineered price features added
    """
    df = prices_df.copy()
    
    #verify ticker column exists before any operations
    #if missing, we cannot seperate companies later
    if "ticker" not in df.columns:
        raise ValueError(
            "ticker column missing from price data."
            "Check fetch_stock_prices output."
        )
    logger.info(
        f"Engineering features for tickers: "
        f"{df['ticker'].unique().tolist()}"
    )

    # sort by ticker then date — critical for lag calculations
    df = df.sort_values(
        ["ticker", "date_clean"]
    ).reset_index(drop=True)

    logger.info("Engineering price features...")

    # ── feature 1: daily return % ─────────────────────────
    # .groupby().shift(1) gives you the PREVIOUS row's value
    # for each ticker independently
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["daily_return"] = (
        (df["close"] - df["prev_close"]) / df["prev_close"] * 100
    ).round(4)

    # ── feature 2: daily price range (volatility proxy) ───
    df["price_range"] = (
        (df["high"] - df["low"]) / df["close"] * 100
    ).round(4)

    # ── feature 3: volume change % ────────────────────────
    df["prev_volume"] = df.groupby("ticker")["volume"].shift(1)
    df["volume_change"] = (
        (df["volume"] - df["prev_volume"]) /
        df["prev_volume"] * 100
    ).round(4)

    # ── feature 4: price vs 5-day rolling average ─────────
    # min_periods=1 means calculate even with fewer than 5 days
    df["price_5day_avg"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df["price_vs_5day_avg"] = (
        (df["close"] / df["price_5day_avg"] - 1) * 100
    ).round(4)

    # ── TARGET: price direction ────────────────────────────
    # shift(-1) gives you the NEXT row's value
    # 1 = price went UP tomorrow, 0 = price went DOWN
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df[TARGET_COLUMN] = (
        df["next_close"] > df["close"]
    ).astype(int)

    # drop helper columns — they served their purpose
    df = df.drop(
        columns=["prev_close", "prev_volume",
                 "price_5day_avg", "next_close"],
        errors="ignore"
    )

    # verify ticker survived all operations
    assert "ticker" in df.columns, \
    "ticker column lost during feature engineering"

    # the last row per ticker has no "next day" — mark as NaN
    last_rows = df.groupby("ticker").tail(1).index
    df.loc[last_rows, TARGET_COLUMN] = np.nan

    logger.info(
        f"Price features engineered. "
        f"Columns: {df.columns.tolist()}"
    )

    return df


# ══════════════════════════════════════════════════════════
# STEP 3 — AGGREGATE SENTIMENT TO DAILY LEVEL
# ══════════════════════════════════════════════════════════

def aggregate_daily_sentiment(
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregates article-level sentiment to daily level.

    WHY aggregate?
        Your sentiment data has one row per article.
        Your price data has one row per trading day.
        To merge them you need both at the same granularity.
        This function converts article-level → daily-level.

    WHAT aggregations we create:

    1. daily_avg_sentiment
       Average sentiment score across all articles that day.
       The main signal.

    2. daily_article_count
       How many articles published that day.
       More articles = more information = more reliable signal.
       One article on a given day is weak signal.
       Five articles all negative = strong signal.

    3. positive_ratio
       What fraction of today's articles are positive.
       Complements the average score.

    4. sentiment_momentum
       Today's average minus yesterday's average.
       Captures whether sentiment is IMPROVING or WORSENING.
       A worsening sentiment trend is often more predictive
       than the absolute level.

    5. sentiment_volatility
       Standard deviation of scores within the day.
       High volatility = mixed signals = uncertain day.
       Low volatility = consistent signal = reliable day.

    Args:
        sentiment_df: article-level sentiment DataFrame

    Returns:
        daily-level sentiment DataFrame
        one row per ticker per trading day
    """
    logger.info("Aggregating sentiment to daily level...")

    df = sentiment_df.copy()

    # aggregate by ticker + date
    daily = df.groupby(["ticker", "date_clean"]).agg(
        daily_avg_sentiment  = ("sentiment_score",  "mean"),
        daily_article_count  = ("sentiment_score",  "count"),
        positive_ratio       = ("sentiment_label",
                                lambda x: (x=="positive").sum() / len(x)),
        negative_ratio       = ("sentiment_label",
                                lambda x: (x=="negative").sum() / len(x)),
        sentiment_volatility = ("sentiment_score",  "std"),
        avg_confidence       = ("sentiment_confidence", "mean"),
    ).round(4).reset_index()

    # fill NaN volatility (happens when only 1 article that day)
    # single article = no volatility to measure = fill with 0
    daily["sentiment_volatility"] = daily[
        "sentiment_volatility"
    ].fillna(0)

    # sort for lag calculations
    daily = daily.sort_values(
        ["ticker", "date_clean"]
    ).reset_index(drop=True)

    # sentiment momentum — day over day change
    daily["sentiment_momentum"] = daily.groupby("ticker")[
        "daily_avg_sentiment"
    ].diff().round(4)

    logger.info(
        f"Daily sentiment aggregated: "
        f"{len(daily)} rows "
        f"({daily['ticker'].nunique()} companies, "
        f"{daily['date_clean'].nunique()} unique dates)"
    )

    return daily


# ══════════════════════════════════════════════════════════
# STEP 4 — MERGE SENTIMENT WITH PRICES
# ══════════════════════════════════════════════════════════

def merge_sentiment_and_prices(
    daily_sentiment: pd.DataFrame,
    price_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges daily sentiment with daily price features.

    WHAT type of merge and WHY:

    We use a LEFT JOIN from prices onto sentiment.
    Why prices on left?
        Price data exists every trading day.
        Sentiment data only exists on days with news.
        Left join keeps ALL price days — days with no
        news get NaN sentiment (handled below).

    WHY not inner join?
        Inner join would only keep days where BOTH
        sentiment AND price exist. This loses many
        price-only days which contain important
        price trend information your model needs.

    REAL WORLD PARALLEL:
        Bloomberg's quant team uses exactly this join
        strategy — price data is the spine, alternative
        data (sentiment, satellite imagery, credit card
        data) is joined onto it. Days with missing
        alternative data get forward-filled.

    Args:
        daily_sentiment: output of aggregate_daily_sentiment()
        price_features: output of engineer_price_features()

    Returns:
        merged DataFrame with all features combined
    """
    logger.info("Merging sentiment with price features...")

    # merge on ticker + date_clean (both DataFrames have this)
    merged = pd.merge(
        price_features,
        daily_sentiment,
        on=["ticker", "date_clean"],
        how="left",   # keep all price rows
    )

    # ── handle missing sentiment days ─────────────────────
    # days with no news → fill with neutral (0.0)
    # WHY fill with 0 not drop?
    # Dropping removes price data we need for trend features.
    # 0.0 (neutral) is the correct assumption for no-news days
    # — absence of news is not negative news.

    sentiment_cols = [
        "daily_avg_sentiment",
        "daily_article_count",
        "positive_ratio",
        "negative_ratio",
        "sentiment_volatility",
        "avg_confidence",
        "sentiment_momentum",
    ]

    for col in sentiment_cols:
        if col in merged.columns:
            if col == "daily_article_count":
                merged[col] = merged[col].fillna(0)
            else:
                merged[col] = merged[col].fillna(0.0)

    logger.info(
        f"Merged dataset: {len(merged)} rows, "
        f"{len(merged.columns)} columns"
    )
    logger.info(
        f"Columns: {merged.columns.tolist()}"
    )

    return merged


# ══════════════════════════════════════════════════════════
# STEP 5 — ADD LAG FEATURES
# ══════════════════════════════════════════════════════════

def add_lag_features(
    df: pd.DataFrame,
    lag_days: list[int] = LAG_DAYS,
) -> pd.DataFrame:
    """
    Adds lagged versions of key features.

    WHAT is a lag feature?
        A lag feature is the value of a column N days ago.
        sentiment_score_lag1 = yesterday's sentiment score.
        sentiment_score_lag2 = sentiment from 2 days ago.

    WHY do lag features matter?
        Markets do not react to news instantly.
        A negative article published at 6pm is read
        by traders overnight. The market reacts the
        NEXT morning. So yesterday's sentiment is
        often MORE predictive of today's price than
        today's sentiment.

        Research from academic finance shows a 1-3 day
        lag between news sentiment and price movement
        for individual stocks. This is called the
        "post-earnings announcement drift" phenomenon
        and is one of the most studied anomalies in
        financial economics.

    REAL WORLD EXAMPLE:
        If you build a model WITHOUT lag features:
        "Today's sentiment predicts today's return"
        → the model looks into the future (data leakage!)
        → impossibly good training accuracy
        → terrible real-world performance

        With lag features:
        "Yesterday's sentiment predicts today's return"
        → realistic signal, no leakage
        → lower training accuracy but real predictive power

    Args:
        df: merged DataFrame
        lag_days: list of lag periods e.g. [1, 2, 3]

    Returns:
        DataFrame with lag columns added
    """
    df = df.copy()
    df = df.sort_values(
        ["ticker", "date_clean"]
    ).reset_index(drop=True)

    # features to lag — the most predictive signals
    features_to_lag = [
        "daily_avg_sentiment",
        "sentiment_momentum",
        "daily_return",
        "volume_change",
    ]

    logger.info(f"Adding lag features for {lag_days} days...")

    for feature in features_to_lag:
        if feature not in df.columns:
            continue
        for lag in lag_days:
            col_name = f"{feature}_lag{lag}"
            df[col_name] = df.groupby("ticker")[feature].shift(lag)
            logger.info(f"  Added: {col_name}")

    return df


# ══════════════════════════════════════════════════════════
# STEP 6 — ADD TIME FEATURES
# ══════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar-based features.

    WHY time features matter in financial markets:

    1. day_of_week
       Monday returns are historically lower than other days
       (the "Monday effect" — well documented in academic
       finance). Friday afternoons see selling as traders
       avoid weekend risk. Your model should know what day it is.

    2. is_monday / is_friday
       Binary flags for the most anomalous days.

    3. week_of_year
       Captures seasonal patterns — January effect,
       summer slowdown, year-end tax selling.

    4. days_since_last_article
       How long since the last news article?
       Long silence followed by negative news = bigger impact.
       Regular coverage = market already priced in the news.

    Args:
        df: merged DataFrame with date_clean column

    Returns:
        DataFrame with time features added
    """
    df = df.copy()

    # parse dates
    df["_dt"] = pd.to_datetime(df["date_clean"], errors="coerce")

    df["day_of_week"]  = df["_dt"].dt.dayofweek  # 0=Mon, 4=Fri
    df["day_name"]     = df["_dt"].dt.day_name()
    df["week_of_year"] = df["_dt"].dt.isocalendar().week.astype(int)
    df["is_monday"]    = (df["day_of_week"] == 0).astype(int)
    df["is_friday"]    = (df["day_of_week"] == 4).astype(int)

    # days since last article per company
    df = df.sort_values(["ticker", "date_clean"])

    def days_since_article(group):
        """
        For each row, how many days since this company
        had at least one article?
        """
        group = group.copy()
        has_article = group["daily_article_count"] > 0
        last_article_date = None
        days_list = []
        for _, row in group.iterrows():
            if has_article[row.name]:
                last_article_date = pd.to_datetime(row["date_clean"])
                days_list.append(0)
            else:
                if last_article_date is None:
                    days_list.append(np.nan)
                else:
                    current = pd.to_datetime(row["date_clean"])
                    days_list.append((current - last_article_date).days)
        group["days_since_last_article"] = days_list
        return group

    # apply days_since_article per ticker
    # we do NOT use groupby.apply here because pandas
    # drops the groupby key column in some versions
    # instead we loop explicitly and preserve ticker

    result_groups = []
    for ticker_name, group in df.groupby("ticker"):
        group = group.copy()
        processed = days_since_article(group)
        # ensure ticker column survived the function
        if "ticker" not in processed.columns:
            processed["ticker"] = ticker_name
        result_groups.append(processed)

    df = pd.concat(result_groups, ignore_index=True)

    # final guarantee — ticker must exist
    assert "ticker" in df.columns, \
        "ticker column lost in add_time_features"

    # drop helper column
    df = df.drop(columns=["_dt"], errors="ignore")

    logger.info("Time features added")
    return df


# ══════════════════════════════════════════════════════════
# STEP 7 — FINAL CLEAN AND SAVE
# ══════════════════════════════════════════════════════════

def build_feature_matrix(
    sentiment_path: str = SENTIMENT_FILTERED_PATH,
    save_path: str      = FEATURES_PATH,
) -> pd.DataFrame:
    """
    Master function — runs the complete Phase 4 pipeline.

    This is the ONE function your notebook calls.
    It orchestrates all 6 steps above in sequence.

    Pipeline:
    1. Load sentiment data
    2. Determine date range from sentiment data
    3. Fetch stock prices for that date range
    4. Engineer price features + target variable
    5. Aggregate sentiment to daily level
    6. Merge sentiment + prices
    7. Add lag features
    8. Add time features
    9. Clean and save

    Args:
        sentiment_path: path to filtered sentiment CSV
        save_path: where to save the feature matrix

    Returns:
        complete feature matrix DataFrame
    """
    logger.info("=" * 55)
    logger.info("Phase 4 — Building Feature Matrix")
    logger.info("=" * 55)

    # ── load sentiment data ────────────────────────────────
    sentiment_df = pd.read_csv(sentiment_path)
    logger.info(
        f"Loaded sentiment data: {sentiment_df.shape}"
    )

    # ── determine date range ───────────────────────────────
    # fetch prices from 30 days before first article
    # to 5 days after last article
    # Why extend beyond article dates?
    # Price trend BEFORE the article period gives context.
    # Price AFTER gives us the target variable.

    min_date = pd.to_datetime(
        sentiment_df["date_clean"].min()
    ) - timedelta(days=30)
    max_date = pd.to_datetime(
        sentiment_df["date_clean"].max()
    ) + timedelta(days=5)

    start_str = min_date.strftime("%Y-%m-%d")
    end_str   = max_date.strftime("%Y-%m-%d")

    logger.info(f"Fetching prices from {start_str} to {end_str}")

    tickers = sorted(sentiment_df["ticker"].unique().tolist())

    # ── fetch prices ───────────────────────────────────────
    prices_raw = fetch_stock_prices(tickers, start_str, end_str)

    # ── engineer price features ────────────────────────────
    price_features = engineer_price_features(prices_raw)

    # ── aggregate sentiment ────────────────────────────────
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)

    # ── merge ──────────────────────────────────────────────
    merged = merge_sentiment_and_prices(
        daily_sentiment, price_features
    )

    # ── add lag features ───────────────────────────────────
    with_lags = add_lag_features(merged)

    # ── add time features ──────────────────────────────────
    with_time = add_time_features(with_lags)

    # ── final clean ───────────────────────────────────────
    # drop rows where target is NaN
    # (last row per ticker has no "next day" price)
    final = with_time.dropna(
        subset=[TARGET_COLUMN]
    ).reset_index(drop=True)

    # drop rows where close price is NaN
    final = final.dropna(subset=["close"]).reset_index(drop=True)

    # ── save ──────────────────────────────────────────────
    ensure_dirs(os.path.dirname(save_path))
    final.to_csv(save_path, index=False)

    # ── final report ──────────────────────────────────────
    logger.info("\n-- Feature Matrix Summary --")
    logger.info(f"Shape: {final.shape}")
    logger.info(f"Columns ({len(final.columns)}):")
    for col in final.columns:
        logger.info(f"  {col}")
    logger.info(
        f"\nTarget distribution:\n"
        f"{final[TARGET_COLUMN].value_counts().to_string()}"
    )
    logger.info(f"\nSaved to: {save_path}")

    return final