"""
data_collector.py
────────────────────────────────────────────────────────────
Handles all data collection from Yahoo Finance.

Used by:
    notebooks/01_data_collection.ipynb
    app/dashboard.py  (Phase 6)
────────────────────────────────────────────────────────────
"""

import yfinance as yf
import pandas as pd
import os
import sys

# add project root to path so we can import config and utils
# this is needed when running from the notebooks/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPANIES, RAW_DATA_PATH
from src.utils import get_logger, parse_yahoo_date, validate_article, ensure_dirs

# get a logger for this specific module
logger = get_logger(__name__)


def fetch_raw_news(ticker_symbol: str) -> list[dict]:
    """
    Fetches raw news articles from Yahoo Finance for one company.

    Why isolate this into its own function?
    Single Responsibility Principle — one function does one thing.
    If Yahoo changes their API, you fix only this function.
    Everything else in the codebase stays the same.

    Args:
        ticker_symbol: stock ticker string e.g. "AAPL"

    Returns:
        list of raw article dicts from Yahoo Finance API
        Returns empty list if fetch fails (never crashes the pipeline)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        raw_news = ticker.news
        logger.info(f"Fetched {len(raw_news)} raw articles for {ticker_symbol}")
        return raw_news

    except Exception as e:
        # log the error but do NOT crash
        # one company failing should never stop the whole pipeline
        logger.error(f"Failed to fetch {ticker_symbol}: {e}")
        return []


def parse_article(raw_article: dict, ticker_symbol: str) -> dict:
    """
    Parses one raw Yahoo Finance article dict into a
    clean, standardised format.

    Why a separate parse function?
    Because Yahoo Finance has changed their response format
    before (as you discovered in Colab). If it changes again,
    you update only this function. The rest of the pipeline
    never knows the difference.

    Args:
        raw_article: raw dict from Yahoo Finance API
        ticker_symbol: which company this article is about

    Returns:
        clean dict with standardised keys
    """
    # Yahoo Finance nests content inside a 'content' key
    content   = raw_article.get("content", {})

    # extract title and summary
    title     = content.get("title",   "No Title")
    summary   = content.get("summary", "No Summary")

    # extract URL — nested inside canonicalUrl
    canonical = content.get("canonicalUrl", {})
    url       = canonical.get("url", "")

    # parse date using our centralised utility
    raw_date        = content.get("pubDate", "")
    date_str, time_str = parse_yahoo_date(raw_date)

    return {
        "ticker"  : ticker_symbol,
        "title"   : title,
        "summary" : summary,
        "date"    : date_str,
        "time"    : time_str,
        "url"     : url,
    }


def collect_company_news(ticker_symbol: str) -> tuple[list[dict], dict]:
    """
    Full collection pipeline for one company:
    fetch → parse → validate → return clean articles.

    Also returns a stats dict — this is how you track
    data quality over time. In production, these stats
    would go into a monitoring dashboard.

    Args:
        ticker_symbol: stock ticker e.g. "AAPL"

    Returns:
        tuple of (clean_articles_list, stats_dict)

    Example:
        >>> articles, stats = collect_company_news("AAPL")
        >>> print(stats)
        {'ticker': 'AAPL', 'fetched': 10, 'valid': 9, 'rejected': 1}
    """
    stats = {
        "ticker"  : ticker_symbol,
        "fetched" : 0,
        "valid"   : 0,
        "rejected": 0,
    }

    # step 1 — fetch raw
    raw_articles = fetch_raw_news(ticker_symbol)
    stats["fetched"] = len(raw_articles)

    # step 2 — parse and validate each article
    clean_articles = []

    for raw in raw_articles:
        # parse into clean format
        article = parse_article(raw, ticker_symbol)

        # validate quality
        is_valid, reason = validate_article(article)

        if is_valid:
            clean_articles.append(article)
            stats["valid"] += 1
        else:
            stats["rejected"] += 1
            logger.warning(
                f"Rejected article [{ticker_symbol}]: "
                f'"{article.get("title", "")[:50]}" — reason: {reason}'
            )

    return clean_articles, stats


def collect_all_companies(
    companies: list[str] = COMPANIES,
    save_path: str = RAW_DATA_PATH
) -> pd.DataFrame:
    """
    Master collection function — collects news for ALL companies,
    combines into one DataFrame, saves to CSV.

    This is what the notebook calls. One function call
    runs the entire data collection pipeline.

    Why default arguments (companies=COMPANIES)?
    It uses config.py defaults but lets you override for testing.
    For example: collect_all_companies(companies=["AAPL"])
    This is called "dependency injection" and makes code testable.

    Args:
        companies: list of ticker symbols to collect
        save_path: where to save the CSV

    Returns:
        DataFrame with all collected articles
    """
    logger.info("=" * 55)
    logger.info("Starting data collection pipeline")
    logger.info(f"Companies: {companies}")
    logger.info("=" * 55)

    # ensure output directory exists
    ensure_dirs(os.path.dirname(save_path))

    all_articles = []
    all_stats    = []

    # collect each company independently
    for symbol in companies:
        articles, stats = collect_company_news(symbol)
        all_articles.extend(articles)
        all_stats.append(stats)

    # build the main DataFrame
    df = pd.DataFrame(all_articles)

    # ── data quality report ───────────────────────────────
    # print this every run so you notice if data quality drops
    stats_df = pd.DataFrame(all_stats)

    logger.info("\n── Collection Summary ──────────────────")
    logger.info(f"\n{stats_df.to_string(index=False)}")
    logger.info(f"\nTotal valid articles : {stats_df['valid'].sum()}")
    logger.info(f"Total rejected       : {stats_df['rejected'].sum()}")
    rejection_rate = (
        stats_df["rejected"].sum() /
        max(stats_df["fetched"].sum(), 1) * 100
    )
    logger.info(f"Rejection rate       : {rejection_rate:.1f}%")
    logger.info("────────────────────────────────────────")

    # ── save ──────────────────────────────────────────────
    df.to_csv(save_path, index=False)
    logger.info(f"Saved {len(df)} articles to {save_path}")

    return df