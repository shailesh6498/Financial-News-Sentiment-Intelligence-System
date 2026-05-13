"""
utils.py
────────────────────────────────────────────────────────────
Shared helper functions for the Financial Sentiment System.

Why this file exists:
    Instead of copy-pasting the same date parser in 4 files,
    we write it once here and import it everywhere.
    Fix it here → fixed everywhere. This is the DRY principle.
    (DRY = Don't Repeat Yourself — one of the most important
    principles in professional software engineering)

Used by:
    src/data_collector.py
    src/sentiment.py
    src/features.py
────────────────────────────────────────────────────────────
"""

import os
import logging
from datetime import datetime


# ── LOGGING SETUP ─────────────────────────────────────────
# Why logging instead of print()?
# print() is for exploring. logging is for production.
# logging gives you timestamps, severity levels, and the
# ability to write to a file — so you have a record of
# everything that happened when your system ran overnight.
# Every company uses logging. Zero use print() in production.

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a consistent format.

    Why a function for this?
    So every module gets the same format with one line:
        logger = get_logger(__name__)

    Args:
        name: usually pass __name__ (the module's own name)

    Returns:
        configured Logger object

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Collecting data for AAPL")
        2024-11-01 14:23:01 | data_collector | INFO | Collecting data for AAPL
    """
    logger = logging.getLogger(name)

    # only add handler if logger does not already have one
    # (prevents duplicate log lines if module is imported twice)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # format: timestamp | module name | level | message
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # handler 1: print to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # handler 2: write to a log file
        # so you have a permanent record of every run
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/pipeline.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ── DATE UTILITIES ────────────────────────────────────────
# Why centralise date parsing?
# Yahoo Finance returns dates in a specific format.
# If that format ever changes, you fix it in ONE place.

def parse_yahoo_date(raw_date: str) -> tuple[str, str]:
    """
    Parses a Yahoo Finance ISO date string into
    separate clean date and time strings.

    Why return both date and time separately?
    - date is used for merging with stock price data
    - time is used for intra-day analysis (does morning
      news affect prices differently than evening news?)
    This is a real research question in financial NLP.

    Args:
        raw_date: ISO format string e.g. "2024-11-01T14:23:00Z"

    Returns:
        tuple of (date_str, time_str)
        e.g. ("2024-11-01", "14:23")
        Returns ("unknown", "unknown") if parsing fails.

    Example:
        >>> parse_yahoo_date("2024-11-01T14:23:00Z")
        ('2024-11-01', '14:23')
    """
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return "unknown", "unknown"


def days_ago(date_str: str) -> int:
    """
    Returns how many days ago a date string was.
    Useful for filtering out stale news.

    Args:
        date_str: "YYYY-MM-DD" format string

    Returns:
        int number of days ago, or 9999 if unparseable

    Example:
        >>> days_ago("2024-10-31")  # if today is 2024-11-01
        1
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.now() - date).days
    except (ValueError, TypeError):
        return 9999


# ── DATA VALIDATION ───────────────────────────────────────
# Why validate data?
# Garbage in = garbage out. Your model is only as good
# as the data you feed it. Catching bad data early —
# before it silently corrupts your model — is a
# senior engineering habit that saves hours of debugging.

def validate_article(article: dict) -> tuple[bool, str]:
    """
    Checks whether a parsed article meets minimum
    quality standards for use in the pipeline.

    Why return (bool, reason) instead of just bool?
    So you can log WHY an article was rejected, not just
    that it was. This makes debugging data issues fast.

    Args:
        article: dict with keys ticker, title, summary,
                 date, time, url

    Returns:
        (True, "ok") if valid
        (False, reason_string) if invalid

    Example:
        >>> validate_article({"title": "", ...})
        (False, "empty title")
    """
    if not article.get("title") or article["title"] == "No Title":
        return False, "empty title"

    if not article.get("date") or article["date"] == "unknown":
        return False, "unparseable date"

    if len(article.get("title", "").split()) < 3:
        return False, "title too short (< 3 words)"

    return True, "ok"


# ── FILE UTILITIES ────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """
    Creates directories if they do not exist.
    Accepts multiple paths at once.

    Why a helper for this?
    os.makedirs(..., exist_ok=True) is verbose to type
    repeatedly. This wraps it cleanly.

    Args:
        *paths: any number of directory path strings

    Example:
        >>> ensure_dirs("data/raw", "data/processed", "logs")
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
