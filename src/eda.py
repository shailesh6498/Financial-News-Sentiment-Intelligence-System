"""
eda.py
────────────────────────────────────────────────────────────
All EDA and visualisation functions for the Financial
Sentiment Intelligence System.

Designed specifically for this dataset structure:
    ticker     : str  — company ticker e.g. "AAPL"
    title      : str  — news headline
    summary    : str  — article summary
    date       : str  — "01 November 2024, 02:30 PM UTC"
    url        : str  — article URL
    date_clean : str  — "2024-11-01" (YYYY-MM-DD)

Design principle:
    Every function takes a DataFrame and returns either
    a Figure or a summary DataFrame — never crashes on
    missing optional columns, always logs what it finds.

Real-world context:
    At Meta, EDA functions like these live in shared
    internal libraries so every team uses the same
    analysis standards. You are building your own version.

Used by:
    notebooks/02_eda.ipynb
    notebooks/05_modelling.ipynb  (evaluation charts)
    app/dashboard.py              (Phase 6)
────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

from src.utils import get_logger, ensure_dirs
from config import REPORTS_PATH

logger = get_logger(__name__)

# ── global chart style ────────────────────────────────────
# set once here — affects every chart in this module
# consistent visual identity across all figures
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#f8f9fa",
    "axes.grid"         : True,
    "grid.color"        : "white",
    "grid.linewidth"    : 1.2,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : False,
    "axes.spines.bottom": False,
    "font.family"       : "DejaVu Sans",
    "axes.titleweight"  : "bold",
    "axes.titlepad"     : 14,
    "axes.labelpad"     : 8,
    "figure.titlesize"  : 14,
    "figure.titleweight": "bold",
})

COLORS = {
    "AAPL" : "#5B8DB8",
    "GOOGL": "#E8834D",
    "META" : "#6BAF7A",
    "AMZN" : "#C9637A",
    "MSFT" : "#9B7EC8",
}
PALETTE = list(COLORS.values())


# ── shared date parser ────────────────────────────────────
# this is the single place that knows your date format
# "01 November 2024, 02:30 PM UTC"
# every function that needs a parsed datetime calls this
# Why: if Yahoo ever changes the format, fix it here once

def _parse_dates(df: pd.DataFrame) -> pd.Series:
    """
    Parses your Phase 1 date column into proper datetimes.

    Tries date_clean first (fastest, most reliable) then
    falls back to parsing the full date string.

    Why date_clean first?
    "2024-11-01" is unambiguous and fast to parse.
    The full date string needs dayfirst=True and is slower.

    Args:
        df: DataFrame with date and/or date_clean columns

    Returns:
        Series of datetime64 values (NaT where parsing fails)
    """
    # try date_clean first — it is YYYY-MM-DD, clean and fast
    if "date_clean" in df.columns:
        parsed = pd.to_datetime(df["date_clean"], errors="coerce")
        if parsed.notna().sum() > 0:
            logger.info(
                f"Dates parsed from date_clean "
                f"({parsed.notna().sum()}/{len(df)} valid)"
            )
            return parsed

    # fallback: parse the full date string
    # format: "01 November 2024, 02:30 PM UTC"
    if "date" in df.columns:
        parsed = pd.to_datetime(
            df["date"], dayfirst=True, errors="coerce"
        )
        logger.info(
            f"Dates parsed from date column "
            f"({parsed.notna().sum()}/{len(df)} valid)"
        )
        return parsed

    # nothing worked — return empty series
    logger.warning("Could not parse any dates from this DataFrame")
    return pd.Series(pd.NaT, index=df.index)


def _extract_hours(df: pd.DataFrame) -> pd.Series:
    """
    Extracts publication hour from your date column.

    Your date column "01 November 2024, 02:30 PM UTC"
    contains time — we parse that to get the hour.
    This tells us whether news was published before or
    after market open (9:30am ET = 14:30 UTC).

    Why does publication time matter for your model?
    A headline at 8am ET hits BEFORE the market opens.
    Traders read it and act at 9:30am. That article's
    sentiment should predict the SAME day's price.
    A headline at 6pm ET hits AFTER market close.
    It influences the NEXT day's open. Your feature
    engineering in Phase 4 will use this distinction.

    Args:
        df: DataFrame with date column

    Returns:
        Series of integer hours (0-23), NaN where unavailable
    """
    if "date" in df.columns:
        parsed = pd.to_datetime(
            df["date"], dayfirst=True, errors="coerce"
        )
        if parsed.notna().sum() > 0:
            return parsed.dt.hour

    logger.warning("Could not extract hours — time not in data")
    return pd.Series(np.nan, index=df.index)


def _save_figure(fig: plt.Figure, filename: str) -> str:
    """
    Saves a figure to reports/figures/ and returns the path.

    Args:
        fig: matplotlib Figure object
        filename: just the filename e.g. "01_overview.png"

    Returns:
        full path where figure was saved
    """
    ensure_dirs(REPORTS_PATH)
    path = os.path.join(REPORTS_PATH, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Chart saved → {path}")
    return path


# ══════════════════════════════════════════════════════════
# LAYER 1 — STRUCTURAL ANALYSIS
# ══════════════════════════════════════════════════════════

def structural_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-level summary: type, nulls, uniqueness, samples.

    Why this before anything else?
    You cannot analyse what you do not understand.
    This function answers: what do I have, is it complete,
    and does it look right? In 30 seconds.

    Interview answer:
        "First thing I do with any new dataset is a
        structural summary — types, null rates, cardinality,
        and a sample value per column. It surfaces quality
        issues before they silently corrupt analysis."

    Args:
        df: any DataFrame

    Returns:
        summary DataFrame — one row per column
    """
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append({
            "column"        : col,
            "dtype"         : str(series.dtype),
            "null_count"    : int(series.isnull().sum()),
            "null_pct"      : f"{series.isnull().mean()*100:.1f}%",
            "unique_values" : int(series.nunique()),
            "sample_value"  : (
                str(series.dropna().iloc[0])[:60]
                if len(series.dropna()) > 0
                else "ALL NULL"
            ),
        })

    summary = pd.DataFrame(rows)
    logger.info(
        f"Structural summary complete: "
        f"{len(df)} rows × {len(df.columns)} cols"
    )
    return summary


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Checks for duplicate articles at two levels:
    exact title match and near-duplicate (first 60 chars).

    Why near-duplicates matter:
        Wire services like Reuters and AP publish the same
        story multiple times with minor title edits.
        "Apple Q4 earnings beat estimates" and
        "Apple Q4 earnings beat analyst estimates" are
        effectively the same sentiment signal. Counting
        them twice would bias your model.

    Interview answer:
        "I check exact duplicates by full title match, and
        near-duplicates by comparing title prefixes — a
        fast heuristic that catches wire-service repubs
        without expensive string similarity computation."

    Args:
        df: DataFrame with a title column

    Returns:
        dict with counts and examples
    """
    # exact duplicates
    exact_mask  = df.duplicated(subset=["title"], keep=False)
    exact_dupes = df[exact_mask]

    # near-duplicates — same first 60 characters lowercased
    df_temp               = df.copy()
    df_temp["_prefix"]    = df_temp["title"].str[:60].str.lower()
    near_mask             = df_temp.duplicated(subset=["_prefix"], keep=False)
    near_dupes            = df_temp[near_mask]

    result = {
        "exact_duplicate_count": int(len(exact_dupes)),
        "near_duplicate_count" : int(len(near_dupes)),
        "exact_examples"       : exact_dupes["title"].head(3).tolist(),
    }

    logger.info(
        f"Duplicates — exact: {result['exact_duplicate_count']}, "
        f"near: {result['near_duplicate_count']}"
    )
    return result


# ══════════════════════════════════════════════════════════
# LAYER 2 — UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════

def plot_article_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Three-panel chart:
    1. Articles per company (bar)
    2. News volume over time (line)
    3. Publication hour distribution (histogram)

    All three built using your actual columns:
    ticker, date_clean, date (for hour extraction)

    Why publication hour matters:
        Pre-market news (before 14:30 UTC / 9:30am ET)
        influences same-day prices.
        Post-market news influences next-day open.
        This timing distinction becomes a feature in Phase 4.

    Args:
        df: DataFrame with ticker, date_clean, date columns

    Returns:
        matplotlib Figure saved to reports/figures/
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Article Distribution Analysis")

    # ── panel 1: articles per company ──────────────────
    counts     = df.groupby("ticker").size().sort_values(ascending=False)
    bar_colors = [COLORS.get(t, "#888") for t in counts.index]

    bars = axes[0].bar(
        counts.index, counts.values,
        color=bar_colors, width=0.6, zorder=3
    )
    axes[0].set_title("Articles per Company")
    axes[0].set_xlabel("Company")
    axes[0].set_ylabel("Article Count")

    # value labels on top of each bar
    # why: never make the reader estimate bar heights
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center", va="bottom",
            fontweight="bold", fontsize=11
        )

    # ── panel 2: articles per date ──────────────────────
    # use date_clean (YYYY-MM-DD) for clean x-axis grouping
    daily = df.groupby("date_clean").size().sort_index()

    axes[1].plot(
        range(len(daily)), daily.values,
        marker="o", linewidth=2,
        color=PALETTE[0], markerfacecolor="white",
        markeredgewidth=2, markersize=7, zorder=3
    )
    axes[1].fill_between(
        range(len(daily)), daily.values,
        alpha=0.15, color=PALETTE[0]
    )
    axes[1].set_title("News Volume Over Time")
    axes[1].set_xlabel("Date (chronological)")
    axes[1].set_ylabel("Articles Published")

    # show only first, middle, last date — avoids x-axis clutter
    n              = len(daily)
    tick_pos       = [0, n // 2, n - 1] if n > 2 else list(range(n))
    tick_labels    = [daily.index[i] for i in tick_pos]
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")

    # ── panel 3: publication hour ───────────────────────
    # extract hour from your "01 November 2024, 02:30 PM UTC" column
    hours = _extract_hours(df).dropna()

    if len(hours) > 0:
        axes[2].hist(
            hours, bins=24, range=(0, 24),
            color=PALETTE[2], alpha=0.85,
            edgecolor="white", zorder=3
        )
        # 14:30 UTC = 9:30am ET = NYSE market open
        axes[2].axvline(
            14.5, color="red", linestyle="--",
            alpha=0.8, linewidth=1.5,
            label="NYSE open (14:30 UTC)"
        )
        # 21:00 UTC = 4:00pm ET = NYSE market close
        axes[2].axvline(
            21, color="orange", linestyle="--",
            alpha=0.8, linewidth=1.5,
            label="NYSE close (21:00 UTC)"
        )
        axes[2].legend(fontsize=8)
        axes[2].set_title("Publication Hour (UTC)")
        axes[2].set_xlabel("Hour of Day (0-23)")
        axes[2].set_ylabel("Articles")
    else:
        # graceful fallback — never crash the analysis
        axes[2].text(
            0.5, 0.5,
            "Hour data\nnot available",
            ha="center", va="center",
            transform=axes[2].transAxes,
            fontsize=12, color="gray"
        )
        axes[2].set_title("Publication Hour (UTC)")

    plt.tight_layout()
    _save_figure(fig, "01_article_distribution.png")
    return fig


# ══════════════════════════════════════════════════════════
# LAYER 3 — BIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════

def plot_coverage_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: companies (rows) × dates (columns) = article count.

    Why this chart matters for your model:
        If AAPL has zero articles on a specific date,
        your sentiment signal is missing for that day.
        That is not a zero — it is a gap. Your feature
        engineering in Phase 4 must handle gaps explicitly.
        This chart shows you where those gaps are.

    Interview answer:
        "I cross-tabulate company vs date to find coverage
        gaps. A missing day is not zero sentiment — it is
        unknown sentiment. Treating them the same would
        introduce subtle bias into the model."

    Args:
        df: DataFrame with ticker and date_clean columns

    Returns:
        matplotlib Figure
    """
    # pivot table: rows=ticker, cols=date, values=article count
    pivot = (
        df.groupby(["ticker", "date_clean"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), 4))
    fig.suptitle("Coverage Heatmap — Articles per Company per Day")

    im = ax.imshow(
        pivot.values, aspect="auto",
        cmap="YlOrRd", vmin=0
    )

    # y axis — company names
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # x axis — show every Nth date to avoid overlap
    n_dates = len(pivot.columns)
    step    = max(1, n_dates // 8)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels(
        [pivot.columns[i] for i in range(0, n_dates, step)],
        rotation=45, ha="right", fontsize=9
    )

    # count annotations inside cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 0:
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > 3 else "black"
                )

    plt.colorbar(im, ax=ax, label="Article Count", shrink=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Company")

    plt.tight_layout()
    _save_figure(fig, "02_coverage_heatmap.png")
    return fig


# ══════════════════════════════════════════════════════════
# LAYER 4 — TEXT ANALYSIS
# ══════════════════════════════════════════════════════════

def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Word count statistics for titles and summaries per company.

    Why measure this before FinBERT?
    FinBERT has a 512 token maximum input length.
    If your summaries average 300 words you need a
    truncation strategy. If headlines average 6 words
    you know short-text NLP considerations apply.

    Also flags very short titles (under 5 words) which
    are often low quality — tickers, error messages, ads.

    Interview answer:
        "Before running any NLP model I check token length
        distributions to understand truncation risk and
        flag abnormally short texts that may be noise
        rather than real financial news."

    Args:
        df: DataFrame with title and summary columns

    Returns:
        per-company text stats DataFrame
    """
    df = df.copy()
    df["title_words"]   = df["title"].fillna("").str.split().str.len()
    df["summary_words"] = df["summary"].fillna("").str.split().str.len()

    stats = df.groupby("ticker").agg(
        title_avg_words   =("title_words",   "mean"),
        title_max_words   =("title_words",   "max"),
        title_min_words   =("title_words",   "min"),
        summary_avg_words =("summary_words", "mean"),
        summary_max_words =("summary_words", "max"),
        short_title_count =("title_words",   lambda x: (x < 5).sum()),
    ).round(1)

    return stats


def get_top_words(
    df: pd.DataFrame,
    n: int = 30,
    per_company: bool = False
) -> dict:
    """
    Most frequent meaningful words in headlines.

    Why before FinBERT?
    Word frequency is your fastest qualitative data check.
    If top words are financial terms like "earnings",
    "revenue", "growth" — your data is real financial news.
    If top words are "click", "sponsored", "ad" — you have
    a data quality problem that no model will fix.

    Args:
        df: DataFrame with title and ticker columns
        n: number of top words to return
        per_company: return words per company if True

    Returns:
        dict — {"overall": [(word, count), ...]}
               or {"AAPL": [...], ...} if per_company=True
    """
    # financial news specific stopwords
    # generic stopwords + numbers + common financial boilerplate
    STOPWORDS = {
        "the","a","an","in","of","to","and","for","is","on",
        "at","by","its","with","as","be","that","this","are",
        "was","has","have","it","from","says","said","will",
        "after","over","new","more","than","up","but","not",
        "what","all","been","also","into","about","could",
        "their","he","she","they","we","you","your","our",
        "report","reports","quarter","quarterly","year",
        "billion","million","percent","%",
        "2023","2024","2025","2026",
        "company","companies","inc","corp","ltd",
    }

    def _extract(series: pd.Series) -> list[str]:
        text  = " ".join(series.fillna("").str.lower())
        words = [
            w.strip(".,!?;:'\"()[]{}—-")
            for w in text.split()
        ]
        return [
            w for w in words
            if w not in STOPWORDS and len(w) > 2
        ]

    if per_company:
        result = {}
        for ticker in sorted(df["ticker"].unique()):
            words          = _extract(df[df["ticker"] == ticker]["title"])
            result[ticker] = Counter(words).most_common(n)
        return result
    else:
        words = _extract(df["title"])
        return {"overall": Counter(words).most_common(n)}


def plot_word_frequency(
    df: pd.DataFrame,
    top_n: int = 20
) -> plt.Figure:
    """
    Side-by-side word frequency: overall + per company.

    The per-company panels are the key insight here.
    Different companies generate different vocabulary —
    AMZN has "AWS", "Prime", "warehouse"; META has
    "privacy", "regulation", "ads". This tells you that
    a single global sentiment model may miss company-
    specific language patterns. Your FinBERT model in
    Phase 3 handles this because it was trained on
    diverse financial text.

    Args:
        df: DataFrame with title and ticker columns
        top_n: how many words to show in main panel

    Returns:
        matplotlib Figure
    """
    word_data   = get_top_words(df, n=top_n)["overall"]
    per_co_data = get_top_words(df, n=8, per_company=True)
    companies   = list(per_co_data.keys())
    n_co        = len(companies)

    # layout: wide left panel + grid of company panels
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Headline Language Analysis")

    gs = gridspec.GridSpec(
        2, n_co + 1,
        figure=fig,
        width_ratios=[2.5] + [1] * n_co,
        hspace=0.5, wspace=0.4
    )

    # ── left: overall top words ─────────────────────────
    ax_main = fig.add_subplot(gs[:, 0])
    words   = [w for w, _ in word_data]
    counts  = [c for _, c in word_data]

    bars = ax_main.barh(
        range(len(words)), counts,
        color=PALETTE[0], alpha=0.85, zorder=3
    )
    ax_main.set_yticks(range(len(words)))
    ax_main.set_yticklabels(words, fontsize=10)
    ax_main.invert_yaxis()
    ax_main.set_title(f"Top {top_n} Words\n(all companies)")
    ax_main.set_xlabel("Frequency")

    for bar, count in zip(bars, counts):
        ax_main.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontsize=8
        )

    # ── right: per-company panels ───────────────────────
    for idx, ticker in enumerate(companies):
        row         = idx % 2
        col         = idx // 2 + 1
        ax          = fig.add_subplot(gs[row, col])
        word_counts = per_co_data[ticker]
        co_words    = [w for w, _ in word_counts]
        co_counts   = [c for _, c in word_counts]

        ax.barh(
            range(len(co_words)), co_counts,
            color=COLORS.get(ticker, "#888"),
            alpha=0.85, zorder=3
        )
        ax.set_yticks(range(len(co_words)))
        ax.set_yticklabels(co_words, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(
            ticker, fontweight="bold",
            color=COLORS.get(ticker, "#888")
        )
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    _save_figure(fig, "03_word_frequency.png")
    return fig


# ══════════════════════════════════════════════════════════
# LAYER 5 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════

def plot_temporal_patterns(df: pd.DataFrame) -> plt.Figure:
    """
    Two-panel temporal analysis:
    1. Daily article count per company (line chart)
    2. Articles by day of week (bar chart)

    Why day-of-week matters for your model:
        Earnings releases happen on specific days.
        Weekend news has no immediate market reaction
        because markets are closed. Tuesday-Thursday
        is peak earnings season activity.
        Day of week will be a feature in Phase 4.

    Interview answer:
        "Temporal EDA told me that most articles are
        published mid-week, consistent with US earnings
        call scheduling. This led to adding day-of-week
        as a feature and handling weekend news separately
        — it influences Monday open, not Friday close."

    Args:
        df: DataFrame with ticker, date, date_clean columns

    Returns:
        matplotlib Figure
    """
    # parse dates using our shared utility
    # works with your actual date column format
    df_temp               = df.copy()
    df_temp["_dt"]        = _parse_dates(df_temp)
    df_temp["_day_name"]  = df_temp["_dt"].dt.day_name()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temporal Patterns in News Coverage")

    # ── panel 1: per-company volume over time ───────────
    for ticker, color in COLORS.items():
        subset = df_temp[df_temp["ticker"] == ticker]
        if len(subset) == 0:
            continue

        daily = subset.groupby("date_clean").size().sort_index()
        axes[0].plot(
            range(len(daily)), daily.values,
            label=ticker, color=color,
            linewidth=2, marker="o",
            markersize=5, alpha=0.85
        )

    axes[0].set_title("Daily Article Count per Company")
    axes[0].set_xlabel("Date (chronological)")
    axes[0].set_ylabel("Articles")
    axes[0].legend(fontsize=9)

    # ── panel 2: articles by day of week ────────────────
    day_order  = ["Monday","Tuesday","Wednesday",
                  "Thursday","Friday","Saturday","Sunday"]
    day_counts = (
        df_temp.groupby("_day_name")
        .size()
        .reindex(day_order, fill_value=0)
    )

    bar_colors = (
        [PALETTE[3]] * 5    # weekdays
        + ["#cccccc"] * 2   # weekend — grey to signal low trading
    )

    axes[1].bar(
        range(7), day_counts.values,
        color=bar_colors, alpha=0.85,
        edgecolor="white", zorder=3
    )
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(
        [d[:3] for d in day_order], fontsize=10
    )
    axes[1].set_title("Articles by Day of Week")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Total Articles")

    # shade weekend columns to signal no market activity
    axes[1].axvspan(
        4.5, 6.5, alpha=0.08,
        color="gray", label="Weekend\n(no market)"
    )
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save_figure(fig, "04_temporal_patterns.png")
    return fig


# ══════════════════════════════════════════════════════════
# LAYER 6 — CLEAN AND SAVE
# ══════════════════════════════════════════════════════════

def clean_and_save(
    df: pd.DataFrame,
    save_path: str
) -> pd.DataFrame:
    """
    Applies cleaning steps justified by EDA findings,
    saves the processed dataset ready for Phase 3.

    Cleaning decisions and why:
    1. Remove exact duplicate titles
       → duplicates inflate sentiment signal for one event
    2. Remove rows where date_clean is null or unknown
       → cannot merge with stock prices without a valid date
    3. Strip whitespace from text columns
       → consistent input for FinBERT tokeniser
    4. Add title_word_count
       → Phase 3 uses this to flag very short headlines
    5. Add date_parsed (proper datetime)
       → Phase 4 needs this for time-based feature joins
    6. Sort by ticker + date_clean
       → clean ordering for sequential model input

    Why clean AFTER EDA not before?
    EDA tells you what cleaning is needed and why.
    Cleaning before EDA is guessing. You never guess
    at a production data pipeline.

    Interview answer:
        "I always EDA before cleaning. EDA is diagnosis,
        cleaning is treatment. You don't prescribe before
        you diagnose."

    Args:
        df: raw DataFrame from Phase 1
        save_path: full path to save cleaned CSV

    Returns:
        cleaned DataFrame
    """
    original_len = len(df)
    df           = df.copy()

    logger.info("Starting data cleaning...")

    # ── step 1: remove exact duplicate titles ─────────
    before = len(df)
    df     = df.drop_duplicates(subset=["title"], keep="first")
    removed = before - len(df)
    logger.info(f"  Removed {removed} duplicate titles")

    # ── step 2: remove rows with invalid dates ─────────
    # date_clean must be a valid YYYY-MM-DD string
    before = len(df)
    df     = df[
        df["date_clean"].notna()
        & (df["date_clean"] != "unknown")
        & (df["date_clean"] != "")
    ]
    logger.info(f"  Removed {before - len(df)} rows with invalid dates")

    # ── step 3: strip whitespace from text columns ─────
    for col in ["title", "summary", "url"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    logger.info("  Stripped whitespace from text columns")

    # ── step 4: add title_word_count ───────────────────
    # Phase 3 uses this to skip very short or empty titles
    df["title_word_count"] = df["title"].str.split().str.len()
    logger.info("  Added title_word_count column")

    # ── step 5: add date_parsed ────────────────────────
    # proper datetime object for Phase 4 time-based joins
    df["date_parsed"] = _parse_dates(df)
    logger.info("  Added date_parsed column")

    # ── step 6: sort by ticker + date ─────────────────
    df = df.sort_values(
        ["ticker", "date_clean"]
    ).reset_index(drop=True)
    logger.info("  Sorted by ticker and date")

    # ── save ───────────────────────────────────────────
    ensure_dirs(os.path.dirname(save_path))
    df.to_csv(save_path, index=False)

    logger.info(
        f"Cleaning complete: "
        f"{original_len} → {len(df)} rows "
        f"({original_len - len(df)} removed) "
        f"saved to {save_path}"
    )

    return df