"""
dashboard.py
────────────────────────────────────────────────────────────
Live Streamlit dashboard for the Financial Sentiment
Intelligence System.

HOW TO RUN LOCALLY:
    streamlit run app/dashboard.py

HOW IT WORKS:
    1. User types a company ticker (e.g. AAPL)
    2. App fetches latest news from Yahoo Finance
    3. FinBERT scores each headline
    4. XGBoost predicts price direction
    5. Charts and scores displayed in real time

WHY STREAMLIT:
    Turns Python into a website with zero HTML/CSS/JS.
    Used at Google DeepMind, Airbnb, Uber for data apps.
    Deploys to Streamlit Cloud free — live URL instantly.

ARCHITECTURE:
    This file is the ONLY file that changes for the UI.
    All business logic lives in src/ modules.
    The dashboard just calls those functions and
    displays the results. Clean separation of concerns.
────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
from datetime import datetime, timedelta

# ── path setup ────────────────────────────────────────────
# add project root so we can import src/ and config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import (
    COMPANIES,
    COLORS,
    MODEL_METRICS_PATH,
    XGBOOST_MODEL_PATH,
    SENTIMENT_FILTERED_PATH,
    FEATURES_PATH,
    BASE_DIR,
)

# ── page configuration ────────────────────────────────────
# this must be the FIRST streamlit command in the file
st.set_page_config(
    page_title    = "SentimentIQ — Financial Intelligence",
    page_icon     = "📈",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── custom CSS ────────────────────────────────────────────
# why: streamlit's default styling is functional but plain
# a polished UI makes your product look trustworthy
# investors and users judge products by how they look
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #0e1117; }

    /* metric cards */
    div[data-testid="metric-container"] {
        background-color: #1c2130;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
    }

    /* positive sentiment */
    .sentiment-positive {
        color: #48bb78;
        font-weight: bold;
        font-size: 2rem;
    }

    /* negative sentiment */
    .sentiment-negative {
        color: #fc8181;
        font-weight: bold;
        font-size: 2rem;
    }

    /* neutral sentiment */
    .sentiment-neutral {
        color: #a0aec0;
        font-weight: bold;
        font-size: 2rem;
    }

    /* prediction badge */
    .prediction-up {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }

    .prediction-down {
        background: linear-gradient(135deg, #fc8181, #e53e3e);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* disclaimer */
    .disclaimer {
        color: #718096;
        font-size: 0.75rem;
        border-top: 1px solid #2d3748;
        padding-top: 8px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── cached functions ──────────────────────────────────────
# @st.cache_resource: load once, reuse across all users
# without caching, every user visit would reload FinBERT
# from disk — 15 seconds each time. Cached = instant.
# This is how production ML serving works — load once,
# serve many.

@st.cache_resource(show_spinner="Loading AI model...")
def load_finbert_cached():
    """
    Loads FinBERT model once and caches it in memory.
    All subsequent calls return the cached version instantly.
    WHY: FinBERT is 500MB. Loading it for every user request
    would make the app unusably slow. Cache it once at startup.
    """
    try:
        from src.sentiment import load_finbert
        return load_finbert()
    except Exception as e:
        return None, None


@st.cache_resource(show_spinner="Loading prediction model...")
def load_xgboost_cached():
    """
    Loads saved XGBoost model and feature names.
    WHY: Same reason as FinBERT — load once, serve many.
    """
    try:
        from src.model import load_model
        return load_model()
    except Exception as e:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_news(ticker: str) -> pd.DataFrame:
    """
    Fetches latest news for a company.

    ttl=3600 means cache for 1 hour.
    WHY: Yahoo Finance API has rate limits. If 100 users
    all search for AAPL in the same minute, we only call
    the API once and serve cached results to the other 99.
    After 1 hour, the cache refreshes automatically.

    This is called a read-through cache — the same pattern
    used by every major website including Google and Twitter.
    """
    import yfinance as yf
    from datetime import datetime

    try:
        ticker_obj = yf.Ticker(ticker)
        raw_news   = ticker_obj.news

        articles = []
        for article in raw_news:
            content   = article.get("content", {})
            title     = content.get("title", "")
            summary   = content.get("summary", "")
            canonical = content.get("canonicalUrl", {})
            url       = canonical.get("url", "")
            raw_date  = content.get("pubDate", "")

            try:
                dt         = datetime.strptime(
                    raw_date, "%Y-%m-%dT%H:%M:%SZ"
                )
                date_str   = dt.strftime("%b %d, %Y %I:%M %p UTC")
                date_clean = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str   = "Unknown date"
                date_clean = "unknown"

            if title:
                articles.append({
                    "title"     : title,
                    "summary"   : summary[:200] + "..." if len(summary) > 200 else summary,
                    "date"      : date_str,
                    "date_clean": date_clean,
                    "url"       : url,
                    "ticker"    : ticker,
                })

        return pd.DataFrame(articles)

    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_price_cached(ticker: str) -> pd.DataFrame:
    """
    Fetches 30 days of stock prices.
    Cached for 1 hour for same reasons as news.
    """
    import yfinance as yf
    try:
        raw = yf.download(
            ticker,
            period="30d",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()

        # flatten MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.reset_index()
        raw["date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
        return raw[["date", "close", "volume"]].dropna()

    except Exception as e:
        return pd.DataFrame()


def score_headlines(
    news_df: pd.DataFrame,
    tokenizer,
    model,
) -> pd.DataFrame:
    """
    Scores headlines with FinBERT and returns results.
    WHY we score live: this gives users real-time sentiment
    on today's news, not yesterday's pre-computed scores.
    Real-time is what makes this product genuinely useful.
    """
    if news_df.empty or tokenizer is None:
        return news_df

    from src.sentiment import score_headline

    scores = []
    for title in news_df["title"]:
        result = score_headline(title, tokenizer, model)
        scores.append(result)

    news_df = news_df.copy()
    news_df["sentiment_label"] = [s["sentiment_label"] for s in scores]
    news_df["sentiment_score"] = [s["sentiment_score"] for s in scores]
    news_df["confidence"]      = [s["sentiment_confidence"] for s in scores]

    return news_df


def make_prediction(
    ticker     : str,
    news_df    : pd.DataFrame,
    price_df   : pd.DataFrame,
    xgb_model,
    feat_names : list,
) -> dict:
    """
    Generates a price direction prediction for a company.

    HOW it works:
    1. Compute today's average sentiment from live news
    2. Get latest price features from price history
    3. Build a feature row matching training format
    4. Feed to XGBoost and get probability

    WHY this approach:
    We cannot run the full Phase 4 pipeline live for
    every user request — it would take too long. Instead
    we build a simplified feature row from live data
    that approximates what the training features look like.
    """
    if xgb_model is None or feat_names is None:
        return {
            "direction"  : "UNKNOWN",
            "confidence" : 0,
            "prob_up"    : 50,
            "prob_down"  : 50,
            "available"  : False,
        }

    try:
        # compute sentiment features from live news
        avg_sentiment = 0.0
        article_count = 0
        positive_ratio = 0.0

        if not news_df.empty and "sentiment_score" in news_df.columns:
            avg_sentiment  = float(news_df["sentiment_score"].mean())
            article_count  = len(news_df)
            pos_count      = (news_df["sentiment_label"] == "positive").sum()
            positive_ratio = pos_count / max(article_count, 1)

        # compute price features from recent history
        daily_return     = 0.0
        price_range      = 0.0
        volume_change    = 0.0
        price_vs_5day    = 0.0

        if not price_df.empty and len(price_df) >= 2:
            closes = price_df["close"].values
            daily_return  = float(
                (closes[-1] - closes[-2]) / closes[-2] * 100
            )
            price_vs_5day = float(
                (closes[-1] / closes[-5:].mean() - 1) * 100
            ) if len(closes) >= 5 else 0.0

        # build feature row — all values default to 0
        # features that cannot be computed get 0 (neutral)
        feature_row = {feat: 0.0 for feat in feat_names}

        # fill in what we can compute
        live_features = {
            "daily_avg_sentiment"      : avg_sentiment,
            "daily_article_count"      : float(article_count),
            "positive_ratio"           : positive_ratio,
            "daily_return"             : daily_return,
            "price_vs_5day_avg"        : price_vs_5day,
            "daily_avg_sentiment_lag1" : avg_sentiment,
            "daily_return_lag1"        : daily_return,
        }

        for feat, val in live_features.items():
            if feat in feature_row:
                feature_row[feat] = val

        # make prediction
        row_df   = pd.DataFrame([feature_row])[feat_names]
        prob_up  = float(xgb_model.predict_proba(row_df)[0][1])
        prob_down = 1 - prob_up
        direction = "UP" if prob_up >= 0.5 else "DOWN"

        return {
            "direction"  : direction,
            "confidence" : round(max(prob_up, prob_down) * 100, 1),
            "prob_up"    : round(prob_up * 100, 1),
            "prob_down"  : round(prob_down * 100, 1),
            "available"  : True,
        }

    except Exception as e:
        return {
            "direction"  : "UNKNOWN",
            "confidence" : 0,
            "prob_up"    : 50,
            "prob_down"  : 50,
            "available"  : False,
        }


# ══════════════════════════════════════════════════════════
# MAIN DASHBOARD UI
# ══════════════════════════════════════════════════════════

def main():

    # ── sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stocks-growth.png",
                 width=60)
        st.title("SentimentIQ")
        st.caption("Financial News Intelligence")
        st.divider()

        st.subheader("Search Company")
        ticker_input = st.text_input(
            "Enter ticker symbol",
            value="AAPL",
            placeholder="e.g. AAPL, GOOGL, TSLA",
            help="Enter any Yahoo Finance ticker. US stocks, "
                 "Indian stocks (RELIANCE.NS), crypto (BTC-USD)"
        ).upper().strip()

        st.divider()
        st.subheader("Tracked Companies")
        st.caption("Pre-computed daily signals")
        for company in COMPANIES:
            if st.button(company, key=f"btn_{company}",
                         use_container_width=True):
                ticker_input = company
                st.rerun()

        st.divider()
        st.caption("🔄 Data refreshes every hour")
        st.caption("📊 Model retrained daily")
        st.caption(
            f"🕒 Last updated: "
            f"{datetime.now().strftime('%b %d %H:%M UTC')}"
        )

    # ── main content ──────────────────────────────────────
    st.title(f"📈 {ticker_input} — Sentiment Intelligence")
    st.caption(
        "Real-time financial news sentiment analysis "
        "powered by FinBERT · Predictions by XGBoost"
    )

    # load models (cached — instant after first load)
    tokenizer, finbert_model = load_finbert_cached()
    xgb_model, feat_names    = load_xgboost_cached()

    # show model status
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if finbert_model is not None:
            st.success("✅ FinBERT NLP model loaded")
        else:
            st.warning("⚠️ NLP model loading...")
    with col_status2:
        if xgb_model is not None:
            st.success("✅ XGBoost prediction model loaded")
        else:
            st.info("ℹ️ Prediction model unavailable")

    st.divider()

    # fetch live data
    with st.spinner(f"Fetching live data for {ticker_input}..."):
        news_df  = fetch_company_news(ticker_input)
        price_df = fetch_stock_price_cached(ticker_input)

    if news_df.empty:
        st.error(
            f"No data found for ticker '{ticker_input}'. "
            f"Please check the ticker symbol and try again."
        )
        st.info(
            "Examples: AAPL (Apple), GOOGL (Google), "
            "TSLA (Tesla), RELIANCE.NS (Reliance Industries)"
        )
        return

    # score sentiment
    with st.spinner("Analysing sentiment with FinBERT..."):
        news_df = score_headlines(news_df, tokenizer, finbert_model)

    # get prediction
    prediction = make_prediction(
        ticker_input, news_df, price_df, xgb_model, feat_names
    )

    # ── row 1: key metrics ────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    avg_score = float(news_df["sentiment_score"].mean()) \
        if "sentiment_score" in news_df.columns else 0.0
    article_count = len(news_df)
    pos_count = int(
        (news_df["sentiment_label"] == "positive").sum()
    ) if "sentiment_label" in news_df.columns else 0
    neg_count = int(
        (news_df["sentiment_label"] == "negative").sum()
    ) if "sentiment_label" in news_df.columns else 0

    with col1:
        score_color = (
            "🟢" if avg_score > 0.1
            else "🔴" if avg_score < -0.1
            else "⚪"
        )
        st.metric(
            label="Avg Sentiment Score",
            value=f"{score_color} {avg_score:+.3f}",
            help="FinBERT score: +1=very positive, -1=very negative"
        )

    with col2:
        st.metric(
            label="Articles Analysed",
            value=f"📰 {article_count}",
            help="Number of news articles scored today"
        )

    with col3:
        st.metric(
            label="Positive / Negative",
            value=f"🟢 {pos_count}  🔴 {neg_count}",
            help="Article count by sentiment label"
        )

    with col4:
        if prediction["available"]:
            direction_emoji = (
                "🚀" if prediction["direction"] == "UP"
                else "📉"
            )
            st.metric(
                label="Tomorrow's Prediction",
                value=f"{direction_emoji} {prediction['direction']}",
                delta=f"{prediction['confidence']}% confidence",
                help="XGBoost model prediction for next trading day"
            )
        else:
            st.metric(
                label="Tomorrow's Prediction",
                value="⏳ Building...",
                help="Model needs more data — collecting daily"
            )

    st.divider()

    # ── row 2: charts ─────────────────────────────────────
    chart_col1, chart_col2 = st.columns([1.5, 1])

    with chart_col1:
        st.subheader("📊 Sentiment by Article")
        if "sentiment_score" in news_df.columns and not news_df.empty:
            fig = go.Figure()

            colors_map = {
                "positive": "#48bb78",
                "negative": "#fc8181",
                "neutral" : "#a0aec0",
            }

            for label in ["positive", "neutral", "negative"]:
                mask   = news_df["sentiment_label"] == label
                subset = news_df[mask]
                if subset.empty:
                    continue

                fig.add_trace(go.Bar(
                    x=subset.index,
                    y=subset["sentiment_score"],
                    name=label.capitalize(),
                    marker_color=colors_map[label],
                    text=subset["title"].str[:40] + "...",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Score: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                ))

            fig.add_hline(
                y=0, line_dash="dash",
                line_color="gray", opacity=0.5
            )
            fig.update_layout(
                title="Sentiment Score per Headline",
                xaxis_title="Article Index",
                yaxis_title="Sentiment Score (-1 to +1)",
                plot_bgcolor="#1c2130",
                paper_bgcolor="#1c2130",
                font_color="white",
                barmode="relative",
                height=350,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right",  x=1
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Score sentiment to see chart")

    with chart_col2:
        st.subheader("🎯 Sentiment Distribution")
        if "sentiment_label" in news_df.columns:
            label_counts = news_df["sentiment_label"].value_counts()

            fig_pie = go.Figure(go.Pie(
                labels=label_counts.index,
                values=label_counts.values,
                hole=0.5,
                marker=dict(
                    colors=["#48bb78", "#a0aec0", "#fc8181"]
                ),
            ))
            fig_pie.update_layout(
                plot_bgcolor="#1c2130",
                paper_bgcolor="#1c2130",
                font_color="white",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── row 3: price chart ────────────────────────────────
    if not price_df.empty:
        st.subheader(f"💹 {ticker_input} — 30-Day Price History")

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=price_df["date"],
            y=price_df["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#63b3ed", width=2),
            fill="tozeroy",
            fillcolor="rgba(99, 179, 237, 0.1)",
        ))
        fig_price.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            plot_bgcolor="#1c2130",
            paper_bgcolor="#1c2130",
            font_color="white",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── row 4: news feed ──────────────────────────────────
    st.subheader("📰 Latest Headlines with Sentiment")
    st.caption("Scored in real-time by FinBERT")

    if not news_df.empty:
        for _, row in news_df.head(10).iterrows():
            label = row.get("sentiment_label", "neutral")
            score = row.get("sentiment_score", 0.0)
            conf  = row.get("confidence", 0.0)

            emoji = (
                "🟢" if label == "positive"
                else "🔴" if label == "negative"
                else "⚪"
            )
            score_str = f"{score:+.3f}"

            with st.expander(
                f"{emoji} {score_str} | {row['title'][:80]}..."
                if len(row["title"]) > 80
                else f"{emoji} {score_str} | {row['title']}"
            ):
                st.write(f"**Summary:** {row.get('summary', 'N/A')}")
                st.write(f"**Published:** {row.get('date', 'Unknown')}")
                st.write(
                    f"**Sentiment:** {label.upper()} "
                    f"(score: {score:+.3f}, "
                    f"confidence: {conf*100:.0f}%)"
                )
                if row.get("url"):
                    st.markdown(f"[Read full article →]({row['url']})")

    # ── disclaimer ────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This tool is for informational
    and educational purposes only. It does not constitute financial
    advice. Sentiment scores and predictions are generated by AI
    models and may contain errors. Past model performance does not
    guarantee future results. Always consult a qualified financial
    advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()