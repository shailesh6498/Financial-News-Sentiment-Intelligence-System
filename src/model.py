"""
model.py
────────────────────────────────────────────────────────────
ML model training, evaluation, and saving for the
Financial Sentiment Intelligence System.

WHAT this file does:
    Takes the feature matrix from Phase 4 and trains
    an XGBoost classifier to predict next-day stock
    price direction (up=1, down=0).

WHY XGBoost:
    - Works excellently on small tabular datasets (120 rows)
    - Handles missing values natively (no imputation needed)
    - Produces feature importance — you can explain every
      prediction, which is essential for financial systems
    - Industry standard: used at Amazon, Google, Two Sigma
    - Trains in seconds, not hours

DESIGN PRINCIPLE — baseline first:
    Every ML system at every MAANG company starts with
    the simplest model that could work. XGBoost is that
    model for tabular financial data. You measure its
    performance, then build LSTM to compare against it.
    Without a baseline you cannot justify complexity.

REAL WORLD CONTEXT:
    Two Sigma hedge fund ($60B AUM) uses gradient boosted
    trees as their primary model class for tabular signals.
    Renaissance Technologies, the most successful hedge
    fund in history, uses similar ensemble methods.
    You are using the same class of model.

Used by:
    notebooks/05_modelling.ipynb
    app/dashboard.py (Phase 6 — loads saved model)
────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import (
    FEATURES_PATH,
    TARGET_COLUMN,
    NON_FEATURE_COLUMNS,
    TRAIN_TEST_SPLIT_DATE,
    XGBOOST_PARAMS,
    XGBOOST_MODEL_PATH,
    MODEL_METRICS_PATH,
    REPORTS_PATH,
    COLORS,
    BASE_DIR,
)
from src.utils import get_logger, ensure_dirs

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# STEP 1 — PREPARE DATA FOR TRAINING
# ══════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepares the feature matrix for model training.

    WHAT this does:
        1. Selects only predictive feature columns
        2. Splits into train/test by date (not random)
        3. Separates features (X) from target (y)
        4. Handles remaining missing values

    WHY we exclude certain columns:
        NON_FEATURE_COLUMNS contains identifiers (ticker,
        date) and raw price columns. We exclude raw prices
        because they create scale problems — Apple at $291
        and a small-cap at $4 are incomparable. We already
        engineered normalised features (daily_return,
        price_vs_5day_avg) that capture the same information
        without the scale problem.

    WHY time-based split not random:
        Random split on time series creates data leakage.
        If the model trains on May 10th data and tests on
        April 15th data, it has seen the future during
        training. Time-based split means the model always
        learns from the past and predicts the future —
        exactly as it will work in production.

    Args:
        df: complete feature matrix from Phase 4

    Returns:
        tuple of (X_train, X_test, y_train, y_test,
                  feature_names, train_df, test_df)
    """
    logger.info("Preparing features for training...")

    # ── select feature columns ─────────────────────────
    all_cols      = df.columns.tolist()
    feature_cols  = [
        c for c in all_cols
        if c not in NON_FEATURE_COLUMNS
        and c != TARGET_COLUMN
    ]

    logger.info(f"Feature columns selected: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")

    # ── time-based train/test split ────────────────────
    # WHY: preserves temporal order — model trains on past,
    # tests on future. Never shuffle time series data.
    df["date_parsed"] = pd.to_datetime(
        df["date_clean"], errors="coerce"
    )
    split_date = pd.to_datetime(TRAIN_TEST_SPLIT_DATE)

    train_df = df[df["date_parsed"] <  split_date].copy()
    test_df  = df[df["date_parsed"] >= split_date].copy()

    logger.info(
        f"Train set: {len(train_df)} rows "
        f"({train_df['date_clean'].min()} to "
        f"{train_df['date_clean'].max()})"
    )
    logger.info(
        f"Test set : {len(test_df)} rows "
        f"({test_df['date_clean'].min()} to "
        f"{test_df['date_clean'].max()})"
    )

    # ── separate features and target ──────────────────
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[TARGET_COLUMN].copy()

    # ── handle missing values ─────────────────────────
    # WHY fill with 0: missing lag features (first rows
    # per company) genuinely mean "no prior data" —
    # 0 is the correct neutral value, not an imputation
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # drop rows where target is missing
    # (last row per ticker has no next-day price)
    valid_train = y_train.notna()
    valid_test  = y_test.notna()
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_test  = X_test[valid_test]
    y_test  = y_test[valid_test]

    logger.info(
        f"Final shapes — "
        f"X_train: {X_train.shape}, "
        f"X_test: {X_test.shape}"
    )

    # class distribution
    train_up_pct = y_train.mean() * 100
    test_up_pct  = y_test.mean() * 100
    logger.info(
        f"Target distribution — "
        f"Train: {train_up_pct:.1f}% up | "
        f"Test: {test_up_pct:.1f}% up"
    )

    return (
        X_train, X_test,
        y_train, y_test,
        feature_cols,
        train_df, test_df
    )


# ══════════════════════════════════════════════════════════
# STEP 2 — TRAIN XGBOOST
# ══════════════════════════════════════════════════════════

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """
    Trains an XGBoost classifier with cross-validation.

    WHAT is XGBoost doing internally:
        XGBoost builds decision trees one at a time.
        Each new tree focuses entirely on fixing the
        errors the previous trees made. After 100 trees,
        the combined prediction is much more accurate
        than any single tree.

        This is called gradient boosting because it
        uses the mathematical gradient (direction of
        steepest error) to decide what to fix next.
        Like a hiker finding the fastest route downhill
        by always stepping in the direction of steepest
        descent.

    WHAT each hyperparameter does:

    n_estimators=100:
        Build 100 trees. More trees = more accurate but
        slower and risks overfitting. 100 is the sweet
        spot for small datasets.

    max_depth=4:
        Each tree can ask at most 4 yes/no questions.
        "Is sentiment > 0.3? Is it Monday? Is volume up?"
        Deeper trees memorise training data (overfit).
        Shallow trees generalise better to new data.

    learning_rate=0.1:
        How much each new tree corrects the previous ones.
        High rate = fast learning but overshoots.
        Low rate = slow but more precise.
        0.1 is the industry standard starting point.

    subsample=0.8:
        Each tree only sees 80% of training rows,
        chosen randomly. This prevents any single
        tree from memorising the full training set.
        Called "bagging" — a key technique at Google
        and Amazon for preventing overfitting.

    colsample_bytree=0.8:
        Each tree only sees 80% of features.
        Forces the model to find multiple independent
        signals rather than relying on one feature.

    scale_pos_weight:
        Handles class imbalance. You have 68 up days
        and 52 down days. Without this, the model
        would be biased toward predicting "up" because
        it is more common. scale_pos_weight tells the
        model to weight the minority class higher.
        Formula: count(negative) / count(positive)

    Args:
        X_train: training features DataFrame
        y_train: training target Series

    Returns:
        trained XGBoost model object
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    logger.info("Training XGBoost classifier...")
    logger.info(f"Training on {len(X_train)} rows, {X_train.shape[1]} features")

    # ── calculate class weight ─────────────────────────
    # WHY: your data has more up days than down days (57/43)
    # Without weighting, model learns "just predict up"
    # and gets 57% accuracy doing nothing intelligent
    n_negative  = int((y_train == 0).sum())
    n_positive  = int((y_train == 1).sum())
    scale_weight = n_negative / max(n_positive, 1)

    logger.info(
        f"Class balance — up: {n_positive}, down: {n_negative}, "
        f"scale_weight: {scale_weight:.3f}"
    )

    # ── build model ────────────────────────────────────
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = scale_weight

    # remove keys that newer XGBoost versions reject
    params.pop("use_label_encoder", None)

    model = XGBClassifier(**params)

    # ── cross-validation ───────────────────────────────
    # WHY cross-validation:
    # With only 96 training rows, any single train/test
    # split might be lucky or unlucky. Cross-validation
    # splits the training data into 5 folds, trains on 4,
    # tests on 1, rotates through all combinations.
    # The average of 5 evaluations is a more reliable
    # estimate of true model performance.
    #
    # StratifiedKFold ensures each fold has the same
    # class ratio as the full dataset — important when
    # classes are imbalanced.

    cv = StratifiedKFold(n_splits=5, shuffle=False)

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="roc_auc"
    )

    logger.info(
        f"Cross-validation AUC scores: "
        f"{[round(s, 3) for s in cv_scores]}"
    )
    logger.info(
        f"Mean CV AUC: {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )

    # ── final training on full training set ───────────
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    logger.info("XGBoost training complete")
    return model


# ══════════════════════════════════════════════════════════
# STEP 3 — EVALUATE THE MODEL
# ══════════════════════════════════════════════════════════

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> dict:
    """
    Comprehensive model evaluation with 5 metrics.

    WHAT each metric tells you and WHY it matters:

    1. ACCURACY
       What: % of all predictions that are correct
       Why it can mislead: if 70% of days go up, a
       model that ALWAYS predicts up gets 70% accuracy
       without learning anything. Always check alongside
       precision and recall.

    2. PRECISION
       What: of all days we predicted UP, how many
       actually went up?
       Real world: precision matters for investors who
       only trade when they have high confidence. High
       precision = fewer false signals.

    3. RECALL
       What: of all actual UP days, how many did we
       correctly identify?
       Real world: recall matters for investors who
       want to catch every opportunity. High recall =
       fewer missed opportunities.

    4. AUC-ROC (Area Under the Curve)
       What: measures the model's ability to distinguish
       between up days and down days at any threshold.
       0.5 = no better than random. 1.0 = perfect.
       Why it is the gold standard: works correctly
       even with imbalanced classes. Used by every
       serious ML team at MAANG.

    5. FEATURE IMPORTANCE
       What: which features did XGBoost rely on most
       to make its predictions?
       Why it matters: if the model says sentiment
       features barely mattered and price momentum
       drove everything — that is a finding. It means
       your sentiment pipeline is not yet contributing
       signal. Honest self-assessment. Then you
       investigate why and improve.

    Args:
        model: trained XGBoost model
        X_test: test features
        y_test: test target
        feature_names: list of feature column names

    Returns:
        dict of all metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
    )

    logger.info("Evaluating model on held-out test set...")

    # ── generate predictions ───────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # ── compute all metrics ────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_pred_prob)
    cm        = confusion_matrix(y_test, y_pred)

    # ── feature importance ─────────────────────────────
    importance_scores = model.feature_importances_
    feature_importance = dict(
        zip(feature_names, importance_scores)
    )
    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    metrics = {
        "accuracy"          : round(float(accuracy),  4),
        "precision"         : round(float(precision), 4),
        "recall"            : round(float(recall),    4),
        "f1_score"          : round(float(f1),        4),
        "auc_roc"           : round(float(auc),       4),
        "confusion_matrix"  : cm.tolist(),
        "feature_importance": feature_importance,
        "n_test_samples"    : int(len(y_test)),
        "n_train_samples"   : int(len(model.get_booster()
                               .get_dump())),
        "test_up_pct"       : round(float(y_test.mean()*100), 1),
        "pred_up_pct"       : round(float(
                               pd.Series(y_pred).mean()*100), 1),
    }

    # ── log human-readable summary ─────────────────────
    logger.info("\n" + "="*50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Test samples    : {metrics['n_test_samples']}")
    logger.info(f"Accuracy        : {metrics['accuracy']*100:.1f}%")
    logger.info(f"Precision       : {metrics['precision']*100:.1f}%")
    logger.info(f"Recall          : {metrics['recall']*100:.1f}%")
    logger.info(f"F1 Score        : {metrics['f1_score']*100:.1f}%")
    logger.info(f"AUC-ROC         : {metrics['auc_roc']:.4f}")
    logger.info(f"\nTop 5 features by importance:")
    for feat, score in list(feature_importance.items())[:5]:
        bar = "█" * int(score * 100)
        logger.info(f"  {feat:<35} {bar} ({score:.4f})")
    logger.info("\n" + classification_report(y_test, y_pred))

    return metrics


# ══════════════════════════════════════════════════════════
# STEP 4 — VISUALISE RESULTS
# ══════════════════════════════════════════════════════════

def plot_model_results(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict,
    feature_names: list[str],
) -> None:
    """
    Produces three essential model evaluation charts.

    Chart 1 — Feature Importance (horizontal bar chart)
        Shows which features XGBoost relied on most.
        If sentiment features rank high — your NLP
        pipeline is adding real signal.
        If price features dominate — sentiment alone
        is not predictive yet (common with small datasets).

    Chart 2 — ROC Curve
        Plots True Positive Rate vs False Positive Rate
        at every possible prediction threshold.
        The area under this curve (AUC) is your headline
        metric. Diagonal line = random guessing = 0.5 AUC.
        Your curve should bow toward the top-left corner.

    Chart 3 — Confusion Matrix
        Shows exactly where the model is right and wrong.
        Top-left:  predicted DOWN, actually DOWN (correct)
        Top-right: predicted DOWN, actually UP (wrong)
        Bottom-left: predicted UP, actually DOWN (wrong)
        Bottom-right: predicted UP, actually UP (correct)

    Args:
        model: trained XGBoost model
        X_test, y_test: test data
        metrics: output from evaluate_model()
        feature_names: list of feature names
    """
    from sklearn.metrics import roc_curve

    ensure_dirs(REPORTS_PATH)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        "Phase 5 — XGBoost Model Evaluation",
        fontweight="bold", fontsize=14
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── chart 1: feature importance ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    feat_imp = metrics["feature_importance"]
    top_n    = 15
    top_feats  = list(feat_imp.keys())[:top_n]
    top_scores = list(feat_imp.values())[:top_n]

    colors_fi = [
        "#5B8DB8" if "sentiment" in f
        else "#6BAF7A" if "return" in f or "price" in f
        else "#C9637A" if "volume" in f
        else "#9B7EC8"
        for f in top_feats
    ]

    ax1.barh(
        range(len(top_feats)), top_scores,
        color=colors_fi, alpha=0.85, zorder=3
    )
    ax1.set_yticks(range(len(top_feats)))
    ax1.set_yticklabels(
        [f[:30] for f in top_feats], fontsize=8
    )
    ax1.invert_yaxis()
    ax1.set_title(
        "Feature Importance\n"
        "(blue=sentiment, green=price, red=volume)",
        fontsize=10
    )
    ax1.set_xlabel("Importance Score")
    ax1.grid(axis="x", alpha=0.3)

    # ── chart 2: ROC curve ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_val      = metrics["auc_roc"]

    ax2.plot(
        fpr, tpr,
        color="#5B8DB8", linewidth=2.5,
        label=f"XGBoost (AUC = {auc_val:.3f})"
    )
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#5B8DB8")
    ax2.plot(
        [0, 1], [0, 1],
        color="gray", linestyle="--",
        linewidth=1.5, label="Random (AUC = 0.500)"
    )
    ax2.set_title("ROC Curve\n(higher and left = better)", fontsize=10)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    # ── chart 3: confusion matrix ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])

    cm = np.array(metrics["confusion_matrix"])
    im = ax3.imshow(cm, cmap="Blues", aspect="auto")

    labels = ["Predicted DOWN", "Predicted UP"]
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_yticklabels(
        ["Actual DOWN", "Actual UP"], fontsize=9
    )
    ax3.set_title(
        "Confusion Matrix\n"
        "(diagonal = correct predictions)",
        fontsize=10
    )

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax3.text(
                j, i, str(val),
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color=color
            )

    plt.colorbar(im, ax=ax3, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "08_model_evaluation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Evaluation chart saved: {path}")
    plt.show()


# ══════════════════════════════════════════════════════════
# STEP 5 — SAVE MODEL AND METRICS
# ══════════════════════════════════════════════════════════

def save_model(model, metrics: dict, feature_names: list) -> None:
    """
    Saves the trained model and metrics to disk.

    WHY save the model:
        Training takes time. Once trained you save the
        model to a .pkl file (pickle — Python's way of
        serialising objects to disk). The dashboard loads
        this file instantly instead of retraining every time.

    WHY save metrics separately as JSON:
        The dashboard needs to display accuracy, AUC etc.
        without loading the full model. JSON is human-
        readable and loadable in 1 line. Your README can
        show live metrics by reading this file.

    WHY save feature names:
        The model was trained on a specific set of features
        in a specific order. When the dashboard feeds new
        data to the model, it must use the exact same
        features in the exact same order. Saving feature
        names alongside the model prevents this mismatch.

    Args:
        model: trained XGBoost model
        metrics: evaluation metrics dict
        feature_names: ordered list of feature names
    """
    ensure_dirs(os.path.dirname(XGBOOST_MODEL_PATH))
    ensure_dirs(os.path.dirname(MODEL_METRICS_PATH))

    # save model as pickle
    with open(XGBOOST_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model"        : model,
            "feature_names": feature_names,
            "model_type"   : "XGBoost",
            "target"       : TARGET_COLUMN,
        }, f)
    logger.info(f"Model saved: {XGBOOST_MODEL_PATH}")

    # save metrics as JSON — human readable
    metrics_to_save = {
        k: v for k, v in metrics.items()
        if k != "feature_importance"
    }
    metrics_to_save["top_features"] = list(
        metrics["feature_importance"].keys()
    )[:10]
    metrics_to_save["top_feature_scores"] = [
        round(float(v), 4)
        for v in list(
            metrics["feature_importance"].values()
        )[:10]
    ]

    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    logger.info(f"Metrics saved: {MODEL_METRICS_PATH}")


# ══════════════════════════════════════════════════════════
# STEP 6 — PREDICT ON NEW DATA (used by dashboard)
# ══════════════════════════════════════════════════════════

def load_model() -> tuple:
    """
    Loads the saved model from disk.

    This is what the Streamlit dashboard calls.
    It loads the model once when the app starts,
    then reuses it for every prediction.

    Returns:
        tuple of (model, feature_names)
    """
    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {XGBOOST_MODEL_PATH}. "
            f"Run Phase 5 notebook first."
        )

    with open(XGBOOST_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    logger.info(
        f"Model loaded: {saved['model_type']} "
        f"({len(saved['feature_names'])} features)"
    )
    return saved["model"], saved["feature_names"]


def predict_for_ticker(
    ticker: str,
    feature_row: pd.DataFrame,
) -> dict:
    """
    Makes a prediction for a single ticker using
    the latest available features.

    This is the function the dashboard calls for
    each company. It loads the model, aligns features,
    and returns a prediction with confidence.

    Args:
        ticker: company ticker e.g. "AAPL"
        feature_row: one row of features for this ticker

    Returns:
        dict with prediction, confidence, and label
    """
    model, feature_names = load_model()

    # align feature order — must match training order exactly
    row = feature_row[feature_names].fillna(0)

    prob_up   = float(model.predict_proba(row)[0][1])
    prob_down = 1 - prob_up
    direction = "UP" if prob_up >= 0.5 else "DOWN"

    return {
        "ticker"    : ticker,
        "direction" : direction,
        "confidence": round(max(prob_up, prob_down) * 100, 1),
        "prob_up"   : round(prob_up * 100, 1),
        "prob_down" : round(prob_down * 100, 1),
    }


# ══════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════

def run_full_pipeline(
    features_path: str = FEATURES_PATH,
) -> tuple:
    """
    Runs the complete Phase 5 pipeline end to end.

    One function call:
    load → prepare → train → evaluate → visualise → save

    Args:
        features_path: path to feature matrix CSV

    Returns:
        tuple of (model, metrics, feature_names)
    """
    logger.info("=" * 55)
    logger.info("Phase 5 — ML Modelling Pipeline")
    logger.info("=" * 55)

    # load
    df = pd.read_csv(features_path)
    logger.info(f"Loaded feature matrix: {df.shape}")

    # prepare
    (
        X_train, X_test,
        y_train, y_test,
        feature_names,
        train_df, test_df
    ) = prepare_features(df)

    # train
    model = train_xgboost(X_train, y_train)

    # evaluate
    metrics = evaluate_model(
        model, X_test, y_test, feature_names
    )

    # visualise
    plot_model_results(
        model, X_test, y_test, metrics, feature_names
    )

    # save
    save_model(model, metrics, feature_names)

    logger.info("=" * 55)
    logger.info("Phase 5 complete")
    logger.info(f"Model accuracy : {metrics['accuracy']*100:.1f}%")
    logger.info(f"AUC-ROC        : {metrics['auc_roc']:.4f}")
    logger.info(f"Model saved to : {XGBOOST_MODEL_PATH}")
    logger.info("=" * 55)

    return model, metrics, feature_names
