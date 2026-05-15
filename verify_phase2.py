"""
verify_phase2.py
────────────────────────────────────────────────────────────
Run this before Phase 2 to confirm every dependency exists.
This is called a smoke test — a quick sanity check that
everything is wired up before you invest time running the
full pipeline.

Real-world context:
    Every CI/CD pipeline at MAANG starts with smoke tests.
    Before any deployment, a set of fast checks confirm
    the environment is sane. You are building that habit.

Run with:
    python verify_phase2.py
────────────────────────────────────────────────────────────
"""

import os
import sys

print("=" * 55)
print("Phase 2 Pre-flight Verification")
print("=" * 55)

all_passed = True

def check(label: str, condition: bool, fix: str = "") -> None:
    """Prints a pass/fail line and tracks overall status."""
    global all_passed
    status = "✅" if condition else "❌"
    print(f"  {status}  {label}")
    if not condition:
        all_passed = False
        if fix:
            print(f"       FIX: {fix}")


# ── 1. check required files exist ────────────────────────
print("\n── Files ───────────────────────────────────────────")

check(
    "config.py exists",
    os.path.exists("config.py"),
    "Create config.py in project root"
)
check(
    "src/utils.py exists",
    os.path.exists("src/utils.py"),
    "Create src/utils.py"
)
check(
    "src/data_collector.py exists",
    os.path.exists("src/data_collector.py"),
    "Create src/data_collector.py"
)
check(
    "src/eda.py exists",
    os.path.exists("src/eda.py"),
    "Create src/eda.py — see Phase 2 instructions"
)
check(
    "data/raw/all_news.csv exists",
    os.path.exists("data/raw/all_news.csv"),
    "Run notebooks/01_data_collection.ipynb first"
)
check(
    "data/processed/ folder exists",
    os.path.exists("data/processed"),
    "Run: mkdir data/processed"
)
check(
    "reports/figures/ folder exists",
    os.path.exists("reports/figures"),
    "Run: mkdir reports && mkdir reports/figures"
)

# ── 2. check imports work ────────────────────────────────
print("\n── Imports ─────────────────────────────────────────")

try:
    import pandas as pd
    check("pandas importable", True)
except ImportError:
    check("pandas importable", False, "pip install pandas")

try:
    import matplotlib
    check("matplotlib importable", True)
except ImportError:
    check("matplotlib importable", False, "pip install matplotlib")

try:
    import numpy
    check("numpy importable", True)
except ImportError:
    check("numpy importable", False, "pip install numpy")

try:
    sys.path.insert(0, ".")
    from config import COMPANIES, RAW_DATA_PATH, PROCESSED_DATA_PATH
    check("config.py imports cleanly", True)
except Exception as e:
    check("config.py imports cleanly", False, str(e))

try:
    from src.utils import get_logger, parse_yahoo_date, validate_article
    check("src/utils.py imports cleanly", True)
except Exception as e:
    check("src/utils.py imports cleanly", False, str(e))

try:
    from src.data_collector import collect_all_companies
    check("src/data_collector.py imports cleanly", True)
except Exception as e:
    check("src/data_collector.py imports cleanly", False, str(e))

try:
    from src.eda import (
        structural_summary,
        check_duplicates,
        plot_article_distribution,
        plot_coverage_heatmap,
        compute_text_stats,
        plot_word_frequency,
        plot_temporal_patterns,
        clean_and_save,
    )
    check("src/eda.py imports cleanly", True)
except Exception as e:
    check("src/eda.py imports cleanly", False, str(e))

# ── 3. check data quality ────────────────────────────────
print("\n── Data ────────────────────────────────────────────")

if os.path.exists("data/raw/all_news.csv"):
    try:
        import pandas as pd
        df = pd.read_csv("data/raw/all_news.csv")

        check(
            f"all_news.csv has rows ({len(df)} found)",
            len(df) > 0,
            "Re-run Phase 1 notebook"
        )
        check(
            "required columns present (ticker, title, date)",
            all(c in df.columns for c in ["ticker","title","date"]),
            "Check Phase 1 output columns"
        )
        check(
            f"has data for 5 companies ({df['ticker'].nunique()} found)",
            df["ticker"].nunique() == 5,
            "Re-run Phase 1 with all 5 companies"
        )
        check(
            "no fully empty title column",
            df["title"].notna().all(),
            "Check data_collector.py parse_article function"
        )
    except Exception as e:
        check("all_news.csv readable", False, str(e))

# ── 4. final verdict ─────────────────────────────────────
print("\n" + "=" * 55)
if all_passed:
    print("✅ ALL CHECKS PASSED — ready to run Phase 2 notebook")
else:
    print("❌ SOME CHECKS FAILED — fix the issues above first")
    print("   Then re-run: python verify_phase2.py")
print("=" * 55)