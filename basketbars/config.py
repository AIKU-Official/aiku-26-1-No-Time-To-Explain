from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# basketbars/  ->  project root (one level up)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Default basket-bar dataset produced by make_dataset.py
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "basket_bars.csv")

# ---------------------------------------------------------------------------
# Date splits (date format: '%Y-%m-%d'); the basket-bar `date` column carries
# the full timestamp (e.g. '2021-01-04 10:30:00'), but lexicographic comparison
# with these YYYY-MM-DD bounds works correctly for ISO-formatted strings.
# ---------------------------------------------------------------------------
TRAIN_START_DATE = '2026-01-01'
TRAIN_END_DATE = '2026-04-31'

TRADE_START_DATE = '2026-05-01'
TRADE_END_DATE = '2026-05-31'

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Standard FinRL technical indicators (computed by AlpacaDownloader.add_technical_indicator)
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Basket-bar specific features engineered in AlpacaDownloader.make_basket_bars
CUSTOM_FEATURES = [
    "duration",        # number of 1-min bars aggregated into the basket bar
    "avg_tick_size",   # mean (volume / trade_count)
    "hhi",             # cross-sectional concentration of dollar volume
    "cs_vol",          # cross-sectional dispersion of intra-bar returns
    "agg_imbalance",   # participation-weighted price pressure
    "time_sin",        # intraday time embedding
    "time_cos",
    "day_sin",         # day-of-week embedding
    "day_cos",
]

# Full feature vector fed to the PPO state space.
# Set to INDICATORS only to reproduce the starter-kit baseline.
TECH_INDICATORS = INDICATORS + CUSTOM_FEATURES
