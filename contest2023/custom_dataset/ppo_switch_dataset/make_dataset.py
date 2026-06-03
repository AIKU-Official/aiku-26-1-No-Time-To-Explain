"""
make_dataset.py
---------------
Generates 4 transformed variants of real_train_data.csv by applying
window-based transformations to the `close` column per ticker.

Transformations (w=6, lag-1 applied so no lookahead):
    min   : close[t] = min(close[t-w] ... close[t-1])
    max   : close[t] = max(close[t-w] ... close[t-1])
    mean  : close[t] = mean(close[t-w] ... close[t-1])
    ema   : close[t] = EWM(alpha=beta)[t-1],  beta = 2/7

Output files (same dir as this script):
    min_train_data.csv
    max_train_data.csv
    mean_train_data.csv
    ema_train_data.csv

Usage:
    python make_dataset.py                    # from project root or script dir
    python make_dataset.py --window 6         # explicit window size
    python make_dataset.py --no-shift         # no lag (include current day)
"""

import os
import argparse
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_FILE   = os.path.join(SCRIPT_DIR, "real_train_data.csv")

WINDOW = 6       # rolling window size
BETA   = 2 / 7   # EMA decay factor  (alpha in pandas ewm)


# ── transformation helpers (per-ticker Series → transformed Series) ─────────
def apply_min(s: pd.Series, w: int, shift: bool) -> pd.Series:
    result = s.rolling(window=w, min_periods=1).min()
    return result.shift(1) if shift else result


def apply_max(s: pd.Series, w: int, shift: bool) -> pd.Series:
    result = s.rolling(window=w, min_periods=1).max()
    return result.shift(1) if shift else result


def apply_mean(s: pd.Series, w: int, shift: bool) -> pd.Series:
    result = s.rolling(window=w, min_periods=1).mean()
    return result.shift(1) if shift else result


def apply_ema(s: pd.Series, beta: float, shift: bool) -> pd.Series:
    """
    Recursive EMA:  ema[t] = (1 - beta) * ema[t-1] + beta * close[t]
    pandas: ewm(alpha=beta, adjust=False).mean()
    adjust=False → uses the recursive (non-batch) definition matching the paper.
    """
    result = s.ewm(alpha=beta, adjust=False).mean()
    return result.shift(1) if shift else result


# ── main ────────────────────────────────────────────────────────────────────
def make_datasets(window: int = WINDOW, beta: float = BETA, shift: bool = True):
    print(f"Loading: {SRC_FILE}")
    df = pd.read_csv(SRC_FILE, index_col=0)

    transforms = {
        "min":  ("rolling", apply_min),
        "max":  ("rolling", apply_max),
        "mean": ("rolling", apply_mean),
        "ema":  ("ema",     apply_ema),
    }

    for name, (kind, fn) in transforms.items():
        out = df.copy()

        # apply per-ticker so rolling/ewm stays within each asset's time series
        if kind == "rolling":
            transformed_close = (
                out.groupby("tic")["close"]
                   .transform(lambda s: fn(s, window, shift))
            )
        else:  # ema
            transformed_close = (
                out.groupby("tic")["close"]
                   .transform(lambda s: fn(s, beta, shift))
            )

        out["close"] = transformed_close

        # rows where close becomes NaN (first row per ticker after lag-1 shift)
        n_nan = out["close"].isna().sum()
        if n_nan:
            print(f"  [{name}] dropping {n_nan} NaN rows (first row per ticker after lag-1 shift)")
            out.dropna(subset=["close"], inplace=True)
            out.reset_index(drop=True, inplace=True)

        out_path = os.path.join(SCRIPT_DIR, f"{name}_train_data.csv")
        out.to_csv(out_path)
        print(f"  [{name}] saved → {out_path}  (rows: {len(out)})")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PPO-Switch transformed datasets")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Rolling window size for min/max/mean (default: {WINDOW})")
    parser.add_argument("--beta", type=float, default=BETA,
                        help=f"EMA decay factor beta (default: {BETA:.4f} = 2/7)")
    parser.add_argument("--no-shift", action="store_true",
                        help="Disable lag-1 shift (include current day in window, lookahead)")
    args = parser.parse_args()

    make_datasets(window=args.window, beta=args.beta, shift=not args.no_shift)
