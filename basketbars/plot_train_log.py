"""Visualize PPO training logs (SB3 / FinRL `progress.csv`).

세 프로젝트(basketbars / starter-kit / SZU-FIN-621)가 모두 SB3 CSV 로거로 남기는
`progress.csv`를 읽어 학습 곡선을 그린다. 경로를 여러 개 주면 한 그림에 겹쳐
비교할 수 있다.

레이아웃 예:
  basketbars   : experiments/<exp>/results/ppo/progress.csv
  starter-kit  : experiments/<exp>/results/ppo/progress.csv
  SZU-FIN-621  : experiments/<exp>/results/ppo_{ema,max,mean,min,real}/progress.csv

사용 예:
  # 한 실험의 모든 run(ppo*)을 겹쳐 그리기 (SZU의 5개 변형 비교)
  python basketbars/plot_train_log.py \
      contest2023/SZU-FIN-621/Model/experiments/baseline

  # 세 프로젝트의 대표 run을 한 그림에 비교
  python basketbars/plot_train_log.py \
      experiments/exp0 \
      contest2023/task-1-stock-trading-starter-kit/experiments/baseline \
      contest2023/SZU-FIN-621/Model/experiments/baseline/results/ppo_mean \
      --out compare.png --smooth 3
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

X = "time/total_timesteps"

# FinRL DRLAgent의 SB3 CSV 로거 컬럼 기준. 로그에 있는 것만 자동으로 그린다.
DEFAULT_METRICS = [
    "train/reward_mean",
    "train/loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/explained_variance",
]


def find_runs(path: str):
    """path(파일 또는 폴더) 아래의 progress.csv들을 csv 경로 리스트로 반환."""
    if os.path.isfile(path) and path.endswith(".csv"):
        return [path]
    return sorted(glob.glob(os.path.join(path, "**", "progress.csv"), recursive=True))


def make_label(csv_path: str) -> str:
    """.../<exp>/results/<run>/progress.csv -> '<exp>/<run>'."""
    run = os.path.basename(os.path.dirname(csv_path))            # ppo / ppo_ema ...
    results = os.path.dirname(os.path.dirname(csv_path))         # .../results
    exp = os.path.basename(os.path.dirname(results))             # <exp>
    return f"{exp}/{run}" if exp else run


def smooth(series: pd.Series, k: int) -> pd.Series:
    return series.rolling(k, min_periods=1).mean() if k and k > 1 else series


def main():
    ap = argparse.ArgumentParser(description="PPO 학습 로그(progress.csv) 시각화")
    ap.add_argument("paths", nargs="+",
                    help="progress.csv 또는 그것을 포함하는 폴더 (여러 개 가능)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="각 run 라벨 (run 개수와 일치해야 함)")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                    help="그릴 metric 컬럼들 (기본: reward_mean/loss/value_loss/...)")
    ap.add_argument("--smooth", type=int, default=1,
                    help="이동평균 윈도우 (기본 1 = 원본 그대로)")
    ap.add_argument("--out", default="train_log.png", help="저장할 PNG 경로")
    args = ap.parse_args()

    # 경로 -> run 목록
    csvs = []
    for p in args.paths:
        found = find_runs(p)
        if not found:
            print(f"  (경고) progress.csv 없음: {p}")
        csvs.extend(found)
    if not csvs:
        raise SystemExit("progress.csv를 찾지 못했습니다: " + ", ".join(args.paths))

    if args.labels:
        if len(args.labels) != len(csvs):
            raise SystemExit(
                f"--labels 개수({len(args.labels)})가 run 개수({len(csvs)})와 다릅니다."
            )
        labels = args.labels
    else:
        labels = [make_label(c) for c in csvs]

    dfs = []
    for label, path in zip(labels, csvs):
        df = pd.read_csv(path)
        dfs.append((label, df))
        print(f"loaded: {label}  ({len(df)} rows)  <- {path}")

    metrics = [m for m in args.metrics if any(m in df.columns for _, df in dfs)]
    if not metrics:
        raise SystemExit("선택한 metric이 어떤 로그에도 없습니다: " + ", ".join(args.metrics))

    ncol = 2 if len(metrics) > 1 else 1
    nrow = (len(metrics) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        for label, df in dfs:
            if metric not in df.columns or X not in df.columns:
                continue
            sub = df[[X, metric]].dropna()
            if sub.empty:
                continue
            ax.plot(sub[X], smooth(sub[metric], args.smooth), label=label, linewidth=1.3)
        ax.set_title(metric)
        ax.set_xlabel("total_timesteps")
        ax.grid(True, alpha=0.3)

    # 범례는 run이 여러 개일 때 첫 패널에 한 번만
    if len(dfs) > 1:
        axes[0].legend(fontsize=8)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle("PPO training logs", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
