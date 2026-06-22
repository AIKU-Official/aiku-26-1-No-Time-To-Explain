"""Compare backtest results (test.py가 남기는 `results.csv`).

각 실험의 `experiments/<exp>/results/results.csv` (컬럼: date, account_value)를
모아 누적 수익률과 낙폭(drawdown)을 한 그림에 겹쳐 그린다. 초기 자본이 모두
$1M이므로 **누적 수익률(%)로 정규화**해 비교한다.

레이아웃 예 (세 프로젝트 공통):
  experiments/<exp>/results/results.csv

사용 예:
  # basketbars의 모든 실험을 한 번에 모아 비교
  python basketbars/plot_backtest.py experiments

  # 세 프로젝트를 한 그림에 (서로 기간이 다르면 --x step 권장)
  python basketbars/plot_backtest.py \
      experiments/exp0 \
      contest2023/task-1-stock-trading-starter-kit/experiments/baseline \
      contest2023/SZU-FIN-621/Model/experiments/baseline \
      --x step --out backtest_compare.png
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_results(path: str):
    """path(파일/폴더) 아래의 results.csv들을 경로 리스트로 반환."""
    if os.path.isfile(path) and path.endswith(".csv"):
        return [path]
    hits = glob.glob(os.path.join(path, "**", "results", "results.csv"), recursive=True)
    if not hits:  # results/ 없이 바로 results.csv 인 경우도 허용
        hits = glob.glob(os.path.join(path, "**", "results.csv"), recursive=True)
    return sorted(hits)


def exp_dir_of(csv_path: str) -> str:
    """.../<exp>/results/results.csv -> .../<exp> 경로."""
    results_dir = os.path.dirname(csv_path)          # .../<exp>/results
    if os.path.basename(results_dir) == "results":
        return os.path.dirname(results_dir)          # .../<exp>
    return results_dir


def unique_labels(exp_dirs):
    """경로들의 '가장 짧은 유일한 꼬리'를 라벨로 만든다 (프로젝트 간 동명이실험 구분)."""
    parts = [d.replace("\\", "/").rstrip("/").split("/") for d in exp_dirs]
    labels = []
    for i, pp in enumerate(parts):
        depth = 1
        while depth < len(pp):
            cand = "/".join(pp[-depth:])
            others = ["/".join(q[-depth:]) for j, q in enumerate(parts) if j != i]
            if cand not in others:
                break
            depth += 1
        labels.append("/".join(pp[-depth:]))
    return labels


def metrics(av: pd.Series):
    """누적수익률(%), 낙폭 시계열(%), 최종수익률(%), MDD(%)."""
    cumret = (av / av.iloc[0] - 1.0) * 100.0
    running_max = av.cummax()
    drawdown = (av / running_max - 1.0) * 100.0
    return cumret, drawdown, cumret.iloc[-1], drawdown.min()


def main():
    ap = argparse.ArgumentParser(description="백테스트 결과(results.csv) 비교 plot")
    ap.add_argument("paths", nargs="+",
                    help="results.csv 또는 그것을 포함하는 폴더 (여러 개 가능)")
    ap.add_argument("--labels", nargs="*", default=None, help="범례 라벨 직접 지정")
    ap.add_argument("--x", choices=["date", "step"], default="date",
                    help="x축: date(날짜) 또는 step(0~1 정규화 진행도). 기간이 다른 "
                         "run들을 겹쳐 볼 땐 step 권장.")
    ap.add_argument("--out", default="backtest_compare.png", help="저장할 PNG 경로")
    args = ap.parse_args()

    csvs = []
    for p in args.paths:
        found = find_results(p)
        if not found:
            print(f"  (경고) results.csv 없음: {p}")
        csvs.extend(found)
    if not csvs:
        raise SystemExit("results.csv를 찾지 못했습니다: " + ", ".join(args.paths))

    if args.labels:
        if len(args.labels) != len(csvs):
            raise SystemExit(f"--labels 개수({len(args.labels)})가 run 개수({len(csvs)})와 다릅니다.")
        labels = args.labels
    else:
        labels = unique_labels([exp_dir_of(c) for c in csvs])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=(args.x == "step"))

    print(f"{'run':<40} {'final_return%':>14} {'max_drawdown%':>14} {'points':>7}")
    print("-" * 78)
    for label, path in zip(labels, csvs):
        df = pd.read_csv(path)
        if "account_value" not in df.columns:
            print(f"  (건너뜀) account_value 컬럼 없음: {path}")
            continue
        av = df["account_value"].astype(float).reset_index(drop=True)
        cumret, drawdown, final_ret, mdd = metrics(av)

        if args.x == "date" and "date" in df.columns:
            x = pd.to_datetime(df["date"], errors="coerce")
        else:
            n = len(av)
            x = pd.Series([i / (n - 1) if n > 1 else 0.0 for i in range(n)])

        leg = f"{label}  ({final_ret:+.1f}%, MDD {mdd:.1f}%)"
        ax1.plot(x, cumret, linewidth=1.4, label=leg)
        ax2.plot(x, drawdown, linewidth=1.1, label=label)
        print(f"{label:<40} {final_ret:>14.2f} {mdd:>14.2f} {len(av):>7}")

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.set_title("Cumulative return (%)")
    ax1.set_ylabel("return %")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("drawdown %")
    ax2.set_xlabel("progress (0~1)" if args.x == "step" else "date")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Backtest comparison", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
