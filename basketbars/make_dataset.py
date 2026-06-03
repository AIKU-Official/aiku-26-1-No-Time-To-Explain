"""Build a basket-bar dataset from Alpaca 1-minute data.

Fetches 1-minute bars via AlpacaDownloader, aggregates them into dollar-based
basket bars (with technical indicators + custom microstructure features), and
saves a FinRL-compatible CSV that train.py / test.py consume.

Examples
--------
# Default DJIA-30 dataset over the full configured span
python basketbars/make_dataset.py

# Custom tickers / range / output path
python basketbars/make_dataset.py \
    --start_date 2021-01-01 --end_date 2021-12-31 \
    --tickers AAPL AMZN MSFT \
    --num_bars_per_day 10 \
    --out data/basket_bars_2021.csv
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import config_tickers
from config import DATA_DIR, DEFAULT_DATA_FILE, TRAIN_START_DATE, TRADE_END_DATE
from downloader import AlpacaDownloader


def parse_args():
    p = argparse.ArgumentParser(description="Basket-bar dataset 생성 스크립트")
    p.add_argument("--start_date", default=TRAIN_START_DATE, help="YYYY-MM-DD")
    p.add_argument("--end_date", default=TRADE_END_DATE, help="YYYY-MM-DD")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=config_tickers.DJIA30_TICKER,
        help="종목 리스트 (공백 구분). 미지정 시 DJIA30 전체.",
    )
    p.add_argument(
        "--num_bars_per_day",
        type=int,
        default=10,
        help="하루 평균 목표 basket bar 개수 (threshold 산출에 사용)",
    )
    p.add_argument(
        "--chunk_months",
        type=int,
        default=1,
        help="1분봉을 몇 개월 단위 청크로 받을지 (진행률/캐시 단위)",
    )
    p.add_argument("--cache_dir", default=None, help="raw 청크 캐시 폴더 (기본: data/raw_cache)")
    p.add_argument("--no_cache", action="store_true", help="캐시 무시하고 항상 새로 받기")
    p.add_argument("--out", default=DEFAULT_DATA_FILE, help="저장할 CSV 경로")
    return p.parse_args()


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main():
    args = parse_args()

    downloader = AlpacaDownloader(
        start_date=_to_dt(args.start_date),
        end_date=_to_dt(args.end_date),
        ticker_list=args.tickers,
    )

    print(f"[1/3] 1분봉 다운로드(청크 {args.chunk_months}개월): "
          f"{args.start_date} ~ {args.end_date}, {len(args.tickers)} 종목")
    data_df = downloader.fetch_data_chunked(
        months_per_chunk=args.chunk_months,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )

    print(f"[2/3] basket bars 생성 (num_bars_per_day={args.num_bars_per_day})")
    # threshold()는 num_bars_per_day 인자를 받지만 make_basket_bars 내부에서
    # 기본값으로 호출되므로, 사용자가 지정한 값을 반영하기 위해 monkey-patch한다.
    _orig_threshold = downloader.threshold
    downloader.threshold = lambda bd, num_bars_per_day=args.num_bars_per_day: _orig_threshold(
        bd, num_bars_per_day
    )
    basket_df = downloader.make_basket_bars(data_df)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or DATA_DIR, exist_ok=True)
    basket_df.to_csv(args.out, index=False)

    print(f"[3/3] 저장 완료 -> {args.out}")
    print(f"  shape: {basket_df.shape}")
    print(f"  tickers: {basket_df['tic'].nunique()}")
    print(f"  bars(=timesteps): {basket_df['date'].nunique()}")
    print(f"  date range: {basket_df['date'].min()} ~ {basket_df['date'].max()}")
    print(f"  NaN: {basket_df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
