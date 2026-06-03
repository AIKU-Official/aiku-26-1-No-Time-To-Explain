"""Backtest a trained PPO agent on a basket-bar dataset.

Keeps the contest-required CLI:
    python basketbars/test.py --start_date 2021-11-01 --end_date 2021-12-01 \
        --data_file data/basket_bars.csv --exp_name exp0

Generates results.csv (account value per timestep) and plot.png in the
experiment's results directory, matching the starter-kit output format.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.plot import backtest_stats
from stable_baselines3 import PPO

from config import (
    DEFAULT_DATA_FILE,
    EXPERIMENTS_DIR,
    TECH_INDICATORS,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}


def parse_args():
    p = argparse.ArgumentParser(description="basket-bar PPO 백테스트")
    p.add_argument("--start_date", default=TRADE_START_DATE, help="Trade start (YYYY-MM-DD)")
    p.add_argument("--end_date", default=TRADE_END_DATE, help="Trade end (YYYY-MM-DD)")
    p.add_argument("--data_file", default=DEFAULT_DATA_FILE, help="basket-bar CSV 경로")
    p.add_argument("--exp_name", default="exp0")
    return p.parse_args()


def build_env_kwargs(stock_dimension: int) -> dict:
    state_space = 1 + 2 * stock_dimension + len(TECH_INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    return {
        "hmax": 100,
        "initial_amount": 1000000,  # contest 규칙: 초기 자본 $1M (변경 금지)
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": TECH_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }


def main():
    args = parse_args()
    from finrl.config import TRAINED_MODEL_DIR as _BASE_TRAINED, RESULTS_DIR as _BASE_RESULTS

    trained_model_dir = os.path.join(EXPERIMENTS_DIR, args.exp_name, _BASE_TRAINED)
    results_dir = os.path.join(EXPERIMENTS_DIR, args.exp_name, _BASE_RESULTS)
    check_and_make_directories([trained_model_dir, results_dir])

    processed_full = pd.read_csv(args.data_file)
    trade = data_split(processed_full, args.start_date, args.end_date)
    print(f"trade rows: {len(trade)}, timesteps: {trade.index.nunique()}")

    stock_dimension = len(trade.tic.unique())
    env_kwargs = build_env_kwargs(stock_dimension)

    # Environment
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

    # Load trained PPO and backtest
    agent = DRLAgent(env=e_trade_gym)
    agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    trained_ppo = PPO.load(os.path.join(trained_model_dir, "trained_ppo"))

    df_result_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo, environment=e_trade_gym
    )

    print("============== Backtest Results ===========")
    perf_stats_all = backtest_stats(account_value=df_result_ppo)
    print(perf_stats_all)

    # Plot
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    df_result_ppo.plot()
    plt.savefig(os.path.join(results_dir, "plot.png"))

    df_result_ppo.to_csv(os.path.join(results_dir, "results.csv"), index=False)
    print(f"saved -> {os.path.join(results_dir, 'results.csv')}")


if __name__ == "__main__":
    main()
