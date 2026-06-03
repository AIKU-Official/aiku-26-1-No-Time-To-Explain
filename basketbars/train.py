"""Train a FinRL PPO agent on a basket-bar dataset.

Reads a basket-bar CSV produced by make_dataset.py, splits it by the `date`
column (each basket bar = one env timestep), and trains the FinRL PPO agent.

Example
-------
python basketbars/train.py \
    --data_file data/basket_bars.csv \
    --start_date 2014-01-06 --end_date 2020-07-31 \
    --exp_name exp0 --total_timesteps 80000
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from stable_baselines3.common.logger import configure

from config import (
    DEFAULT_DATA_FILE,
    EXPERIMENTS_DIR,
    TECH_INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
)

# PPO configs (contest-fixed model; only hyperparameters are tunable)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}


def parse_args():
    p = argparse.ArgumentParser(description="basket-bar PPO 학습")
    p.add_argument("--data_file", default=DEFAULT_DATA_FILE, help="basket-bar CSV 경로")
    p.add_argument("--start_date", default=TRAIN_START_DATE, help="학습 시작 (YYYY-MM-DD)")
    p.add_argument("--end_date", default=TRAIN_END_DATE, help="학습 종료 (YYYY-MM-DD)")
    p.add_argument("--exp_name", default="exp0")
    p.add_argument("--total_timesteps", type=int, default=80000)
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
    train = data_split(processed_full, args.start_date, args.end_date)
    print(f"train rows: {len(train)}, timesteps: {train.index.nunique()}")

    stock_dimension = len(train.tic.unique())
    env_kwargs = build_env_kwargs(stock_dimension)

    # Environment
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    # PPO agent
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    # logger
    tmp_path = os.path.join(results_dir, "ppo")
    # "stdout"을 넣어야 콘솔에 학습 테이블(rollout/ep_rew_mean, fps 등)이 찍힌다.
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

    trained_ppo = agent.train_model(
        model=model_ppo,
        tb_log_name="ppo",
        total_timesteps=args.total_timesteps,
    )
    trained_ppo.save(os.path.join(trained_model_dir, "trained_ppo"))
    print(f"saved -> {os.path.join(trained_model_dir, 'trained_ppo')}")


if __name__ == "__main__":
    main()
