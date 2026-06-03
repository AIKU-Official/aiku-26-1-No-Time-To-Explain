import pandas as pd
import os
import sys

# contest2023/ 를 import 경로에 추가해 `custom_dataset` 패키지를 찾게 한다
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS
from finrl.config import INDICATORS
from custom_dataset.config import TRAIN_START_DATE, TRAIN_END_DATE
# Contestants are welcome to split the data in their own way for model tuning


def parse_args():
    p = argparse.ArgumentParser(description='starter kit 학습')
    p.add_argument('--exp_name', default='exp0')
    return p.parse_args()

args = parse_args()

processed_full = pd.read_csv('./contest2023/custom_dataset/train_data.csv')
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)

# Environment configs
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}


if __name__ == '__main__':
    from finrl.config import TRAINED_MODEL_DIR as _BASE_TRAINED, RESULTS_DIR as _BASE_RESULTS
    trained_model_dir = os.path.join('./contest2023/task-1-stock-trading-starter-kit/experiments', args.exp_name, _BASE_TRAINED)
    results_dir       = os.path.join('./contest2023/task-1-stock-trading-starter-kit/experiments', args.exp_name, _BASE_RESULTS)

    check_and_make_directories([trained_model_dir])

    # Environment
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    # PPO agent
    agent = DRLAgent(env = env_train)
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

    # set up logger
    tmp_path = results_dir + '/ppo'
    new_logger_ppo = configure(tmp_path, ["csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

    trained_ppo = agent.train_model(model=model_ppo,
                                tb_log_name='ppo',
                                total_timesteps=80000)
    
    trained_ppo.save(trained_model_dir + '/trained_ppo')
