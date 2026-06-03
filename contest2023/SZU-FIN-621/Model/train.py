"""
@author: Zhong Anyang, Chao Kaiyin, and Chen Geying from Shenzhen University
@supervisor: Yin Jianfei and Joshua Zhexue Huang
@describe: This software serves as our submission for the 4th ACM ICAIF 2023 FinRL Contest.
@date: 2023-11-12
"""
import pandas as pd
import os
import argparse
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS

# Contestants are welcome to split the data in their own way for model tuning
TRAIN_START_DATE = '2026-01-01'
TRAIN_END_DATE = '2026-04-31'

def parse_args():
    p = argparse.ArgumentParser(description='PPO-Switch 학습')
    p.add_argument('--data_name', default='ema')
    p.add_argument('--exp_name', default='exp0')
    return p.parse_args()

args = parse_args()
########################################################################################################################
model_name = f'ppo_{args.data_name}'
processed_full = pd.read_csv(f'./contest2023/custom_dataset/ppo_switch_dataset/{args.data_name}_train_data.csv')
########################################################################################################################

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)

# Environment configs
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
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
    trained_model_dir = os.path.join('./contest2023/SZU-FIN-621/Model/experiments', args.exp_name, _BASE_TRAINED)
    results_dir       = os.path.join('./contest2023/SZU-FIN-621/Model/experiments', args.exp_name, _BASE_RESULTS)
    check_and_make_directories([trained_model_dir, results_dir])

    # Environment
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    # PPO agent
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    # set up logger
    tmp_path = results_dir + '/' + model_name
    new_logger_ppo = configure(tmp_path, ["csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=80000)

    trained_ppo.save(trained_model_dir + '/' + model_name)
