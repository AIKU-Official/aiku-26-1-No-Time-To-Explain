import argparse
import os
import pandas as pd

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS
from finrl.config import *
import itertools
import config_tickers
from config import TRAIN_START_DATE, TRADE_END_DATE

def parse_args():
    p = argparse.ArgumentParser(description='LOAM dataset 생성 스크립트')
    p.add_argument('--start_date',  default=TRAIN_START_DATE)
    p.add_argument('--end_date',    default=TRADE_END_DATE)
    p.add_argument('--save_path',   default='./contest2023/custom_dataset')
    p.add_argument('--file_name',   default='train_data.csv')
    return p.parse_args()

def main():
    args = parse_args()
        
    print(f"데이터 다운로드 중... ({args.start_date} ~ {args.end_date})")
    df_raw = YahooDownloader(start_date = args.start_date,
                             end_date = args.end_date,
                             ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
    
    print("데이터 전처리 중...")
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = INDICATORS,
                         use_vix=True,
                         use_turbulence=True,
                         user_defined_feature = False)

    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    train = processed_full.fillna(0)
    # train = data_split(train, start=args.start_date, end=args.end_date)
    print(f"데이터셋 길이: {len(train)}")
    print(f"최종 데이터셋 크기: {train.shape}")
    print(f"ticker 개수: {train['tic'].nunique()}")
    print()
    print(f"데이터셋 head:\n{train.head()}")
    print()
    print(f"데이터셋 tail:\n{train.tail()}")
    file_path = os.path.join(args.save_path, args.file_name)
    
    # check_and_make_directories(args.save_path)
    train.to_csv(file_path)
    print(f"데이터셋이 '{file_path}'에 저장되었습니다.")

if __name__ == '__main__':
    main()