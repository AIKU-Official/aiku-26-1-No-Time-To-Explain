from __future__ import annotations

import os
import pytz
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

import config_tickers
from config import INDICATORS

class AlpacaDownloader:
    def __init__(self, start_date: datetime, end_date: datetime, ticker_list: list):
        self.start_date = start_date    # ex) datetime(yyyy, mm, dd) -> "yyyy-mm-dd 00:00:00"
        self.end_date = end_date        # ex) datetime(yyyy, mm, dd) -> "yyyy-mm-dd 00:00:00"
        self.ticker_list = ticker_list  # ex) ["AAPL", "MSFT", "GOOG"]

        load_dotenv()
        ALPACA_KEY = os.getenv("ALPACA_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_SECRET")
        self.client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        request_params = StockBarsRequest(
            symbol_or_symbols=self.ticker_list,
            timeframe=TimeFrame.Minute,
            start=self.start_date, # 시작 날짜
            end=self.end_date,   # 종료 날짜
            adjustment=Adjustment.ALL
            )
        
        print("데이터를 가져오는 중입니다")
        data_df = self.client.get_stock_bars(request_params).df
        print("가져오기 완료!")
        
        data_df = data_df.reset_index()
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df['timestamp'] = data_df['timestamp'].dt.tz_convert('US/Eastern')
        data_df = data_df.set_index('timestamp')

        # 본장(09:30 ~ 16:00) 데이터만 추출
        data_df = data_df.between_time('09:30', '16:00')
        data_df = data_df.reset_index()

        # synchronization
        unique_dates = pd.to_datetime(data_df['timestamp'].dt.date.unique())
        full_rth_indices = []
        for d in unique_dates:
            day_index = pd.date_range(
            start=f"{d} 09:30:00",
            end=f"{d} 16:00:00",
            freq='1min',
            tz='US/Eastern'
            )
            full_rth_indices.extend(day_index)
        full_rth_indices = pd.DatetimeIndex(full_rth_indices)
        data_df = data_df.pivot(index='timestamp', columns='symbol', values=['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'])
        data_df = data_df.reindex(full_rth_indices)

        # 1. 가격 및 기술지표는 직전 값으로 채우기 (Forward Fill)
        ffill_cols = ['open', 'high', 'low', 'close', 'vwap']
        for col in ffill_cols:
            if col in data_df.columns:
                data_df[col] = data_df[col].ffill()

        # 2. 거래량 및 체결 건수는 0으로 채우기
        zero_cols = ['volume', 'trade_count']
        for col in zero_cols:
            if col in data_df.columns:
                data_df[col] = data_df[col].fillna(0)

        if data_df.isnull().sum().sum() > 0:
            raise ValueError("데이터에 여전히 결측치가 존재합니다. 결측치 처리 로직을 확인해주세요.")

        data_df = data_df.stack(level='symbol', future_stack=True).reset_index()
        data_df.columns = ['timestamp', 'tic', 'close', 'high', 'low', 'open', 'trade_count', 'volume', 'vwap']
        data_df = data_df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)
        # print("Shape of DataFrame: ", data_df.shape)
        return data_df
    
    def make_basket_bars(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df = data_df.pivot(index='timestamp', columns='tic', values=['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'])
        dollar = data_df['vwap'] * data_df['volume']
        basket_dollar = dollar.sum(axis=1)
        threshold = self.threshold(basket_dollar, num_bars_per_day=10)

        cum_basket_dollar = basket_dollar.cumsum()
        bar_id = (cum_basket_dollar // threshold).astype(int)
        data_df['bar_id'] = bar_id
        data_df = data_df.set_index('bar_id', append=True)
        data_df = data_df.stack(level='tic', future_stack=True).reset_index()

        data_df['vol_price'] = data_df['vwap'] * data_df['volume']
        data_df['avg_tick_size'] = data_df['volume'] / data_df['trade_count']
        basket_bars = data_df.groupby(['bar_id', 'tic']).agg({
            'timestamp': ['min', 'max', 'count'],
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'trade_count': 'sum',
            'vol_price': 'sum',
            'avg_tick_size': 'mean'
        })
        basket_bars['vwap'] = basket_bars['vol_price'] / basket_bars['volume']
        basket_bars = basket_bars.drop(columns=['vol_price']).reset_index()

        new_columns = []
        for col in basket_bars.columns:
            if col == ('timestamp', 'min'): new_columns.append('start_time')
            elif col == ('timestamp', 'max'): new_columns.append('end_time')
            elif col == ('timestamp', 'count'): new_columns.append('duration') # count를 duration으로!
            elif isinstance(col, tuple): new_columns.append(col[0])
            else: new_columns.append(col)
        basket_bars.columns = new_columns

        # add features
        data_df['dollar_vol'] = data_df['close'] * data_df['volume']
        total_dollar_per_bar = data_df.groupby('bar_id')['dollar_vol'].transform('sum')
        data_df['participation_rate'] = data_df['dollar_vol'] / total_dollar_per_bar
        hhi = data_df.groupby('bar_id')['participation_rate'].apply(lambda x: np.sum(x**2))
        basket_bars['hhi'] = basket_bars['bar_id'].map(hhi)
        data_df['ret'] = (data_df['close'] / data_df['open']) - 1
        cs_vol = data_df.groupby('bar_id')['ret'].std()
        basket_bars['cs_vol'] = basket_bars['bar_id'].map(cs_vol)
        data_df['price_pressure'] = (data_df['close'] - data_df['open']) / (data_df['high'] - data_df['low'] + 1e-5)
        agg_imbalance = data_df.groupby('bar_id').apply(lambda x: np.sum(x['price_pressure'] * x['participation_rate']), include_groups=False)
        basket_bars['agg_imbalance'] = basket_bars['bar_id'].map(agg_imbalance)
        
        # time embedding
        basket_bars['time_sin'] = np.sin(2 * np.pi * (basket_bars.groupby('tic')['duration'].cumsum() % 390) / 390)
        basket_bars['time_cos'] = np.cos(2 * np.pi * (basket_bars.groupby('tic')['duration'].cumsum() % 390) / 390)
        # day embedding
        basket_bars['day_sin'] = np.sin(2 * np.pi * basket_bars['end_time'].dt.dayofweek / 5)
        basket_bars['day_cos'] = np.cos(2 * np.pi * basket_bars['end_time'].dt.dayofweek / 5)
        # add technical indicators
        basket_bars = self.add_technical_indicator(basket_bars)
        cols_to_fill = basket_bars.columns[basket_bars.isnull().any()]
        basket_bars[cols_to_fill] = basket_bars.groupby('tic')[cols_to_fill].transform(lambda x: x.ffill().bfill())
        return basket_bars
    
    def threshold(self, basket_dollar: pd.Series, num_bars_per_day: int=30) -> int:
        # 1. 날짜별로 그룹화하여 하루 총 거래 대금 계산
        daily_total_volume = basket_dollar.groupby(basket_dollar.index.date).sum()

        # 2. 전체 기간의 일평균 거래 대금 산출
        avg_daily_volume = daily_total_volume.mean()

        # 3. 목표 바 개수로 나누어 임계값 설정
        target_bars = num_bars_per_day
        threshold = avg_daily_volume // target_bars

        return threshold

    def add_technical_indicator(self, data):
        df = data.copy()
        stock = StockDataFrame.retype(df.copy())
        unique_ticker = stock.tic.unique()
        
        for indicator in INDICATORS:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["end_time"] = df[df.tic == unique_ticker[i]][
                        "end_time"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "end_time", indicator]], on=["tic", "end_time"], how="left"
            )
        df = df.sort_values(by=["end_time", "tic"])
        return df
    
if __name__ == "__main__":
    alpacadownloader = AlpacaDownloader(
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 31),
        ticker_list=['AAPL','AMZN'] # config_tickers.DJIA30_TICKER
    )

    data_df = alpacadownloader.fetch_data()
    basket_df = alpacadownloader.make_basket_bars(data_df)
    print(basket_df.head())
    print(basket_df.isnull().sum().sum())
    print(basket_df.shape)
    print(basket_df.columns)