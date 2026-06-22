# BasketBars

📢 2026년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 팀원

| 팀원 | 역할 |
| ---- | ---- |
| [장국영](kukyoung830@korea.ac.kr) | 팀장, 아이디어 제시, 코드 작성 |
| [이재승](jseuyi@gamil.com) | 부팀장, 아이디어 디벨롭, financial domain expert |
| [양경학](wwyghww@gmail.com) | 팀원, 아이디어 디벨롭, 자료조사 |
| [김용주](sd081104@gmail.com) | 팀원, 아이디어 디벨롭, 자료조사 |

## 소개

1분봉(1-minute) 데이터로 **dollar-based basket bars**를 만들어 FinRL PPO 에이전트를 학습/백테스트하는 파이프라인입니다. ICAIF 2023 FinRL Contest **Task 1 (Data-Centric Stock Trading)** 에 적용하는 것을 목표로 합니다.

아이디어: 캘린더 일봉 대신, 바스켓 전체의 누적 거래대금이 임계값에 도달할 때마다 하나의 bar를 끊습니다(정보 기반 샘플링). 각 basket bar는 곧 RL 환경의 한 timestep이 되며, 표준 기술지표에 더해 마이크로구조 피처(거래 집중도, 단면 변동성, 체결 불균형, 시간/요일 임베딩 등)를 담습니다.

## 방법론

### Dollar Basket Bar

캘린더 일봉 대신, 바스켓 전체의 누적 거래대금(`vwap × volume`)이 threshold에 도달할 때마다 bar를 끊는 **정보 기반 샘플링** 방식을 사용합니다.

- **threshold**: 전체 기간 일평균 거래대금 ÷ `num_bars_per_day`
- **bar_id**: 거래대금 cumsum 기반 → 스파이크로 불연속 가능, FinRL 환경 index로는 `date` 컬럼(factorize) 사용
- 각 basket bar = RL 환경의 1 timestep, 같은 bar 내 모든 종목이 동일 `date` 공유

### 피처 구성

기본 지표(`INDICATORS`): macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma

basket bar 고유 피처(`CUSTOM_FEATURES`): duration, avg_tick_size, hhi, cs_vol, agg_imbalance, time_sin/cos, day_sin/cos

starter-kit baseline과 비교하려면 `config.py`에서 `TECH_INDICATORS = INDICATORS`로 변경해 커스텀 피처를 제거합니다.

### 모델 설계

- **상태공간**: `1 + 2·stock_dim + len(TECH_INDICATORS)·stock_dim`
- **행동공간**: 종목별 매수/매도/홀드
- **보상**: 총자산 변화 (`reward_scaling=1e-4`)
- **초기자본**: $1,000,000 (대회 규칙, 변경 금지)
- **모델**: FinRL PPO (대회 고정 모델, 하이퍼파라미터만 튜닝)

## 환경 설정

Python 3.10

```bash
pip install swig
pip install box2d
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일을 만들고 Alpaca 키를 넣습니다 (데이터 다운로드 단계에서만 사용).

```
ALPACA_KEY=your_key
ALPACA_SECRET=your_secret
```

## 사용 방법

> 모든 명령은 **프로젝트 루트**(`BasketBars/`)에서 실행합니다.

### 1) 데이터셋 생성 — 1분봉 → basket bar CSV

```bash
python basketbars/make_dataset.py \
    --start_date 2025-01-01 --end_date 2026-05-31 \
    --num_bars_per_day 10 \
    --chunk_months 1 \
    --out data/basket_bars.csv
```

- 종목 미지정 시 **DJIA30 전체**를 받습니다. 특정 종목만 받으려면 `--tickers AAPL AMZN MSFT` 처럼 지정합니다.
- 1분봉은 **월 단위 청크**로 받으며 tqdm 진행바로 진행률이 보입니다.
- 각 청크는 `data/raw_cache/`에 캐시되어, **중간에 끊겨도 같은 명령을 다시 실행하면 이미 받은 청크는 건너뛰고 이어받습니다.**

> ⚠️ **train/test를 한 CSV로 함께 만드세요.** threshold는 기간 전체 일평균 거래대금으로 정해지고 bar_id는 전역 cumsum이라, basket bar를 구간별로 따로 만들어 concat하면 bar 경계가 어긋납니다.

옵션 정리:

| 옵션 | 설명 |
| --- | --- |
| `--chunk_months N` | 청크 크기 (기본 1개월) |
| `--num_bars_per_day N` | 하루 평균 목표 bar 개수 (threshold 산출, 기본 10) |
| `--tickers ...` | 종목 리스트 (기본 DJIA30) |
| `--cache_dir PATH` | 캐시 폴더 (기본 `data/raw_cache`) |
| `--no_cache` | 캐시 무시하고 항상 새로 받기 |
| `--out PATH` | 출력 CSV 경로 (기본 `data/basket_bars.csv`) |

### 2) 학습 — PPO

```bash
python basketbars/train.py \
    --data_file data/basket_bars.csv \
    --start_date 2025-01-01 --end_date 2026-02-28 \
    --exp_name exp0 \
    --total_timesteps 80000
```

- 모델은 `experiments/exp0/trained_models/trained_ppo.zip`에 저장됩니다.
- 학습 로그는 `experiments/exp0/results/ppo/progress.csv`에 기록됩니다.

> 💡 파이프라인이 끝까지 도는지 먼저 확인하려면 `--total_timesteps 5000`처럼 작게 줘서 빠르게 한 번 돌려보세요.

### 3) 백테스트

```bash
python basketbars/test.py \
    --start_date 2026-03-01 --end_date 2026-05-31 \
    --data_file data/basket_bars.csv \
    --exp_name exp0
```

- `--exp_name`은 학습 때와 같은 값을 줘야 해당 모델을 불러옵니다.
- 결과: `experiments/exp0/results/results.csv`와 `plot.png`가 생성되고, 콘솔에 누적 수익률·Sharpe·MDD 등이 출력됩니다.

### 4) 학습 로그 시각화

```bash
# 한 실험
python basketbars/plot_train_log.py experiments/exp0 --out experiments/exp0/train_log.png

# 세 프로젝트 비교
python basketbars/plot_train_log.py \
    experiments/exp0 \
    contest2023/task-1-stock-trading-starter-kit/experiments/baseline \
    contest2023/SZU-FIN-621/Model/experiments/baseline/results/ppo_mean \
    --out compare.png --smooth 3
```

- 인자는 `progress.csv` 파일이거나 그것을 포함하는 폴더(하위까지 자동 탐색)면 됩니다.
- `--smooth N`: 이동평균 평활화 / `--metrics ...`: 그릴 지표 지정 / `--labels ...`: 범례 라벨 지정

### 5) 백테스트 결과 비교

```bash
# basketbars 실험 전체 비교
python basketbars/plot_backtest.py experiments

# 세 프로젝트 비교 (기간/granularity가 다르면 --x step 권장)
python basketbars/plot_backtest.py \
    experiments/exp0 \
    contest2023/task-1-stock-trading-starter-kit/experiments/baseline \
    contest2023/SZU-FIN-621/Model/experiments/baseline \
    --x step --out backtest_compare.png
```

- `--x date`(기본): 날짜축. 같은 기간을 비교할 때.
- `--x step`: 0~1 정규화 진행도축. 기간이나 bar 단위(일봉 vs basket bar)가 다른 run을 겹쳐 볼 때 권장.

## 예시 결과

학습 완료 후 `experiments/<exp_name>/` 아래에 아래 파일들이 생성됩니다.

- `results/ppo/progress.csv` — rollout/ep_rew_mean, value_loss 등 학습 곡선
- `results/results.csv` — timestep별 계좌 가치
- `results/plot.png` — 백테스트 포트폴리오 가치 그래프
