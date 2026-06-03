# BasketBars — Dollar Basket Bars for FinRL Stock Trading

1분봉(1-minute) 데이터로 **dollar-based basket bars**를 만들어 FinRL PPO 에이전트를
학습/백테스트하는 파이프라인입니다. ICAIF 2023 FinRL Contest **Task 1 (Data-Centric
Stock Trading)** 에 적용하는 것을 목표로 합니다.

아이디어: 캘린더 일봉 대신, 바스켓 전체의 누적 거래대금이 임계값에 도달할 때마다
하나의 bar를 끊습니다(정보 기반 샘플링). 각 basket bar는 곧 RL 환경의 한 timestep이
되며, 표준 기술지표에 더해 마이크로구조 피처(거래 집중도, 단면 변동성, 체결 불균형,
시간/요일 임베딩 등)를 담습니다.

## 폴더 구조

```
BasketBars/
├── basketbars/              # 이 프로젝트의 코드 (패키지)
│   ├── config.py           # 경로 / 날짜 분할 / 피처 리스트
│   ├── config_tickers.py   # DJIA30 티커
│   ├── downloader.py       # AlpacaDownloader: 1분봉 fetch + basket bar 생성
│   ├── make_dataset.py     # 1분봉 → basket bar CSV 생성 (CLI)
│   ├── train.py            # basket bar CSV로 PPO 학습
│   └── test.py             # 백테스트 → results.csv / plot.png
├── data/                   # 생성된 basket bar CSV (gitignore)
│   └── raw_cache/          # 월별 raw 1분봉 청크 캐시 (이어받기용)
├── experiments/            # exp별 trained_models / results (자동 생성)
├── .env                    # ALPACA_KEY / ALPACA_SECRET
├── requirements.txt
├── README.md
└── contest2023/            # 대회 원본 (참고용, 미수정)
```

---

## 실행 방법 (처음부터 끝까지)

> 모든 명령은 **프로젝트 루트**(`BasketBars/`)에서 실행합니다.

### 0) 환경 준비

Python 3.10 환경에서 필요한 패키지를 설치합니다.

```bash
pip install swig
pip install box2d
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
pip install -r requirements.txt
```

그리고 프로젝트 루트에 `.env` 파일을 만들고 Alpaca 키를 넣습니다 (데이터 다운로드
단계에서만 사용).

```
ALPACA_KEY=your_key
ALPACA_SECRET=your_secret
```

### 1) 데이터셋 생성 — 1분봉 → basket bar CSV

```bash
python basketbars/make_dataset.py \
    --start_date 2025-01-01 --end_date 2026-05-31 \
    --num_bars_per_day 10 \
    --chunk_months 1 \
    --out data/basket_bars.csv
```

- 종목 미지정 시 **DJIA30 전체**를 받습니다. 특정 종목만 받으려면
  `--tickers AAPL AMZN MSFT` 처럼 지정합니다.
- 1분봉은 **월 단위 청크**로 받으며 tqdm 진행바(`fetch 1-min`)로 진행률이 보입니다.
- 각 청크는 `data/raw_cache/`에 캐시되어, **중간에 끊겨도 같은 명령을 다시 실행하면
  이미 받은 청크는 건너뛰고 이어받습니다.**
- 완료되면 `data/basket_bars.csv`가 생성되고, shape / 종목 수 / bar(=timestep) 수 /
  결측치 개수가 출력됩니다.

옵션 정리:

| 옵션 | 설명 |
| --- | --- |
| `--chunk_months N` | 청크 크기 (기본 1개월) |
| `--num_bars_per_day N` | 하루 평균 목표 bar 개수 (threshold 산출, 기본 10) |
| `--tickers ...` | 종목 리스트 (기본 DJIA30) |
| `--cache_dir PATH` | 캐시 폴더 (기본 `data/raw_cache`) |
| `--no_cache` | 캐시 무시하고 항상 새로 받기 |
| `--out PATH` | 출력 CSV 경로 (기본 `data/basket_bars.csv`) |

> ⚠️ **train/test를 한 CSV로 함께 만드세요.** `threshold`는 기간 전체 일평균
> 거래대금으로 정해지고 `bar_id`는 전역 cumsum이라, basket bar를 구간별로 따로
> 만들어 concat하면 bar 경계가 어긋납니다. raw 1분봉은 청크로 받아도 되지만 basket
> bar 생성은 합쳐진 전체 raw에 대해 한 번만 수행됩니다(스크립트가 처리).

### 2) 학습 — PPO

같은 CSV를 날짜로 잘라 학습 구간을 지정합니다.

```bash
python basketbars/train.py \
    --data_file data/basket_bars.csv \
    --start_date 2025-01-01 --end_date 2026-02-28 \
    --exp_name exp0 \
    --total_timesteps 80000
```

- 콘솔에 학습 테이블(`rollout/ep_rew_mean`, `time/fps` 등)이 `n_steps`(2048)마다
  찍힙니다. 진행 로그는 `experiments/exp0/results/ppo/progress.csv`에도 기록됩니다.
- 모델은 `experiments/exp0/trained_models/trained_ppo.zip`에 저장됩니다.

> 💡 파이프라인이 끝까지(모델 저장까지) 도는지 먼저 확인하려면
> `--total_timesteps 5000`처럼 작게 줘서 빠르게 한 번 돌려보세요.

### 3) 백테스트 — results.csv / plot.png

학습 구간과 겹치지 않는 평가 구간을 지정합니다.

```bash
python basketbars/test.py \
    --start_date 2026-03-01 --end_date 2026-05-31 \
    --data_file data/basket_bars.csv \
    --exp_name exp0
```

- `--exp_name`은 학습 때와 같은 값을 줘야 해당 모델을 불러옵니다.
- 결과: `experiments/exp0/results/results.csv`(timestep별 계좌 가치)와
  `plot.png`가 생성되고, 콘솔에 백테스트 지표(누적 수익률, Sharpe, MDD 등)가
  출력됩니다.

---

## 데이터 / 모델 설계

- **상태공간**: `1 + 2·stock_dim + len(TECH_INDICATORS)·stock_dim`
  (잔고 + 가격·보유수량 + 피처). 피처는 `config.TECH_INDICATORS`.
- **행동공간**: 종목별 매수/매도/홀드.
- **보상**: 총자산 변화(`reward_scaling=1e-4`).
- **초기자본**: $1,000,000 (대회 규칙, 변경 금지).
- **모델**: FinRL PPO (대회 고정 모델, 하이퍼파라미터만 튜닝).

### 피처 구성 (`config.py`)

기본 지표(`INDICATORS`): macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30,
close_30_sma, close_60_sma.

basket bar 고유 피처(`CUSTOM_FEATURES`): duration, avg_tick_size, hhi, cs_vol,
agg_imbalance, time_sin/cos, day_sin/cos.

starter-kit baseline과 비교하려면 `config.py`에서
`TECH_INDICATORS = INDICATORS`로 바꿔 커스텀 피처를 빼면 됩니다.

## FinRL 호환 메모

`make_basket_bars`는 각 bar의 `end_time`을 문자열 `date` 컬럼으로 추가합니다. FinRL
`data_split`은 이 `date`로 필터링·factorize하므로 **각 basket bar = 환경의 한
timestep**이 되고, 한 bar 안의 모든 종목이 같은 `date`를 공유합니다. `bar_id`는
거래대금 스파이크로 값이 건너뛸 수 있어(비연속) 직접 index로 쓰면 안 되지만,
`date` factorize는 항상 0,1,2,… 연속 index를 만들어 안전합니다. (합성 데이터로 date
공유·결측치 0·timestep당 종목 수 일치·피처 존재를 검증 완료.)
