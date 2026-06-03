# custom_dataset

LOAM 학습에 사용되는 커스텀 데이터셋 폴더입니다.
Yahoo Finance에서 raw data를 다운로드하고 FinRL의 `FeatureEngineer`로 기술 지표를 추가하여 생성합니다.

---

## 파일 구성

| 파일 | 설명 |
| `config_tickers.py` | 종목 리스트 정의 |
| `make_dataset.py` | 데이터셋 생성 스크립트 |

---

## 데이터셋 생성

```bash
# DJIA 30 기본 데이터셋 생성 (기본값)
python contest2023/custom_dataset/make_dataset.py

# 종목·기간·저장 경로 커스텀
python contest2023/custom_dataset/make_dataset.py \
    --start_date 2010-07-01 \
    --end_date   2023-10-24 \
    --save_path  ./contest2023/custom_dataset \
    --file_name  djia30.csv
```

---

## 컬럼 설명

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `date` | string | 거래일 (YYYY-MM-DD) |
| `tic` | string | 종목 티커 |
| `close` | float | 종가 |
| `high` | float | 고가 |
| `low` | float | 저가 |
| `open` | float | 시가 |
| `volume` | float | 거래량 |
| `day` | float | 요일 (0=월 ~ 4=금) |
| `macd` | float | MACD |
| `boll_ub` | float | 볼린저 밴드 상단 |
| `boll_lb` | float | 볼린저 밴드 하단 |
| `rsi_30` | float | RSI (30일) |
| `cci_30` | float | CCI (30일) |
| `dx_30` | float | DX (30일, 추세 강도) |
| `close_30_sma` | float | 30일 단순이동평균 |
| `close_60_sma` | float | 60일 단순이동평균 |
| `vix` | float | VIX 공포지수 |
| `turbulence` | float | Turbulence 지수 |


---

## 종목 리스트

`config_tickers.py`에 두 종목 그룹이 정의되어 있습니다.

**`DOW_30_TICKER`** (DJIA 30, LOAM 기본): AXP, AMGN, AMZN, AAPL, BA, CAT, CSCO, CVX, GS, HD, HON, IBM, JNJ, KO, JPM, MCD, MMM, MRK, MSFT, NVDA, NKE, PG, TRV, UNH, CRM, VZ, V, WMT, DIS, SHW

**`PPO_SWITCH_TICKER`** (PPO-Switch 비교용): AFL, AMZN, AZO, C, CMCSA, COP, F, GE, GM, GOOGL, LLY, LOW, MDT, MSI, NOC, ORCL, PEP, PFE, QCOM, SBUX, SLB, SRE, T, TT, UPS, WFC, XOM, XRX, ZTS