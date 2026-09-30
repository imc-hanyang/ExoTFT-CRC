# figures/code — figure 생성 코드

새 데이터셋(`dataset/results`, `dataset/origin`) 기준으로 재작성했습니다.
`ref/` 는 예전 데이터 기준 원본 코드이며 참고용으로만 남겨둡니다.

## 구조

| 파일 | 역할 |
|---|---|
| `config.py` | 스타일(rcParams, 색), 경로, 가조시간, **모델 레지스트리(MODELS)** |
| `loader.py` | 사이트별 결과 병합 + 잔차 파생 변수 + 성능지표 |
| `fig*.py` | 개별 figure |
| `run_all.py` | 전체 일괄 실행 |

```bash
cd figures/code
python run_all.py          # 전부
python fig4.py             # 하나만
```

## 모델 5개 (config.MODELS)

| key | label | 집계표(fig4) | 시계열 결과 CSV |
|---|---|---|---|
| `dynamic` | Dynamic | ✅ `dynamic` | ✗ |
| `chen` | Chen et al. | ✅ `chen` | ✗ |
| `kong` | Kong et al. | ✅ `kong` | ✗ |
| `base` | Base | ✅ `base` | ✅ `site{site}_base_fullbatch_test_results.csv` |
| `reco` | R-ECO | ✅ `잔차보정` | ✅ `site{site}_residual_adaptive_results.csv` |

- **fig4** 는 `dataset/results/model_total_results.csv` 집계표를 그대로 씁니다 → 5개 모델 전부
- **fig6/7/8, fig_alpha** 는 15분 단위 시계열이 필요하므로 `base` / `reco` 2개만 그립니다.
  dynamic/chen/kong 의 시계열 예측 CSV가 생기면 해당 슬롯에 `file` / `pred_col` 두 줄만
  채우면 나머지 그림에도 자동 반영됩니다 (`file=None` 인 모델은 조용히 건너뜀).

시계열 CSV 요구사항: `Time` 컬럼 + 예측값 컬럼 1개. `True_Target` 은 base 파일 기준으로
통일하므로 없어도 됩니다. 그리는 순서는 `MODEL_ORDER` 로 조절합니다.

### 집계표 단위

`model_total_results.csv` 의 nmae/nrmse 는 전부 [%] 단위로 통일되어 있어
모든 모델의 `agg_scale = 1.0`, `AGG_CELL_OVERRIDES = {}` 입니다.
나중에 특정 칸만 단위가 다르면 `config.AGG_CELL_OVERRIDES` 에
`{("Site 1", "chen"): 100.0}` 식으로 배수를 넣으면 됩니다.

## 공통 전처리 (loader.load_site)

- base + 각 모델 결과를 `Time` inner join, 원본 기상 피처 병합
- Site 7 의 `2020-12-14 08:00 ~ 2020-12-31 23:45` 구간 제외
- `OPERATION_HOURS` 가조시간 필터 (threshold 필터는 쓰지 않음)
- 모델별 `pred_<key>` / `resid_<key>` / `abs_resid_<key>` 생성
- R-ECO: `Rolling_Base_Pred`, `Alpha_Used`, `Correction(=α·ê)`, `Improvement`, `Improved`

## 생성되는 figure

| 스크립트 | 산출물 | 내용 |
|---|---|---|
| `fig3_a.py` | `3_a.png` | 사이트별 시간대 평균 발전량 |
| `fig3_b.py` | `3_b.png` | 사이트별 발전 시각 분포 (boxplot) |
| `fig4.py` | `4.png`, `4_metrics.csv` | 사이트 × 5개 모델 NMAE / NRMSE / R² (집계표 기반, 오차는 로그축) |
| `fig_alpha.py` | `5_a.png`, `5_b.png`, `5_c.png` | Adaptive α 시계열 / α vs 개선량 / 시간대별 α |
| `fig6_a.py` | `6_a.png` | 특정 하루 실측 vs 모델별 예측 |
| `fig6_b.py` | `6_b.png` | 시간대별 잔차 mean ± std 밴드 |
| `fig7.py` | `7.png` | 사이트별 상대오차 분포 (3D ridge, Base·R-ECO 오버레이) |
| `fig8_a.py` | `8_a.png` | 실측 vs 예측 산점도 + y=x |
| `fig8_b.py` | `8_b_{key}.png`, `8_b_improvement.png` | 날짜 × 시각 잔차 히트맵 / Base 대비 개선량 |
| `fig8_c.py` | `8_c.png` | residual_{t+1} 구간별 주요 변수 정규화 평균 |
| `fig9.py` | `9.png` | IG 피처 중요도의 모델 간 차이 히트맵 (Δ = ExoTFT-CRC − ExoTFT-FB, %p) |
| `fig9_a.py` | `9_a.png` | ExoTFT-CRC 의 IG 피처 중요도 히트맵 (ratio %, 로그 색축) |
| `fig9_b.py` | `9_b.png` | encoder(입력군) 단위 Δ IG 막대 — 사이트 7개 × Endogenous/Weather/Temporal |

## 스크립트 상단 조절 포인트

- `fig6_a.py` : `SITES`, `SELECTED_DATE` (기본 `2020-12-24`)
- `fig6_b.py`, `fig8_a.py`, `fig8_b.py`, `fig8_c.py` : `SITES`
- `fig8_a.py` : `SCATTER_MODELS` (모델 5개면 겹치므로 2~3개로 제한 권장)
- `fig4.py` : `LOG_ERROR_AXIS` (오차 로그축 on/off), `ANNOTATE` (막대 위 값 표기)
- `fig7.py` : `XLIM` (상대오차 범위), `RIDGE_MODELS` (겹쳐 그릴 모델, 2~3개 권장)

## 대기 중인 항목

1. (선택) dynamic / chen / kong 의 15분 단위 예측 CSV → fig6/7/8 에도 반영 가능
