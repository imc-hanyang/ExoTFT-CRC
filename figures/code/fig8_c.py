"""
Fig 8-c. 계절(테스트 구간 진행) × 시각 잔차 히트맵 — ExoTFT-RF 단독, 7 × 1.

  가로 = 테스트 구간을 반달(half-month) 단위로 자른 구간
  세로 = 하루 중 시각(0~23시)
  칸   = 그 구간·그 시각의 평균 잔차 (예측 − 실측), 설비용량 대비 [%]

8_b 와 같은 잔차 정의·색 스케일을 쓰되, 모델은 **ExoTFT-RF(보정 전)** 하나만
싱글 칼럼(7행 × 1열, 사이트당 한 패널)으로 그립니다.

대상 구간은 PERIOD 로 지정합니다. 기본은 **2020-01-01 ~ 2020-10-31**(롤링 파인튜닝
구간)이라 가로축이 월 단위 10칸 = 계절 흐름이 됩니다.

    ⚠ 이 구간의 ExoTFT-RF 예측이 dataset/results 에 있어야 합니다.
      현재 들어 있는 site{N}_res_final_adaptive.csv 는 테스트 구간
      (2020-11-01 ~ 12-31)만 담고 있어, 그대로 실행하면 "구간 내 데이터 없음" 으로
      끝납니다. 롤링 구간 예측을 EXTRA_FILE 패턴의 CSV
      (컬럼: Time, True_Target, Rolling_Base_Pred)로 내려받아 두면 바로 그려집니다.

    PERIOD = None 으로 두면 로드된 전 구간(= 테스트 구간)을 씁니다.
    데이터가 없는 칸(사이트 7 의 12월 후반 등)은 회색으로 비웁니다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import config
import loader

# ── 기준선 스위치 opt-out ─────────────────────────────────────────────────────
# 이 그림은 ExoTFT-RF 자체를 보는 것이므로 config.USE_ROLLING_AS_BASE 와 무관하게
# 모델을 직접 지정합니다.
LOAD_ORDER = config.MODEL_ORDER
MODEL_KEY = "rolling"           # ExoTFT-RF (Rolling_Base_Pred)

SITES = config.sites
FIG_W = config.HALF_W           # 원고 \linewidth 의 절반 — 나란히 두 장 배치용
config.figure_config()

# 대상 구간 (양 끝 포함). None 이면 로드된 전 구간.
PERIOD = ("2020-01-01", "2020-10-31")
# 롤링 구간 예측 CSV. 있으면 이 파일을 우선 사용합니다(없으면 config.MODELS 의 기본
# 결과 파일 = 테스트 구간). 컬럼: Time, True_Target, Rolling_Base_Pred
EXTRA_FILE = "site{site}_rolling_2020_results.csv"

X_BINS = "month"                # "month" | "halfmonth" | "week"
NORMALIZE_BY_CAPACITY = True
CLIP_PERCENTILE = 99            # 색 스케일을 극단값에 뺏기지 않도록 상한 분위수
CMAP = "seismic"
NA_COLOR = "0.85"               # 데이터 없는 칸
# 구간 하나가 이 일수보다 적게 덮이면 축에서 뺍니다. (사이트 8 만 10-29~31 사흘치가
# 있어 "Oct 16–" 칸이 생기고, 나머지 6개 사이트는 통째로 회색이 됩니다.)
MIN_DAYS_PER_BIN = 5

# 레이아웃 [inch]
ROW_H = 0.78 * config.H_SCALE   # 패널(사이트) 하나의 높이 (원고 폭 비율 적용)
PAD_H = 1.45                    # x라벨 + 컬러바 여백
CB_BOTTOM_IN = 0.95             # 서브플롯 영역의 아래 한계
CB_Y_IN = 0.42                  # 컬러바 자체의 y 위치
CB_H_IN = 0.09                  # 컬러바 높이


def _bin_key(t: pd.Series) -> pd.Series:
    """시각 → (정렬용 시작일, 표시 라벨). 빈 Series 도 안전하게 처리."""
    t = pd.Series(pd.to_datetime(pd.Series(t, dtype="object")), dtype="datetime64[ns]")
    if X_BINS == "month":
        start = t.dt.to_period("M").dt.start_time
        label = start.dt.strftime("%b")
    elif X_BINS == "week":
        start = t.dt.to_period("W").dt.start_time
        label = start.dt.strftime("%m-%d")
    else:  # halfmonth
        first = t.dt.day <= 15
        start = t.dt.to_period("M").dt.start_time + pd.to_timedelta(
            np.where(first, 0, 15), unit="D")
        label = start.dt.strftime("%b") + np.where(first, " 1–15", " 16–")
    return start, label


def _grid(df: pd.DataFrame, value: pd.Series) -> pd.DataFrame:
    """(시각 × 구간) 평균 잔차 격자. columns = 구간 시작일(정렬 가능)."""
    start, _ = _bin_key(df["Time"])
    tmp = pd.DataFrame({
        "hour": df["Time"].dt.hour,
        "bin": start,
        "v": np.asarray(value, dtype=float),
    })
    grid = tmp.pivot_table(index="hour", columns="bin", values="v", aggfunc="mean")
    return grid.reindex(index=range(24))


def _norm(grids, clip=CLIP_PERCENTILE):
    vals = np.concatenate([g.values[~np.isnan(g.values)].ravel()
                           for g in grids if g.size])
    vmax = float(np.percentile(np.abs(vals), clip)) if vals.size else 1.0
    return TwoSlopeNorm(vmin=-(vmax or 1.0), vcenter=0.0, vmax=(vmax or 1.0))


def _load(site: int, spec) -> "pd.DataFrame | None":
    """대상 구간의 (Time, True_Target, 예측) 프레임. 없으면 None.

    EXTRA_FILE 이 있으면 그것을, 없으면 config.MODELS 의 기본 결과 파일을 씁니다.
    """
    extra = os.path.join(config.RESULTS_DIR, EXTRA_FILE.format(site=site))
    if os.path.exists(extra):
        df = pd.read_csv(extra)
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.rename(columns={spec.pred_col: spec.pred})
        df.attrs["capacity"] = loader.nominal_capacity(site)
    else:
        df = loader.load_site(site, order=LOAD_ORDER, verbose=False)

    if spec.pred not in df.columns:
        print(f"   [skip] site{site}: {spec.plain} 예측 컬럼 없음")
        return None

    if PERIOD is not None:
        lo, hi = pd.to_datetime(PERIOD[0]), pd.to_datetime(PERIOD[1]) + pd.Timedelta(days=1)
        n0 = len(df)
        df = df[(df["Time"] >= lo) & (df["Time"] < hi)]
        if df.empty:
            print(f"   [skip] site{site}: {PERIOD[0]} ~ {PERIOD[1]} 구간에 "
                  f"{spec.plain} 예측이 없습니다 (파일에는 {n0} 행이 있으나 구간 밖).")
            return None
    return df


def main():
    spec = config.MODELS[MODEL_KEY]

    rows, coverage = [], []
    for site in SITES:
        df = _load(site, spec)
        if df is None or df.empty:
            continue
        cap = df.attrs.get("capacity") or loader.nominal_capacity(site)
        scale = 100.0 / cap if NORMALIZE_BY_CAPACITY else 1.0
        rows.append((site, _grid(df, (df[spec.pred] - df["True_Target"]) * scale)))
        start, _ = _bin_key(df["Time"])
        coverage.append(df.groupby(start)["Time"].apply(
            lambda t: t.dt.normalize().nunique()).to_dict())

    if not rows:
        print(f"   [중단] {PERIOD[0]} ~ {PERIOD[1]} 구간의 {spec.plain} 예측이 없습니다.")
        print(f"          dataset/results/{EXTRA_FILE.format(site='{site}')} 형식으로")
        print( "          (Time, True_Target, Rolling_Base_Pred) 를 넣어 주세요.")
        return

    # 사이트마다 테스트 시작·종료가 달라도 칸이 어긋나지 않도록 구간 축을 공유합니다.
    bins = sorted({b for _, g in rows for b in g.columns})
    # 커버리지가 너무 얇은 구간은 제외 (버린 구간은 아래에서 출력합니다).
    days = {b: max(cov.get(b, 0) for cov in coverage) for b in bins}
    dropped = [b for b in bins if days[b] < MIN_DAYS_PER_BIN]
    bins = [b for b in bins if days[b] >= MIN_DAYS_PER_BIN]
    if not bins:
        have = ", ".join(f"site{s}" for s, _ in rows) or "없음"
        print(f"   [중단] {PERIOD[0]} ~ {PERIOD[1]} 구간에 {MIN_DAYS_PER_BIN}일 이상 덮는 "
              f"칸이 없습니다 (구간 안에 예측이 있는 사이트: {have}).")
        print(f"          dataset/results/{EXTRA_FILE.format(site='{site}')} 형식으로")
        print( "          (Time, True_Target, Rolling_Base_Pred) 를 넣어 주세요.")
        return
    _, labels = _bin_key(pd.Series(bins))
    grids = [g.reindex(columns=bins) for _, g in rows]

    unit = "% of Capacity" if NORMALIZE_BY_CAPACITY else "MW"
    norm = _norm(grids)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(NA_COLOR)

    n = len(rows)
    fig_h = ROW_H * n + PAD_H
    fig, axes = plt.subplots(n, 1, figsize=(FIG_W, fig_h), sharex=True)
    axes = np.atleast_1d(axes)

    mesh = None
    for ax, (site, _), grid in zip(axes, rows, grids):
        mesh = ax.pcolormesh(np.arange(len(bins) + 1), np.arange(25),
                             np.ma.masked_invalid(grid.values),
                             cmap=cmap, norm=norm, shading="flat")
        ax.set_ylim(0, 24)
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_ylabel(f"Site {site}", fontweight="bold")
        config.set_nature_ticks(ax)

    axes[-1].set_xticks(np.arange(len(bins)) + 0.5)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    axes[-1].set_xlabel("Test Period")
    # 시각 축 제목은 패널마다 반복되면 지저분해서 그림 전체에 한 번만 답니다.
    fig.supylabel("Hour of Day", x=0.005, fontsize=config.PT_LABEL)

    plt.tight_layout(pad=0.25, rect=[0, CB_BOTTOM_IN / fig_h, 1, 1], h_pad=0.25)

    # ── 컬러바 1개 ───────────────────────────────────────────────────────────
    p_last = axes[-1].get_position()
    w = p_last.width * 0.85
    cax = fig.add_axes([p_last.x0 + (p_last.width - w) / 2, CB_Y_IN / fig_h,
                        w, CB_H_IN / fig_h])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(f"{spec.plain} Residual: Pred − Actual [{unit}]")
    config.thin_spines(cb.ax)

    out = "8_c.png"
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, out), dpi=300)
    plt.close(fig)
    print(f"saved: {out}  ({X_BINS} bins: {list(labels)})")
    if dropped:
        _, dl = _bin_key(pd.Series(dropped))
        print(f"   dropped (< {MIN_DAYS_PER_BIN} days of data): {list(dl)}")


if __name__ == "__main__":
    main()
