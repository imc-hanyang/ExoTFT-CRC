"""
Fig 8-b. 날짜 × 시각 잔차 히트맵 — ExoTFT-FB / ExoTFT-CRC 2열 비교 (전체 사이트).

  (a)  ExoTFT-FB 잔차      (예측 − 실측)
  (b)  ExoTFT-CRC 잔차  (예측 − 실측)

행 = 사이트. 두 열은 같은 색 스케일(공유 TwoSlopeNorm)을 쓰고 컬러바도 하나만 둡니다.
개선량(|ExoTFT-FB| − |ExoTFT-CRC|) 열은 8_b2 와 표현을 맞추려고 뺐습니다.
사이트마다 설비용량이 다르므로 기본은 설비용량 대비 [%] 로 정규화합니다
(NORMALIZE_BY_CAPACITY = False 로 두면 [MW] 원 스케일).

테스트 구간이 약 2개월(2020-11-01 ~ 2020-12-31)이라 day-of-year 1~365 대신
실제 테스트 날짜 범위를 y축으로 씁니다. 15분 격자를 보간 없이 그대로 pivot 합니다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

import config
import loader

# ── 기준선 스위치 opt-out ─────────────────────────────────────────────────────
# 이 그림은 **ExoTFT-FB** 잔차와의 비교가 목적이므로 config.USE_ROLLING_AS_BASE
# 를 적용받지 않습니다. load_site 에 order 를 명시하면 base→rolling 치환이 없습니다.
LOAD_ORDER = config.MODEL_ORDER
BASE_KEY = config.FULLBATCH_BASE_KEY

SITES = config.sites
FIG_W = config.FULL_W           # 원고 표준 폭(= \linewidth 390 pt) — 모든 그림 공통
config.figure_config()
NORMALIZE_BY_CAPACITY = True
CLIP_PERCENTILE = 99          # 색 스케일을 극단값에 뺏기지 않도록 상한 분위수
CMAP = "seismic"

# 컬러바 배치 (인치 기준 — 사이트 수가 바뀌어도 여백이 따라 늘지 않도록)
CB_BOTTOM_IN = 1.15 * (1 + config.H_SCALE) / 2   # 서브플롯 영역의 아래 한계
CB_Y_IN = 0.42 * (1 + config.H_SCALE) / 2   # 컬러바 자체의 y 위치
CB_H_IN = 0.09                # 컬러바 높이
# 패널 라벨 (a)/(b) 와 x축 제목 사이 간격 [inch].
# 너무 붙으면 x축 제목의 둘째 줄처럼 보이므로 한 줄 띄운 느낌으로 두고,
# 대신 아래 컬러바를 더 내려(CB_Y_IN) 캡션이 패널 쪽에 속해 보이게 합니다.
PANEL_LABEL_GAP_IN = 0.20


def _pivot(df: pd.DataFrame, value: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "date": df["Time"].dt.normalize(),
        "tidx": df["Time"].dt.hour * 4 + df["Time"].dt.minute // 15,
        "v": np.asarray(value, dtype=float),
    })
    grid = tmp.pivot_table(index="date", columns="tidx", values="v", aggfunc="mean")
    return grid.reindex(columns=range(96))


def _norm(grids, clip=CLIP_PERCENTILE):
    vals = np.concatenate([g.values[~np.isnan(g.values)].ravel() for g in grids if g.size])
    vmax = float(np.percentile(np.abs(vals), clip)) if vals.size else 1.0
    vmax = vmax or 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def main():
    base, reco = config.MODELS[BASE_KEY], config.MODELS[config.RECO_KEY]

    # ── 사이트별 2종 격자 만들기 ─────────────────────────────────────────────
    rows = []
    for site in SITES:
        df = loader.load_site(site, order=LOAD_ORDER, verbose=False)
        if base.pred not in df.columns or reco.pred not in df.columns:
            continue
        scale = 100.0 / df.attrs["capacity"] if NORMALIZE_BY_CAPACITY else 1.0
        rows.append((
            site,
            _pivot(df, (df[base.pred] - df["True_Target"]) * scale),
            _pivot(df, (df[reco.pred] - df["True_Target"]) * scale),
        ))

    unit = r"% of $P_{\max}$" if NORMALIZE_BY_CAPACITY else "MW"
    # 두 열이 공유하는 단일 색 스케일
    norm_resid = _norm([g for r in rows for g in (r[1], r[2])])

    n = len(rows)
    # 패널 높이만 원고 폭 비율로 줄이고(H_SCALE), 글자 여백 1.95 in 는 절대값이라 유지
    fig_h = 0.85 * config.H_SCALE * n + 1.95 * (1 + config.H_SCALE) / 2
    fig, axes = plt.subplots(n, 2, figsize=(FIG_W, fig_h), sharex=True)
    axes = np.atleast_2d(axes)

    # 패널 라벨: 열(column)이 곧 패널이므로 (a)/(b) + 모델명을 붙입니다.
    # 위치는 그림 제목 자리가 아니라 x축 제목 아래(컬러바 위)입니다.
    panel_labels = [f"(a) {base.label} Residual", f"(b) {reco.label} Residual"]
    print(f"   panels: (a) {base.plain} residual, (b) {reco.plain} residual")
    mesh_resid = None

    for i, (site, g_base, g_reco) in enumerate(rows):
        y = mdates.date2num(g_base.index.to_pydatetime())
        for j, grid in enumerate((g_base, g_reco)):
            ax = axes[i, j]
            mesh_resid = ax.pcolormesh(np.arange(96), y, grid.values,
                                       cmap=CMAP, norm=norm_resid, shading="nearest")

            ax.yaxis_date()
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            # 사이트별 테스트 시작일이 달라도(예: Site 8 은 10-29 시작) 눈금이
            # 어긋나지 않도록 달력 기준(매월 1일·15일)으로 고정.
            # Site 7 은 이상구간 제외로 범위가 짧아 기존 방식을 유지.
            ax.yaxis.set_major_locator(
                mdates.DayLocator(interval=21) if site == 7
                else mdates.DayLocator(bymonthday=[1, 15]))
            if j == 0:
                ax.set_ylabel(f"Site {site}", fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
            config.set_nature_ticks(ax)

    for ax in axes[-1]:
        ax.set_xticks([0, 24, 48, 72, 95])
        ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:45"])
        ax.set_xlabel("Time of Day")

    plt.tight_layout(pad=0.25, rect=[0, CB_BOTTOM_IN / fig_h, 1, 1],
                     h_pad=0.35, w_pad=0.4)

    # 패널 라벨: 각 열 x축 제목 바로 아래(컬러바 위)에 열 가운데 정렬.
    # y 는 상수로 박지 않고 실제 x라벨 위치에서 재서 붙입니다 — 행 수가 달라도
    # 간격이 그대로 유지됩니다.
    fig.canvas.draw()          # x라벨의 실제 위치를 재려면 한 번 그려야 합니다
    renderer = fig.canvas.get_renderer()
    for j, lab in enumerate(panel_labels):
        pos = axes[-1, j].get_position()
        xlab_bottom = (axes[-1, j].xaxis.label.get_window_extent(renderer).y0
                       / fig.dpi)
        fig.text((pos.x0 + pos.x1) / 2,
                 (xlab_bottom - PANEL_LABEL_GAP_IN) / fig_h, lab,
                 ha="center", va="top", fontsize=config.PT_LABEL, fontweight="bold",
                 fontfamily="Helvetica")

    # ── 컬러바 1개 ───────────────────────────────────────────────────────────
    # 두 열이 같은 norm 을 쓰므로 축 전체 영역 가운데에 하나만 둡니다.
    # 축 실제 위치에서 계산하므로 패널 수·여백이 바뀌어도 따라갑니다.
    p0, p1 = (axes[-1, j].get_position() for j in range(2))
    w = (p1.x1 - p0.x0) * 0.70
    cax = fig.add_axes([(p0.x0 + p1.x1) / 2 - w / 2, CB_Y_IN / fig_h,
                        w, CB_H_IN / fig_h])

    cb = fig.colorbar(mesh_resid, cax=cax, orientation="horizontal")
    cb.set_label(f"Residual: Pred − Actual [{unit}]")
    config.thin_spines(cb.ax)

    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "8_b.png"), dpi=300)
    plt.close(fig)
    print("saved: 8_b.png")


if __name__ == "__main__":
    main()
