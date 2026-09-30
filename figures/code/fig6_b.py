"""
Fig 6-b. 시간대별 잔차 중앙값 + 사분위 밴드 (모델별).

잔차 정의: prediction - actual  (양수 = 과대예측)

BAND 스위치
    "iqr" (기본) 중앙선 = 중앙값, 밴드 = Q1~Q3 (가운데 50%)
                 잔차 분포가 일출·일몰 경계에서 두껍게 비대칭이고 극단값도 많아
                 표준편차 밴드는 위아래로 과장되고 0 을 크게 넘어갑니다.
                 사분위 밴드는 극단값에 끌려가지 않고 비대칭도 그대로 보여줍니다.
    "std"        중앙선 = 평균, 밴드 = 평균 ± 표준편차 (이전 방식)

APPLY_OPERATION_HOURS = False (기본)
    가조시간 필터를 끄고 24시간 전 구간을 그립니다. 채점(성능지표)에는
    OPERATION_HOURS 필터가 필요하지만, 잔차의 시간대별 구조를 보는 이 그림에서는
    일출·일몰 경계와 야간 구간까지 봐야 모델 거동이 온전히 드러납니다.
    True 로 두면 fig4 와 동일한 채점 구간만 그립니다.

주의: Site 7 은 필터와 무관하게 데이터 이상 구간
      (2020-12-14 08:00 ~ 12-31 23:45) 이 loader 에서 항상 제외됩니다.
"""
import os
import matplotlib.pyplot as plt

import config
import loader

# 폭: 이 그림은 원고에서 **두 개가 가로로 나란히** 들어가므로 config.HALF_W
# (= 원고 \linewidth 의 절반)로 저장합니다. width=0.5\linewidth 로 넣으면 배율 1.0 이라
# config 의 pt 가 그대로 인쇄됩니다.
config.figure_config()

# ── 기준선 스위치 opt-out ─────────────────────────────────────────────────────
# 이 그림은 **ExoTFT-FB** 와의 비교가 목적이므로 config.USE_ROLLING_AS_BASE 를
# 적용받지 않습니다. load_site 에 order 를 명시해 넘기면 치환이 일어나지 않습니다.
LOAD_ORDER = config.MODEL_ORDER

# (사이트 목록, 파일 접미사) — 본문용 대표 2개와 부록용 전체 사이트.
# 본문용(6_a|6_b)도 부록용(6_a_all|6_b_all)도 나란히 두 장씩 들어가므로
# 폭은 모두 config.HALF_W.
SITE_SETS = [([5, 7], ""), (config.sites, "_all")]

BAND = "iqr"       # "iqr" = 중앙값 + Q1~Q3  /  "std" = 평균 ± 표준편차

# 중앙선·밴드의 정의(median, Q1–Q3)는 본문·캡션에서 서술하므로 축에는 단위만 둡니다.
YLABEL = "Error [MW]"


def _center_band(series_grouped):
    """(중앙선, 밴드 하한, 밴드 상한) 반환."""
    if BAND == "std":
        agg = series_grouped.agg(["mean", "std"]).sort_index()
        return agg["mean"], agg["mean"] - agg["std"], agg["mean"] + agg["std"]
    agg = series_grouped.quantile([0.25, 0.5, 0.75]).unstack().sort_index()
    return agg[0.5], agg[0.25], agg[0.75]


def draw(sites, suffix=""):
    fig, axes, bottom_axes = config.site_panel_axes(len(sites),
                                                    width=config.HALF_W)

    for ax, site in zip(axes, sites):
        df = loader.load_site(site, order=LOAD_ORDER, verbose=False).copy()
        df["hour_min"] = df["Time"].dt.strftime("%H:%M")

        for spec in loader.available_specs(df):
            err = spec.pred + "_err"
            df[err] = df[spec.pred] - df["True_Target"]     # 과대예측 = +
            center, lo, hi = _center_band(df.groupby("hour_min")[err])
            agg = center                                    # x 축 라벨용 index 보관
            x = range(len(center))
            ax.plot(x, center, color=spec.color, label=spec.label,
                    linewidth=1.0, linestyle=spec.linestyle, zorder=spec.zorder)
            ax.fill_between(x, lo, hi, color=spec.color, alpha=0.25,
                            edgecolor="none", zorder=spec.zorder - 1)

        ax.axhline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_ylabel(YLABEL, fontsize=config.PT_LABEL)
        config.set_nature_ticks(ax)

        if ax in bottom_axes:
            idx = list(agg.index)
            pos, lab, seen = [], [], set()
            for k, s in enumerate(idx):
                h = int(s[:2])
                if h % 2 == 0 and h not in seen:
                    pos.append(k); lab.append(str(h)); seen.add(h)
            ax.set_xticks(pos)
            ax.set_xticklabels(lab, ha="center")
            ax.set_xlabel("Hour")

    # 패널마다 y 눈금 자릿수가 달라 y라벨의 가로 위치가 어긋납니다.
    # align_ylabels 로 가장 바깥 위치에 모두 맞춥니다.
    fig.align_ylabels(axes)

    config.label_panels(axes, labels=[f"Site {s}" for s in sites])

    # 범례가 차지할 높이만큼만 아래를 비웁니다(항목 2개 → 1줄).
    legend_in = 0.22 + 0.08
    plt.tight_layout(pad=0.25, rect=[0, legend_in / fig.get_figheight(), 1, 1],
                     h_pad=0.5)
    # 범례는 레이아웃 확정 후, 축 박스 기준 가운데에 배치
    config.legend_below_axes(fig, axes, ncol=2, handlelength=1.1,
                             handletextpad=0.35, columnspacing=0.9)
    out = f"6_b{suffix}.png"
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, out), dpi=300)
    plt.close(fig)
    print(f"saved: {out}")


def main():
    for sites, suffix in SITE_SETS:
        draw(sites, suffix)


if __name__ == "__main__":
    main()
