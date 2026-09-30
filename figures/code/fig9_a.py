"""
Fig 9-a. ExoTFT-CRC 의 Integrated Gradients(IG) 피처 중요도 히트맵 (단일 패널).

  값 = 그 사이트에서 해당 피처가 차지한 attribution 비중 ratio [%] (사이트별 합 100%)
  행 = 사이트, 열 = 피처.

fig9(= 9.png, Δ = CRC − FB)와 짝이 되는 그림입니다. 데이터만 **ExoTFT-CRC 한 모델의
값 자체**로 바꿨고, 색맵·셀 크기·라벨 회전·컬러바 배치 등 표현은 9.png 를 그대로
따릅니다(같은 rank_rb 반전 색맵, CLIP_PERCENTILE, CELL_H, constrained layout,
shrink/pad/aspect).

다만 이 그림의 값은 비중이라 음수가 없고(0 ~ 51%), 범위가 100배 넘게 벌어져 있어
색 범위만 **로그 스케일**로 둡니다 — 낮음(파랑) → 높음(빨강). 선형으로 두면
상위 4개 열만 색이 살고 나머지 18개 열이 한 덩어리로 뭉칩니다.
기여도가 정확히 0.00% 인 칸은 로그로 표현할 수 없어 회색(NA_COLOR)으로 비웁니다.

FB 에 없는 잔차 입력(Raw_Error, IMF_1~4)도 CRC 에는 값이 있어 22개 피처를 모두
남깁니다. 이 피처들의 기여도는 0.01% 이하라 색으로는 0 과 구분되지 않는데, 그 자체가
"CRC 전용 입력이 feature 단위에서는 거의 기여하지 않는다" 는 결과입니다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import config

# 폭은 원고 표준(config.FULL_W = 원고 \linewidth = 390 pt)으로 고정 — 배율 1.0 이라 config 의
# pt 가 그대로 인쇄됩니다.
config.figure_config()

IG_DIR = os.path.join(config.DATA_ROOT, "ig")
IG_FILE = "IG_residual_feature.csv"     # ExoTFT-CRC(잔차 모델) feature 단위 IG

# 열 정렬 (왼 → 오른쪽):
#   "ratio"     사이트 평균 기여도가 큰 순 (기본, 9.png 의 "abs_delta" 에 대응)
#   "max_ratio" 사이트별 기여도의 **최대값** 기준. 한 사이트에서만 큰 피처도 앞으로.
#   "mean_rank" 모델이 매긴 평균 IG 랭크 순
#   "group"     group(Endogenous/Residual/Weather/Temporal) 별로 묶고 그 안에서 랭크 순
COL_ORDER = "ratio"
GROUP_SEQ = ["Endogenous", "Residual", "Weather", "Temporal"]

ANNOTATE = False        # True 면 셀에 ratio 값(%) 표기
CMAP_NAME = "rank_rb"   # config 커스텀: 빨강 → 파랑. 9.png 처럼 반전해서 씀
NA_COLOR = "0.80"       # 값이 없는 셀 색 (이 그림에서는 발생하지 않음)

CLIP_PERCENTILE = 100   # 색 스케일 상한 분위수(100 = 최대값 그대로)
LOG_VMIN = 0.01         # 로그 스케일 하한 [%] — CSV 의 최소 표기 단위
CELL_H = 0.45 * config.H_SCALE   # 히트맵 셀 한 칸(사이트 1개)의 높이 [inch]


def load_ig(fname: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(IG_DIR, fname), encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["site_num"] = df["site"].astype(str).str.extract(r"(\d+)").astype(int)
    df["rank"] = pd.to_numeric(df["rank"])
    if "ratio" in df.columns:      # "45.40%" → 45.40
        df["ratio"] = pd.to_numeric(df["ratio"].astype(str).str.rstrip("%"),
                                    errors="coerce")
    return df


def main():
    df = load_ig(IG_FILE)

    sites = sorted(df["site_num"].unique())
    mean_rank = df.groupby("feature")["rank"].mean()
    group_of = df.drop_duplicates("feature").set_index("feature")["group"]

    ratio = df.pivot(index="feature", columns="site_num", values="ratio") \
              .reindex(columns=sites)

    # ── 열(피처) 순서 결정 ───────────────────────────────────────────────────
    if COL_ORDER in ("ratio", "max_ratio"):
        mag = ratio.max(axis=1) if COL_ORDER == "max_ratio" else ratio.mean(axis=1)
        key = pd.DataFrame({"m": mag, "r": mean_rank.reindex(mag.index)})
        feature_order = key.sort_values(["m", "r"],
                                        ascending=[False, True]).index.tolist()
    elif COL_ORDER == "group":
        key = pd.DataFrame({"g": group_of.reindex(ratio.index),
                            "r": mean_rank.reindex(ratio.index)})
        key["gi"] = key["g"].map({g: i for i, g in enumerate(GROUP_SEQ)}).fillna(99)
        feature_order = key.sort_values(["gi", "r"]).index.tolist()
    else:
        feature_order = mean_rank.reindex(ratio.index).sort_values().index.tolist()

    ratio = ratio.reindex(index=feature_order)

    vals = np.abs(ratio.values[~np.isnan(ratio.values)])
    dmax = float(np.percentile(vals, CLIP_PERCENTILE)) if vals.size else 1.0
    dmax = dmax or 1.0

    cmap = config.get_cmap(CMAP_NAME).reversed()   # 파랑(낮음) → 빨강(높음)
    cmap.set_bad(NA_COLOR)
    norm = LogNorm(vmin=LOG_VMIN, vmax=dmax)
    ratio = ratio.mask(ratio <= 0)                 # 0.00% 는 로그 표현 불가 → 회색

    # ── 그리기 (가로 방향: x = 피처, y = 사이트) ────────────────────────────
    #   셀 높이(CELL_H)만 고정하고 폭은 남는 공간을 채우게 둡니다. 9.png 와 동일.
    n_f, n_s = len(feature_order), len(sites)
    fig_h = CELL_H * n_s + 1.75              # + 회전 라벨 + 컬러바 여백
    fig, ax = plt.subplots(figsize=(config.FULL_W, fig_h), layout="constrained")
    # 회전된 피처 라벨 때문에 기본 여백이 크게 잡혀 컬러바가 멀어집니다.
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.0, wspace=0.0)

    mesh = ax.pcolormesh(np.arange(n_f + 1),
                         np.arange(n_s + 1),
                         np.ma.masked_invalid(ratio.T.values),
                         cmap=cmap, norm=norm,
                         edgecolors="lightgray", linewidth=0.4)

    ax.set_xticks(np.arange(n_f) + 0.5)
    ax.set_xticklabels(feature_order, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(np.arange(n_s) + 0.5)
    ax.set_yticklabels([f"Site {s}" for s in sites])
    ax.set_ylabel("Site", fontweight="bold")
    ax.invert_yaxis()                     # Site 1 이 맨 위
    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

    if ANNOTATE:
        for i in range(n_s):
            for j in range(n_f):
                v = ratio.values[j, i]
                if not np.isnan(v):
                    ax.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center",
                            va="center", fontsize=config.fs(6),
                            color="white" if (v >= dmax * 0.75 or v <= dmax * 0.15)
                            else "black")

    # ── 컬러바: 히트맵 + x축(회전 라벨) 아래 가로 배치 ────────────────────────
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="bottom",
                      shrink=0.45, pad=0.01, aspect=40)
    ticks = [t for t in (0.01, 0.1, 1, 10, 50) if LOG_VMIN <= t <= dmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.set_label(f"IG contribution [%], log scale   "
                 f"(red = more important in {config.LABEL_CRC}; grey = 0.00%)")
    config.thin_spines(cb.ax)

    out = os.path.join(config.SAVE_DIR, "9_a.png")
    plt.savefig(out)
    plt.close(fig)
    print("saved: 9_a.png")
    print(f"   color scale: log {LOG_VMIN} (blue) ~ {dmax:.1f} % (red); "
          f"{int(ratio.isna().values.sum())} cells with 0.00% drawn in grey")


if __name__ == "__main__":
    main()
