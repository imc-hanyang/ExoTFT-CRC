"""
Fig 9. Integrated Gradients(IG) 피처 중요도의 **모델 간 차이** 히트맵 (단일 패널).

  Δ = ExoTFT-CRC 잔차 모델 ratio − ExoTFT-FB 모델 ratio   [%p]
      **양수(빨강) = 잔차 모델에서 더 중요해진 피처**
      음수(파랑)   = ExoTFT-FB 에서 더 중요했던 피처

  행 = 피처, 열 = 사이트.

이전 버전은 (1) ExoTFT-FB 랭크, (2) ExoTFT-CRC 랭크, (3) 랭크 차이 3열이었는데,
지금은 **차이 패널만** 남기고 지표도 rank 차이 → ratio 차이로 바꿨습니다.
rank 는 ExoTFT-FB 17개 / Residual 22개로 분모가 달라 차이의 크기를 해석하기 어렵고,
등수가 1 계단 바뀐 것과 기여도가 실제로 크게 바뀐 것을 구분하지 못합니다.
ratio(전체 attribution 중 해당 피처의 비중, %)는 두 모델에서 모두 합이 100% 라
그 차이를 %p 로 바로 읽을 수 있습니다.

ExoTFT-FB 모델에 없는 피처(IMF_1~4, Raw_Error)는 Δ 를 정의할 수 없으므로 회색 빈 칸.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

# 폭은 원고 표준(config.FULL_W = 원고 \linewidth = 390 pt)으로 고정 — 배율 1.0 이라 config 의
# pt 가 그대로 인쇄됩니다.
config.figure_config()

IG_DIR = os.path.join(config.DATA_ROOT, "ig")
PANELS = [
    (config.LABEL_FB, "IG_base_feature.csv"),
    (config.LABEL_CRC, "IG_residual_feature.csv"),
]

# 행 정렬 (위 → 아래):
#   "abs_delta" 변동이 큰 순. 사이트별 |Δ| 의 평균이 큰 피처가 맨 위 (기본)
#   "max_delta" 변동이 큰 순이지만 사이트별 |Δ| 의 **최대값** 기준.
#               한 사이트에서만 크게 튄 피처도 위로 올라옵니다.
#   "mean_rank" 두 모델 평균 IG 랭크 순 (= 원래 중요한 피처가 위, 9_ref 방식)
#   "group"     group(Endogenous/Residual/Weather/Temporal) 별로 묶고 그 안에서 랭크 순
ROW_ORDER = "abs_delta"
GROUP_SEQ = ["Endogenous", "Residual", "Weather", "Temporal"]

ANNOTATE = False        # True 면 셀에 Δ 값(%p) 표기
CMAP_NAME = "rank_rb"   # config 커스텀: 빨강 → 파랑. 차이용으로는 반전해서 씀
NA_COLOR = "0.80"       # 해당 없음(ExoTFT-FB 에 없는 피처) 셀 색

CLIP_PERCENTILE = 100   # 색 스케일 상한 분위수(100 = 최대값 그대로)
CELL_H = 0.45 * config.H_SCALE   # 히트맵 셀 한 칸(사이트 1개)의 높이 [inch]

# ExoTFT-FB 에 없는 피처(IMF_1~4, Raw_Error)는 Δ 가 전 사이트 NaN → 행 전체가 회색.
# 차이만 보는 그림에서는 정보가 없으므로 기본은 숨김. False 로 두면 회색 행으로 남습니다.
HIDE_EMPTY_ROWS = True


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
    frames = {title: load_ig(f) for title, f in PANELS}

    allf = pd.concat(frames.values(), ignore_index=True)
    mean_rank = allf.groupby("feature")["rank"].mean()
    group_of = allf.drop_duplicates("feature").set_index("feature")["group"]
    features = mean_rank.sort_values().index.tolist()   # 임시 index (정렬은 아래에서)
    sites = sorted(allf["site_num"].unique())

    # ── Δ ratio 계산 ─────────────────────────────────────────────────────────
    def ratio_pivot(title):
        return (frames[title]
                .pivot(index="feature", columns="site_num", values="ratio")
                .reindex(index=features, columns=sites))

    base_p, res_p = ratio_pivot(PANELS[0][0]), ratio_pivot(PANELS[1][0])
    # ratio 는 클수록 중요 → residual − base 가 양수면 잔차 모델에서 상승
    diff = res_p - base_p
    diff_label = f"\u0394 IG Contribution  ({config.LABEL_CRC} \u2212 {config.LABEL_FB})"

    if HIDE_EMPTY_ROWS:
        keep = diff.notna().any(axis=1)
        dropped = [f for f, k in zip(diff.index, keep) if not k]
        if dropped:
            print(f"  hidden ({config.MODELS['reco'].plain}-only, no {config.MODELS['base'].plain} counterpart): {dropped}")
        diff = diff[keep]

    # ── 행(피처) 순서 결정 ───────────────────────────────────────────────────
    # 변동량 기준 정렬은 Δ 가 있어야 계산되므로 diff 를 만든 뒤에 수행합니다.
    if ROW_ORDER in ("abs_delta", "max_delta"):
        mag = (diff.abs().max(axis=1) if ROW_ORDER == "max_delta"
               else diff.abs().mean(axis=1))
        # 큰 변동이 위(= invert_yaxis 후 첫 행)로 오도록 내림차순.
        # 동률/NaN 은 평균 랭크로 tie-break.
        key = pd.DataFrame({"m": mag, "r": mean_rank.reindex(mag.index)})
        feature_order = key.sort_values(["m", "r"], ascending=[False, True]).index.tolist()
    elif ROW_ORDER == "group":
        key = pd.DataFrame({"g": group_of.reindex(diff.index),
                            "r": mean_rank.reindex(diff.index)})
        key["gi"] = key["g"].map({g: i for i, g in enumerate(GROUP_SEQ)}).fillna(99)
        feature_order = key.sort_values(["gi", "r"]).index.tolist()
    else:
        feature_order = mean_rank.reindex(diff.index).sort_values().index.tolist()

    diff = diff.reindex(index=feature_order)

    vals = np.abs(diff.values[~np.isnan(diff.values)])
    dmax = float(np.percentile(vals, CLIP_PERCENTILE)) if vals.size else 1.0
    dmax = dmax or 1.0

    cmap = config.get_cmap(CMAP_NAME).reversed()   # 파랑(음) → 빨강(양)
    cmap.set_bad(NA_COLOR)

    # ── 그리기 (가로 방향: x = 피처, y = 사이트) ────────────────────────────
    #   셀 높이(CELL_H)만 고정하고 폭은 남는 공간을 채우게 둡니다.
    #   box_aspect 로 비율을 고정하면 남는 세로 공간이 통째로 여백이 되어
    #   x라벨과 컬러바가 히트맵에서 떨어져 보입니다.
    n_f, n_s = len(feature_order), len(sites)
    fig_h = CELL_H * n_s + 1.75              # + 회전 라벨 + 컬러바 + 제목 여백
    fig, ax = plt.subplots(figsize=(config.FULL_W, fig_h), layout="constrained")
    # 회전된 피처 라벨 때문에 기본 여백이 크게 잡혀 컬러바가 멀어집니다.
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.0, wspace=0.0)

    mesh = ax.pcolormesh(np.arange(n_f + 1),
                         np.arange(n_s + 1),
                         diff.T.values, cmap=cmap, vmin=-dmax, vmax=dmax,
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
                v = diff.values[j, i]
                if not np.isnan(v):
                    ax.text(j + 0.5, i + 0.5, f"{v:+.1f}", ha="center",
                            va="center", fontsize=config.fs(6),
                            color="white" if abs(v) >= dmax * 0.75 else "black")

    # ── 컬러바: 히트맵 + x축(회전 라벨·x라벨) 아래 가로 배치 ──────────────────
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="bottom",
                      shrink=0.45, pad=0.01, aspect=40)
    # 9 pt 기준 한 줄 폭 ≈5.2 in 이라 FULL_W(5.42 in) 를 거의 꽉 채웁니다.
    cb.set_label(f"{diff_label}   (red = more important in {config.LABEL_CRC})")
    config.thin_spines(cb.ax)

    out = os.path.join(config.SAVE_DIR, "9.png")
    plt.savefig(out)
    plt.close(fig)
    print("saved: 9.png")


if __name__ == "__main__":
    main()
