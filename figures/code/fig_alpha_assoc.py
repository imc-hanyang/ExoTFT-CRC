"""
Fig α-assoc. Adaptive α 가 무엇에 반응하는가 — 기상/PV 입력 vs 예측 난이도.

  5_d.png  사이트별 α 일단위 계단 + 전일 NMAE(ExoTFT-RF / ExoTFT-CRC) 오버레이,
           α 상승일(▲) / 하락일(▼) 표시
  5_e.png  α 상승일 vs 하락일의 **전일 조건** 분포 비교 (사이트 내 백분위 순위)
  5_f.png  통합 로지스틱 계수 forest plot — 매개(mediation) 검정

통계 계산은 alpha_association.py 를 그대로 재사용합니다(중복 정의 없음).

읽는 법
-------
5_d : α 는 15분 스텝이 아니라 **하루 단위 계단**이고 곱셈형으로만 움직입니다.
      회색 실선(전일 ExoTFT-RF NMAE = 보정 전)이 솟은 다음 날 ▼ 가 찍히는
      경향을 봅니다. 회색 실선 = 보정 전, 초록 파선 = 보정 후(ExoTFT-CRC) 이므로
      **두 선의 간격이 곧 그날 잔차 보정 α·ê 의 순이득**입니다.
      초록이 아래면 이득, 위면 손해.
      (SHOW_DIFFICULTY_IN_5D = True 로 두면 aa.DIFFICULTY 선이 추가돼
       롤링 재학습 효과까지 분리해 볼 수 있습니다.)
5_e : 상자 두 개가 겹치면 그 변수는 α 방향과 무관. GHI·cf_mean 은 겹치고
      nmae_base·vol 은 벌어집니다.
5_f : vol 단독은 유의하지만 nmae_base 를 같이 넣으면 0 선을 물고,
      nmae_base 는 계속 0 을 벗어난 채 남습니다 → 날씨 효과는 예측 난이도에 매개됨.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
import alpha_association as aa

SITES = config.sites

# 5_e / 5_f 에 올릴 변수 (전체 FEATS 중 해석이 되는 대표군만)
# 5_e 에 올릴 변수 — alpha_association 이 계산하는 후보를 폭넓게 스크리닝합니다.
#   제외: vol / vol_std  … 변동성은 5_f 매개 검정에서 예측 난이도에 흡수되는 것이
#                          이미 확인돼, 스크리닝 단계에 둘 이유가 없음
#         clearness      … cf_mean/GHI 로 만든 값이라 사실상 performance ratio 이고
#                          이름도 표준 clearness index 와 달라 오해를 부름
#   improv(Correction net gain) 제외 … α 제어기의 판정 입력 그 자체라 거의 완전분리가
#                                       나옴. 스크리닝 변수와 같은 층위로 읽히면 오해.
#   cf_max 제외 … cf_mean 과 사실상 같은 것을 재는 중복 변수
#   hit_rate(Correction hit rate) 제외 … improv 와 마찬가지로 보정 결과에서 나온
#                                        사후 지표라 기상 조건과 같은 줄에 두면 오해.
#   난이도(aa.DIFFICULTY) 제외 … 5_e 에서 n.s. 라 스크리닝에 정보를 더하지 않음
#                                (5_f 회귀에는 통제변수로 그대로 들어갑니다).
SHOW_FEATS = ["GHI", "DNI", "TSI", "Temp", "Pres", "cf_mean"]
NICE = {
    "GHI": "GHI", "Temp": "Temperature", "cf_mean": "Capacity\nFactor",
    "clearness": "Clearness\nProxy", "vol": "Volatility\n(15-min Abs. Ramp)", "vol_std": "Volatility\nStd",
    "nmae_base": f"{config.LABEL_FB} NMAE\n(Forecast Difficulty)",
    "nmae_roll": f"{config.LABEL_RF} NMAE\n(Forecast Difficulty)",
    "nrmse_base": f"{config.LABEL_FB} NRMSE\n(Forecast Difficulty)",
    "nrmse_roll": f"{config.LABEL_RF} NRMSE\n(Forecast Difficulty)",
    "hit_rate": "Correction\nHit Rate", "improv": "Correction\nNet Gain",
    "day_idx": "Elapsed Day\n(Seasonal Trend)", "DNI": "DNI", "TSI": "TSI", "Pres": "Pressure",
    "cf_max": "Capacity\nFactor (Max)",
}
# ── 5_d 보조축에 올릴 NMAE 선 ────────────────────────────────────────────────
# 기본은 ExoTFT-RF(보정 전) + ExoTFT-CRC(보정 후) 두 선만. 두 선의 간격이 곧
# 잔차 보정 α·ê 의 순이득이라, α 제어기의 판정과 직접 대응합니다.
#
# SHOW_DIFFICULTY_IN_5D = True 로 두면 aa.DIFFICULTY(전일 ExoTFT-RF NRMSE,
# 5_e·5_f 가 쓰는 난이도 지표)를 회색 실선으로 함께 깔아 롤링 재학습 효과까지
# 분리해 볼 수 있습니다. 다만 선이 3개가 되어 패널이 복잡해집니다.
SHOW_DIFFICULTY_IN_5D = False
BASELINE_5D = aa.DIFFICULTY
BASELINE_LABEL = {"nmae_roll": f"Prev-Day {config.LABEL_RF} NMAE",
                  "nmae_base": f"Prev-Day {config.LABEL_FB} NMAE"}

SHOW_ROLLING_IN_5D = True

C_UP, C_DN = config.clist[2], config.clist[0]      # 상승 = 빨강, 하락 = 파랑
# 5_d 과대/과소 보정 채움 — ▲▼ 와 같은 계열이라 "붉은 면 뒤에 ▼" 가 바로 읽힙니다.
C_OVER, C_UNDER = config.clist[2], config.clist[0]
C_OPT = "0.35"                                     # α* 선 (사후 최적, 오라클)
# α* 는 잔차 예측이 실제 잔차와 반대 부호인 날 크게 튑니다. 축을 그 꼬리에
# 뺏기지 않도록 분위수로 자르고, 잘린 점만 경계 마커로 남깁니다.
OPT_CLIP_Q = 0.95
C_AUX = config.clist[7]                            # 회색
C_RECO = config.clist[5]                           # 초록
C_ROLL = config.clist[1]                           # 하늘색

# 5_d 보조축 선 스타일
STYLE_ROLL = dict(color=C_AUX, linewidth=0.7, linestyle="-")     # 회색 실선
STYLE_RECO = dict(color=C_RECO, linewidth=0.7, linestyle="--")   # 초록 파선
# SHOW_DIFFICULTY_IN_5D=True 로 선이 3개가 될 때만 롤링을 하늘색 점선으로 비켜 줌
STYLE_ROLL_ALT = dict(color=C_ROLL, linewidth=0.5, linestyle=":")
#   ▲▼ 가 빨강/파랑을 이미 쓰고 있어서, config.MODELS 의 base(파랑)/reco(빨강)
#   색을 그대로 쓰면 마커와 충돌합니다. 그래서 회색/초록으로 분리했습니다.

# 5_f — 교란 검정: "맑은 하늘 효과" 가 계절 추세의 그림자인가?
#
# 테스트 구간이 11~12월이라 일사가 단조 하강하고, α 상승일도 후반에 몰려 있어
# (경과일 rank-biserial +0.31, p=0.0009) 두 요인이 얽혀 있습니다. 그래서 각
# 블록은 **설명변수 조합을 달리한 별개의 로지스틱 회귀**이며, 같은 변수의 계수가
# 다른 변수를 넣어도 0 을 벗어난 채 남는지를 세로로 훑어보는 그림입니다.
#
#   DNI only / elapsed day only  … 각각 단독으로는 유의
#   DNI + elapsed day            … 둘 다 유의하게 남으면 서로 독립 (핵심 블록)
#   DNI + TSI + elapsed day      … DNI·TSI 는 같은 것(청천)을 재는 중복 변수
#   full                         … 기상·PV·난이도를 다 넣어도 DNI 가 남는지
#
# day_idx 는 더 이상 '숨은 통제변수' 가 아니라 비교 대상이므로 조합에 명시합니다.
FOREST_MODELS = [
    (["DNI"], "DNI only"),
    (["day_idx"], "elapsed day only"),
    (["DNI", "day_idx"], "DNI + elapsed day"),
    (["DNI", "TSI", "day_idx"], "DNI + TSI + elapsed day"),
    (["GHI", "DNI", "Temp", "cf_mean", aa.DIFFICULTY, "day_idx"],
     "weather + PV + difficulty\n+ elapsed day"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 5_d. α 계단 + 전일 예측 난이도
# ══════════════════════════════════════════════════════════════════════════════
# 5_d 패널 한 칸의 높이 [inch]. 폭이 7.2 → 5.42 in 로 좁아졌으니 높이도 같은
# 비율로 줄여, 지면에서 차지하는 자리는 그대로 두고 글자만 제 크기로 인쇄되게 합니다.
PANEL_H = 0.95 * config.H_SCALE      # ≈ 0.71 in
# 5_d 아래쪽 가로 범례가 차지할 높이 [inch] — 글자 높이라 캔버스 폭과 무관한 절대값.
# 좁아진 폭에 6칸이 한 줄로 들어가지 않아 3칸 × 2줄로 접습니다.
LEGEND_H_IN = 0.52
LEGEND_NCOL = 3
LEGEND_COLSPACING = 1.2
def plot_alpha_steps(tabs):
    """α 계단 + 사후 최적 α(α*) 오버레이.

    보조축(전일 NMAE) 대신 **같은 α 축 위에** α* 를 얹습니다. 두 값의 단위가
    같아 twin axis 가 필요 없고, 0 선이 항상 화면 안에 남습니다.

        α_used > α*  →  과대보정  (붉은 채움)  →  다음 갱신에서 α 를 줄여야 함
        α_used < α*  →  과소보정  (푸른 채움)  →  다음 갱신에서 α 를 키워야 함

    따라서 붉은 구간 뒤에 ▼, 푸른 구간 뒤에 ▲ 가 찍히면 제어기가 의도대로
    동작한 것입니다. α* 는 사후에만 알 수 있는 오라클 값이라 알고리즘이
    참조할 수 있는 목표가 아니라 **평가 기준선**입니다.

    α 가 감쇠해 Correction 이 결과 CSV 의 저장 정밀도까지 작아진 블록은
    alpha_association 에서 이미 NaN 으로 비워져 여기서는 빈칸으로 남습니다.
    """
    n = len(tabs)
    fig, axes = plt.subplots(n, 1,
                             figsize=(config.FULL_W, PANEL_H * n + 0.6 + LEGEND_H_IN),
                             sharex=False)
    axes = np.atleast_1d(axes)

    for ax, (site, t) in zip(axes, tabs.items()):
        x = t["day_idx"].values
        a = t["alpha"].values
        opt = t["alpha_opt"].values.astype(float)

        # ── y 범위: α 와 α* 를 함께 담되, α* 의 꼬리에 축을 뺏기지 않도록
        #    분위수로 자릅니다. 잘린 점은 아래에서 삼각 마커로 표시합니다.
        fin = np.isfinite(opt)
        hi = max(a.max(), np.nanquantile(opt[fin], OPT_CLIP_Q) if fin.any() else 0.0)
        lo = min(0.0, np.nanquantile(opt[fin], 1 - OPT_CLIP_Q) if fin.any() else 0.0)
        pad = 0.12 * max(hi - lo, 1e-3)
        ylo, yhi = lo - pad, hi + pad
        shown = np.clip(opt, ylo, yhi)

        # ── 과대/과소 보정 채움 ──────────────────────────────────────────
        # step="post" 로 α 계단과 같은 보간을 써야 면이 계단에 붙습니다.
        # 부호별로 where 를 갈라 칠해야 두 색이 겹쳐 탁해지지 않습니다.
        for msk, col in ((fin & (a > shown), C_OVER), (fin & (a <= shown), C_UNDER)):
            ax.fill_between(x, a, shown, step="post", where=msk,
                            color=col, alpha=0.30, linewidth=0, zorder=1)

        ax.step(x, shown, where="post", color=C_OPT, linewidth=0.7,
                linestyle="--", zorder=3)
        # 축 밖으로 잘린 α* 는 경계에 마커로 남겨 존재를 알립니다.
        for m, mk in ((fin & (opt > yhi), "^"), (fin & (opt < ylo), "v")):
            if m.any():
                ax.scatter(x[m], np.where(opt[m] > yhi, yhi, ylo), marker=mk,
                           s=6, color=C_OPT, edgecolors="none", zorder=5)

        ax.step(x, a, where="post", color="black", linewidth=0.9, zorder=4)

        d = np.diff(a, prepend=np.nan)
        up, dn = d > 1e-12, d < -1e-12
        ax.scatter(x[up], a[up], marker="^", s=11, color=C_UP,
                   edgecolors="none", zorder=6)
        ax.scatter(x[dn], a[dn], marker="v", s=11, color=C_DN,
                   edgecolors="none", zorder=6)

        ax.axhline(0, color="0.6", linewidth=0.4, linestyle=":", zorder=0)
        ax.set_ylabel(r"$\alpha$")
        ax.set_ylim(ylo, yhi)
        # Site 7·8 은 α 선이 라벨 높이까지 올라와 글자에 겹칩니다. 흰 바탕을 깔아
        # 글자가 선에 먹히지 않게 합니다.
        ax.text(0.004, 0.96, f"Site {site}", transform=ax.transAxes,
                va="top", fontsize=config.PT_LABEL, fontweight="bold",
                zorder=7,   # α 선(zorder 4)·마커(6) 보다 위
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                          boxstyle="square,pad=0.12"))
        config.set_nature_ticks(ax)

    axes[-1].set_xlabel("Test Day Index")
    handles = [
        Line2D([], [], color="black", lw=0.9, label=r"$\alpha$ (24h step)"),
        Line2D([], [], color=C_OPT, lw=0.7, ls="--",
               label=r"$\alpha^{*}$ (post-hoc optimum)"),
        Patch(facecolor=C_OVER, alpha=0.30, label="over-correction"),
        Patch(facecolor=C_UNDER, alpha=0.30, label="under-correction"),
        Line2D([], [], marker="^", color=C_UP, lw=0, ms=4, label="increase"),
        Line2D([], [], marker="v", color=C_DN, lw=0, ms=4, label="decrease"),
    ]
    # 범례는 패널 스택 맨 아래에 6칸 한 줄로 가로로 깔고, 패널은 그림 폭을
    # 전부 씁니다.
    bottom = LEGEND_H_IN / fig.get_figheight()
    plt.tight_layout(pad=0.25, rect=[0, bottom, 1, 1], h_pad=0.45)
    fig.legend(handles=handles, loc="lower center", ncol=LEGEND_NCOL,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=LEGEND_COLSPACING, handlelength=1.5,
               handletextpad=0.5, borderpad=0.0)
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "5_d.png"), dpi=300)
    plt.close(fig)
    print("saved: 5_d.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5_e. 상승일 vs 하락일 전일 조건 분포
# ══════════════════════════════════════════════════════════════════════════════
def plot_updown_dist(tabs):
    from scipy.stats import mannwhitneyu
    P = aa.pooled_design(tabs)                 # 사이트 내 백분위 + 전일 시프트 완료
    up, dn = P[P["y"] == 1], P[P["y"] == 0]

    # 상자 영역만 원고 폭 비율로 줄이고(H_SCALE), 2줄 x눈금·라벨 여백 0.45 in 는 유지
    fig, ax = plt.subplots(figsize=(config.FULL_W,
                                    (2.9 - 0.45) * config.H_SCALE + 0.45))
    w = 0.32
    for i, f in enumerate(SHOW_FEATS):
        for grp, off, col in ((up, -w / 2 - 0.03, C_UP), (dn, +w / 2 + 0.03, C_DN)):
            v = grp[f].dropna().values
            bp = ax.boxplot([v], positions=[i + off], widths=w, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=0.9),
                            whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                            boxprops=dict(lw=0.5))
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.55)

        a, b = up[f].dropna().values, dn[f].dropna().values
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rb = 2 * u / (len(a) * len(b)) - 1
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(i, 1.055, star, ha="center", va="bottom", fontsize=config.PT_ANNOT,
                fontweight="bold" if p < 0.05 else "normal",
                color="black" if p < 0.05 else "0.55")
        ax.text(i, 1.005, f"{rb:+.2f}", ha="center", va="bottom", fontsize=config.PT_ANNOT,
                color="0.35")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.5, zorder=1)
    ax.set_xticks(range(len(SHOW_FEATS)))
    ax.set_xticklabels([NICE.get(f, f) for f in SHOW_FEATS], fontsize=config.PT_TICK)
    ax.set_ylim(-0.03, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Previous-Day Condition\n(Within-Site Percentile Rank)",
                  fontsize=config.PT_LABEL)
    # 제목은 논문 캡션이 맡습니다. 캡션에 쓸 표본 수는 콘솔로 찍어 둡니다.
    print(f"   5_e: n = {len(up)} (alpha increase) vs {len(dn)} (alpha decrease) days")
    config.set_nature_ticks(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=C_UP, alpha=0.55, ec="black", lw=0.5,
                             label=r"$\alpha$ increase"),
               plt.Rectangle((0, 0), 1, 1, fc=C_DN, alpha=0.55, ec="black", lw=0.5,
                             label=r"$\alpha$ decrease")]
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=config.PT_LEGEND,
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9)
    plt.tight_layout(pad=0.25)
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "5_e.png"), dpi=300)
    plt.close(fig)
    print("saved: 5_e.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5_f. 로지스틱 계수 forest plot (매개 검정)
# ══════════════════════════════════════════════════════════════════════════════
def plot_forest(tabs):
    P = aa.pooled_design(tabs)
    fits = []
    for cols, name in FOREST_MODELS:
        res = aa.logit(P, cols)
        res["model"] = name
        fits.append(res)
    fits = pd.concat(fits, ignore_index=True)

    fig, ax = plt.subplots(figsize=(config.FULL_W,
                                    0.19 * len(fits) + len(FOREST_MODELS) * 0.26 + 0.9))
    # 블록(모델)마다 헤더 행을 하나 두고 그 아래에 변수들을 배치.
    # 모델명을 y축 왼쪽에 회전시켜 넣으면 변수 라벨과 겹치므로 헤더 행 방식을 씁니다.
    ypos, ylabels, seps, headers = [], [], [], []
    y = 0.0
    for name in [n for _, n in FOREST_MODELS]:
        headers.append((y, name))
        y += 0.85
        for _, row in fits[fits["model"] == name].iterrows():
            ypos.append(y)
            ylabels.append(NICE.get(row["var"], row["var"]).replace("\n", " "))
            y += 1.0
        seps.append(y - 0.1)
        y += 0.5

    for (_, row), yy in zip(fits.iterrows(), ypos):
        lo, hi = row["coef"] - 1.96 * row["se"], row["coef"] + 1.96 * row["se"]
        sig = row["p"] < 0.05
        col = (C_UP if row["coef"] > 0 else C_DN) if sig else "0.62"
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.0, solid_capstyle="round")
        ax.scatter([row["coef"]], [yy], s=16 if sig else 11, color=col,
                   edgecolors="none", zorder=4)
        ax.text(hi + 0.12, yy, f"p={row['p']:.3f}", va="center", fontsize=config.PT_ANNOT,
                color="black" if sig else "0.55")

    ax.axvline(0, color="black", linestyle="--", linewidth=0.6, zorder=1)
    for sp in seps[:-1]:
        ax.axhline(sp, color="0.85", lw=0.4, zorder=0)

    # 블록 헤더: 축 왼쪽 끝에 모델명(설명변수 조합)
    for yy, name in headers:
        ax.text(0.01, yy, f"model: {name}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=config.PT_ANNOT, fontweight="bold", color="0.25")

    lo_all = (fits["coef"] - 1.96 * fits["se"]).min()
    hi_all = (fits["coef"] + 1.96 * fits["se"]).max()
    # 오른쪽 p 값 텍스트 자리 확보. 캔버스가 7.2 → 5.42 in 로 좁아져 같은 7 pt
    # 글자가 데이터 좌표를 더 많이 먹으므로 여유를 H_SCALE 만큼 되돌려 넓힙니다.
    ax.set_xlim(lo_all - 0.25 / config.H_SCALE, hi_all + 1.15 / config.H_SCALE)
    ax.set_ylim(-0.9, max(ypos) + 0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=config.PT_TICK)
    ax.invert_yaxis()
    # 축 단위는 확률이 아니라 log-odds 입니다. 예전 라벨이 "(+ = higher probability
    # of α increase)" 라 축 자체가 확률로 읽혀 오해를 만들었습니다.
    # labelpad: 아래 좌우 화살표 주석과 같은 줄에 놓이지 않도록 한 줄 내려 씁니다.
    ax.set_xlabel(r"Logistic Coefficient (Log-Odds) $\pm$ 95% CI", fontsize=config.PT_LABEL, labelpad=14)
    config.set_nature_ticks(ax)

    # 부호 방향 화살표. x라벨(가운데)과 같은 줄이면 겹치므로, 화살표를 눈금 라벨
    # 바로 아래 좌우 끝에 두고 x라벨은 labelpad 로 그 아래 줄에 내려 놓습니다.
    ax.annotate(r"$\longleftarrow$ $\alpha$ decrease",
                xy=(0.0, -0.052), xycoords="axes fraction", ha="left", va="top",
                fontsize=config.PT_ANNOT, color=C_DN, annotation_clip=False)
    ax.annotate(r"$\alpha$ increase $\longrightarrow$",
                xy=(1.0, -0.052), xycoords="axes fraction", ha="right", va="top",
                fontsize=config.PT_ANNOT, color=C_UP, annotation_clip=False)

    # 위쪽 보조축: 같은 위치를 odds ratio = exp(계수) 로 읽어 줍니다
    lo_x, hi_x = ax.get_xlim()
    ors = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 50]
    ticks = [(np.log(o), o) for o in ors if lo_x <= np.log(o) <= hi_x]
    axt = ax.twiny()
    axt.set_xlim(lo_x, hi_x)
    axt.set_xticks([p for p, _ in ticks])
    axt.set_xticklabels([f"{o:g}" for _, o in ticks], fontsize=config.PT_TICK)
    axt.set_xlabel("Odds Ratio for $\\alpha$ Increase  (= exp of Coefficient)",
                   fontsize=config.PT_LABEL)
    config.set_nature_ticks(axt)
    plt.tight_layout(pad=0.25)
    # y 눈금 라벨이 길어 축 박스가 오른쪽으로 밀립니다. 위쪽 보조축 라벨은 축이
    # 아니라 **그림** 가운데에 맞춰야 오른쪽에서 잘리지 않습니다.
    pos = ax.get_position()
    axt.xaxis.label.set_x((0.5 - pos.x0) / pos.width)
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "5_f.png"), dpi=300)
    plt.close(fig)
    print("saved: 5_f.png")


def main():
    tabs = {}
    for s in SITES:
        try:
            tabs[s] = aa.daily_table(s)
        except (FileNotFoundError, KeyError) as e:
            print(f"   [skip] site {s}: {e}")
    if not tabs:
        print("Alpha_Used 를 가진 사이트가 없어 건너뜁니다.")
        return
    plot_alpha_steps(tabs)
    plot_updown_dist(tabs)
    plot_forest(tabs)


if __name__ == "__main__":
    main()
