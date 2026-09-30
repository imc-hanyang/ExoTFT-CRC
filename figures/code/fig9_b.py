"""
Fig 9-b. Encoder 단위 Integrated Gradients(IG) 기여도의 **모델 간 변화** 막대그래프.

  값 = Δ ratio = ExoTFT-CRC(잔차 모델) − ExoTFT-FB   [%p]
  x  = 사이트 7개, 사이트마다 막대 3개 = Endogenous / Weather / Temporal
  양수 = 잔차 모델에서 그 입력군의 비중이 커졌다 / 음수 = 작아졌다

fig9(= 9.png, feature 단위 Δ 히트맵)의 encoder(입력군) 단위 요약판입니다.
빼는 기준도 9.png 와 동일하게 맞췄습니다 — 잔차 모델(IG_residual_*) 에서
ExoTFT-FB(IG_base_*) 를 빼고, 지표는 rank 가 아니라 ratio 입니다. ratio 는 두
모델 모두 사이트별 합이 100% 라서 그 차이를 %p 로 바로 읽을 수 있습니다.

주의: residual encoder 에는 ExoTFT-CRC 전용 입력군 Residual_Sequence 가 하나 더
있습니다(전 사이트 0.02~0.07%). FB 에 대응 항목이 없어 Δ 를 정의할 수 없고 크기도
무시할 수준이라 제외합니다 — 그래서 사이트당 막대가 3개입니다.
(9.png 이 FB 에 없는 피처를 HIDE_EMPTY_ROWS 로 지우는 것과 같은 이유입니다.)

읽는 법 — 이 그림은 **제로섬(zero-sum) 재배분** 그림입니다
--------------------------------------------------------
ratio 는 두 모델 모두 사이트별 합이 100% 이므로, 한 입력군의 비중이 오르면 다른
입력군이 정확히 그만큼 내려갑니다. 실제로 사이트별 Δ 3개의 합은 −0.07 ~ −0.01 %p
이고, 이 잔량은 위에서 뺀 Residual_Sequence 의 비중(0.02~0.07%) 그 자체입니다.
따라서 막대 높이는 "attribution 총량이 늘었다/줄었다" 가 아니라 **모델이 근거를 어느
입력군으로 옮겼는지**만 말해 줍니다. 총량 비교로 오해하지 않도록 본문에도 이 점을
적어야 합니다.

그 결과 Endogenous 와 Weather 는 거의 완전한 거울상입니다(사이트 간 상관 −0.99).
Temporal 은 |Δ| 가 최대 2.9 %p(Site 1)로 작아, 사실상 **Endogenous ↔ Weather 사이의
1차원 이동**을 세 막대로 보고 있는 셈입니다.

방향은 사이트마다 갈립니다.
  · Endogenous 쪽으로 이동 (내생 신호 의존 강화): Site 1(+10.4), 2(+4.1), 5(+6.5), 7(+7.8)
  · Weather 쪽으로 이동   (기상 입력 의존 강화): Site 4(+3.2), 6(+25.2), 8(+4.0)
Site 6 이 압도적입니다 — Weather 4.65% → 29.84%, Endogenous 94.94% → 69.02%.
다른 사이트의 이동폭이 10 %p 이내인 것과 비교하면 한 자리 수가 다릅니다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

config.figure_config()

IG_DIR = os.path.join(config.DATA_ROOT, "ig")
BASE_FILE = "IG_base_encoder.csv"        # ExoTFT-FB   — 빼는 쪽(기준)
RECO_FILE = "IG_residual_encoder.csv"    # ExoTFT-CRC  — 빼지는 쪽

GROUPS = ["Endogenous", "Meteorological", "Temporal"]
# CSV 의 group 값 → 그림에 표시할 이름 (CSV 에는 여전히 "Weather" 로 저장돼 있음)
GROUP_RENAME = {"Weather": "Meteorological"}
# 색은 config 의 discrete 팔레트(clist)를 순서대로. get_colors 가 개수에 맞춰
# 잘라 주므로 인덱스를 그림마다 손으로 고르지 않습니다.
GROUP_COLORS = config.get_colors(len(GROUPS))

# 아래 가로 범례 한 줄이 차지할 높이 [inch] — 5_d 와 같은 방식.
# 이걸 빼먹으면 축이 그만큼 낮아져 2줄짜리 y축 라벨이 위에서 잘립니다.
LEGEND_H_IN = 0.28

ANNOTATE = True                 # 막대 끝에 값 표기
BAR_W = 0.26                    # 막대 하나의 폭 (3개 → 0.78)
FIG_H = (2.6 - 0.55) * config.H_SCALE + 0.55 + LEGEND_H_IN   # 막대 영역만 축소
                                                            # + 아래 범례 한 줄

# ── 글자 크기 ─────────────────────────────────────────────────────────────────
# 짝 그림인 9.png(fig9.py)와 완전히 같은 체계를 씁니다 — 축 라벨 9 pt / 범례 8 pt
# / tick 7 pt 는 config 기본 그대로이고, 여기서 지정하는 건 막대 위 ±값 하나뿐.
# 9.png 은 ANNOTATE=False 라 셀 안 숫자를 찍지 않아서 **가장 작은 글자가 7 pt tick**
# 입니다. 이 그림만 fs(6)=5.4 pt 를 쓰면 그 바닥선 아래로 혼자 내려가므로
# tick 과 같은 7 pt 로 맞춥니다.
PT_ANNOT = config.PT_ANNOT           # 7 pt — 막대 위 +/− 값
# 아래 가로 범례 3칸 간격 (5_d 와 동일)
LEGEND_COLSPACING = 1.2


def load_ig(fname: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(IG_DIR, fname), encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["group"] = df["group"].astype(str).str.strip().replace(GROUP_RENAME)
    df["site_num"] = df["site"].astype(str).str.extract(r"(\d+)").astype(int)
    df["ratio"] = pd.to_numeric(df["ratio"].astype(str).str.rstrip("%"),
                                errors="coerce")
    return df


def pivot(df, sites, groups):
    return (df.pivot(index="group", columns="site_num", values="ratio")
              .reindex(index=groups, columns=sites))


def main():
    base = load_ig(BASE_FILE)
    reco = load_ig(RECO_FILE)

    sites = sorted(set(base["site_num"]) & set(reco["site_num"]))
    groups = list(GROUPS)

    diff = pivot(reco, sites, groups) - pivot(base, sites, groups)

    x = np.arange(len(sites), dtype=float)
    fig, ax = plt.subplots(figsize=(config.FULL_W, FIG_H), layout="constrained")

    n = len(groups)
    offs = (np.arange(n) - (n - 1) / 2) * BAR_W
    for gi, (g, c) in enumerate(zip(groups, GROUP_COLORS)):
        v = diff.loc[g].values
        ax.bar(x + offs[gi], v, BAR_W, label=g, color=c,
               edgecolor="white", linewidth=0.3, zorder=3)
        if ANNOTATE:
            for xi, vi in zip(x + offs[gi], v):
                if np.isnan(vi):
                    continue
                ax.annotate(f"{vi:+.1f}", (xi, vi), ha="center",
                            va="bottom" if vi >= 0 else "top",
                            xytext=(0, 1.5 if vi >= 0 else -1.5),
                            textcoords="offset points",
                            fontsize=PT_ANNOT, color="0.25")

    ax.axhline(0, color="black", linewidth=0.5, zorder=4)
    ax.set_ylabel(f"Δ IG contribution\n({config.LABEL_CRC} − {config.LABEL_FB}) [%p]")
    vmax = np.nanmax(np.abs(diff.values))
    ax.set_ylim(-vmax * 1.30, vmax * 1.30)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Site {s}" for s in sites])
    ax.set_xlim(-0.5, len(sites) - 0.5)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    config.set_nature_ticks(ax, which="y")

    # 범례는 5_d 와 같이 패널 맨 아래에 가로 한 줄로. constrained layout 이
    # "outside lower center" 자리를 알아서 확보하므로 높이를 손으로 잡지 않습니다.
    # (위에 두면 Site 6 의 +25.2 막대 값과 같은 줄에서 다투게 됩니다.)
    fig.legend(*ax.get_legend_handles_labels(), ncol=len(groups),
               loc="outside lower center", columnspacing=LEGEND_COLSPACING,
               handlelength=1.5, handletextpad=0.5, frameon=False)

    out = "9_b.png"
    plt.savefig(os.path.join(config.SAVE_DIR, out))
    plt.close(fig)
    print(f"saved: {out}")

    # 콘솔 요약 — 그림에 적힌 숫자 확인용
    with pd.option_context("display.float_format", "{:+.2f}".format):
        print(diff.round(2).to_string())


if __name__ == "__main__":
    main()
