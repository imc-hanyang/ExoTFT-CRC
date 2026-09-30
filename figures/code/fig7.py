"""
Fig 7. 사이트별 상대오차 분포 (3D ridge / KDE) — 모델 오버레이 버전.

상대오차 = (예측 − 실측) / 설비용량
사이트 하나당 능선 하나를 두고, 그 위에 모델별 분포를 겹쳐 그립니다.
(색 = 모델, y축 = 사이트)  →  7.png

RIDGE_MODELS 로 겹쳐 그릴 모델을 고릅니다. None 이면 사용 가능한 전체.
모델이 3개 이상이면 겹침이 심해지니 2~3개 권장.

FILL = False (기본) 이면 면을 칠하지 않고 윤곽선만 그립니다.
겹치는 구간에서 뒤쪽 곡선이 가려지지 않아 비교가 쉽습니다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import config
import loader

# 폭은 원고 표준(config.FULL_W = 원고 \linewidth = 390 pt)으로 고정 — 배율 1.0 이라 config 의
# pt 가 그대로 인쇄됩니다.
config.figure_config()

# ── 기준선 스위치 opt-out ─────────────────────────────────────────────────────
# 이 그림은 **ExoTFT-FB** 와의 비교가 목적이므로 config.USE_ROLLING_AS_BASE 를
# 적용받지 않습니다. load_site 에 order 를 명시해 넘기면 치환이 일어나지 않습니다.
LOAD_ORDER = config.MODEL_ORDER

XLIM = (-0.25, 0.25)                    # 상대오차 표시 범위 (설비용량 대비)
RIDGE_MODELS = [config.FULLBATCH_BASE_KEY, config.RECO_KEY]  # None = 전체
FILL = False                            # True = 면 채우기, False = 윤곽선만
ALPHA = 0.55                            # FILL=True 일 때 면 투명도
LINEWIDTH = 1.0

# 능선 선 스타일 (겹치는 구간 구분용). 미지정 모델은 실선.
RIDGE_LINESTYLE = {
    config.FULLBATCH_BASE_KEY: "-",
    config.RECO_KEY: (0, (3.5, 1.5)),   # dashed
}


def main():
    sites = config.sites
    data = {s: loader.load_site(s, order=LOAD_ORDER, verbose=False) for s in sites}

    # 그릴 모델 확정 (MODEL_ORDER 순서 유지)
    keys = RIDGE_MODELS or [k for k in config.MODEL_ORDER
                            if any(config.MODELS[k].pred in data[s].columns for s in sites)]
    keys = [k for k in keys if any(config.MODELS[k].pred in data[s].columns for s in sites)]
    specs = [config.MODELS[k] for k in keys]

    fig = plt.figure(figsize=(config.FULL_W, config.FULL_W * 0.66))
    ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(*XLIM, 400)
    verts, face_colors, edge_colors, zs = [], [], [], []
    max_z = 0.0

    for idx, site in enumerate(sites):
        df = data[site]
        # 같은 사이트 안에서는 등록 순서대로 쌓아 뒤 모델이 위에 오도록 함
        for spec in specs:
            if spec.pred not in df.columns:
                continue
            rel = ((df[spec.pred] - df["True_Target"]) / df.attrs["capacity"]).dropna()
            if rel.empty:
                continue
            z = gaussian_kde(rel)(x)
            max_z = max(max_z, z.max())

            if FILL:
                verts.append([(x[0], 0)] + list(zip(x, z)) + [(x[-1], 0)])
                face_colors.append(mcolors.to_rgba(spec.color, alpha=ALPHA))
                edge_colors.append(mcolors.to_rgba(spec.color, alpha=1.0))
                zs.append(idx)
            else:
                # 면을 칠하지 않고 윤곽선만 → 겹쳐도 뒤쪽 곡선이 보임
                ax.plot(x, z, zs=idx, zdir="y", color=spec.color,
                        linestyle=RIDGE_LINESTYLE.get(spec.key, "-"),
                        linewidth=LINEWIDTH, solid_capstyle="round")

    if FILL:
        poly = PolyCollection(verts, facecolors=face_colors,
                              edgecolors=edge_colors, linewidths=0.6)
        ax.add_collection3d(poly, zs=np.array(zs), zdir="y")

    ax.set_xlim(*XLIM)
    ax.set_ylim(0, len(sites) - 1)
    ax.set_zlim(0, max_z * 1.1)
    ax.set_xlabel(r"Relative Error (of $P_{\max}$)")
    ax.set_ylabel("Site")
    ax.set_zlabel("Probability Density")
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([str(s) for s in sites])

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("white")
    ax.grid(True)
    grid_style = {"color": "gray", "linestyle": "--", "linewidth": 0.5, "alpha": 0.3}
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(grid_style)
    ax.view_init(elev=25, azim=-50)
    # 3D 축은 기본 배율이 작아 캔버스에 흰 여백을 많이 남깁니다. zoom 으로 채웁니다.
    ax.set_box_aspect((4, 3, 2), zoom=1.08)

    if FILL:
        handles = [Patch(facecolor=mcolors.to_rgba(s.color, ALPHA),
                         edgecolor=s.color, label=s.label) for s in specs]
    else:
        handles = [Line2D([], [], color=s.color, linewidth=LINEWIDTH,
                          linestyle=RIDGE_LINESTYLE.get(s.key, "-"), label=s.label)
                   for s in specs]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 0.95),
              frameon=False)

    # 3D 축은 박스 안쪽에 여백을 많이 남깁니다. 예전에는 bbox_inches="tight" 로
    # 잘라냈지만, 그러면 저장 폭이 figsize 와 달라져 원고에서 확대 배율이 끼어듭니다.
    # 대신 축을 캔버스 가득 키워 규격 폭(FULL_W)으로 그대로 저장합니다.
    fig.subplots_adjust(left=0.02, right=0.98, top=1.0, bottom=0.02)
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "7.png"), dpi=300)
    plt.close(fig)
    print("saved: 7.png")


if __name__ == "__main__":
    main()
