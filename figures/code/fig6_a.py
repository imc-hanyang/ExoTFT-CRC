"""
Fig 6-a. 특정 하루의 실측 vs 모델별 예측 프로파일.

SITES / SELECTED_DATE 만 바꾸면 됩니다.
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
SELECTED_DATE = "2020-11-11"  # 테스트 구간(2020-11-01 ~ 2020-12-31) 내 날짜
COLOR_ACTUAL = config.clist[7]


def draw(sites, suffix=""):
    fig, axes, bottom_axes = config.site_panel_axes(len(sites),
                                                    width=config.HALF_W)

    for ax, site in zip(axes, sites):
        df = loader.load_site(site, order=LOAD_ORDER, verbose=False)
        day = df[df["Time"].dt.date == __import__("pandas").to_datetime(SELECTED_DATE).date()]
        if day.empty:
            ax.text(0.5, 0.5, f"no data on {SELECTED_DATE}", ha="center",
                    transform=ax.transAxes)
            continue
        x = day["Time"].dt.strftime("%H:%M")

        # ① 실측 — 회색 형광펜 효과
        ax.plot(x, day["True_Target"], label="Actual", color=COLOR_ACTUAL,
                linewidth=1.6, alpha=0.35, zorder=1)

        # ② 모델별 예측
        for spec in loader.available_specs(df):
            ax.plot(x, day[spec.pred], label=spec.label, color=spec.color,
                    linestyle=spec.linestyle, linewidth=0.8, zorder=spec.zorder)

        ax.set_ylabel("Power [MW]")
        ax.set_ylim(bottom=0)
        config.set_nature_ticks(ax)

        if ax in bottom_axes:
            idx = list(x)
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

    # 범례가 차지할 높이만큼만 아래를 비웁니다. 최종 모델명이 ExoTFT-CRC 로
    # 짧아져 90 mm 폭에 3항목 1줄이 들어갑니다(핸들·열 간격은 줄여 둡니다).
    legend_in = 0.22 + 0.08              # 1줄
    plt.tight_layout(pad=0.25, rect=[0, legend_in / fig.get_figheight(), 1, 1],
                     h_pad=0.5)
    # 범례는 레이아웃 확정 후, 축 박스 기준 가운데에 배치
    config.legend_below_axes(fig, axes, ncol=3, handlelength=1.1,
                             handletextpad=0.35, columnspacing=0.9)
    out = f"6_a{suffix}.png"
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, out), dpi=300)
    plt.close(fig)
    print(f"saved: {out}")


def main():
    for sites, suffix in SITE_SETS:
        draw(sites, suffix)


if __name__ == "__main__":
    main()
