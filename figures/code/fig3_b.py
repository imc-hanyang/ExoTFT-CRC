"""Fig 3-b. 사이트별 발전 시각 분포 (Power > 0 인 시각의 boxplot)."""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
import loader

# 폭: 이 그림은 원고에서 **두 개가 가로로 나란히** 들어가므로 config.HALF_W
# (= 원고 \linewidth 의 절반)로 저장합니다. width=0.5\linewidth 로 넣으면 배율 1.0 이라
# config 의 pt 가 그대로 인쇄됩니다.
config.figure_config()


def minutes_to_hhmm(x, pos):
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


def main():
    sites = config.sites
    colors = config.get_colors(len(sites), cmap_name="clist")

    box_data = []
    for site in sites:
        df = loader.load_origin(site, add_derived=False)
        gen = df[df[config.POWER_COL] > 0]
        box_data.append(gen["Time"].dt.hour * 60 + gen["Time"].dt.minute)

    fig, ax = plt.subplots(figsize=(config.HALF_W, config.HALF_W * 0.78))
    bplot = ax.boxplot(box_data, tick_labels=[str(s) for s in sites], patch_artist=True,
                       boxprops=dict(linewidth=0.8),
                       whiskerprops=dict(linewidth=0.8),
                       capprops=dict(linewidth=0.8),
                       flierprops=dict(marker="o", markersize=0.8, alpha=0.2,
                                       markeredgewidth=0.0, markerfacecolor="grey"),
                       medianprops=dict(linewidth=1.0, color="black"))

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Site")
    ax.set_ylabel("Time of Day (HH:MM)")
    ax.set_ylim(0, 1440)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(180))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(minutes_to_hhmm))
    config.set_nature_ticks(ax)

    plt.tight_layout(pad=0.25)
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "3_b.png"))
    plt.close(fig)
    print("saved: 3_b.png")


if __name__ == "__main__":
    main()
