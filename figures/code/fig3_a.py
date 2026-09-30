"""Fig 3-a. 사이트별 시간대 평균 발전량 (원본 전 기간)."""
import os
import matplotlib.pyplot as plt

import config
import loader

# 폭: 이 그림은 원고에서 **두 개가 가로로 나란히** 들어가므로 config.HALF_W
# (= 원고 \linewidth 의 절반)로 저장합니다. width=0.5\linewidth 로 넣으면 배율 1.0 이라
# config 의 pt 가 그대로 인쇄됩니다.
config.figure_config()


def main():
    sites = config.sites
    colors = config.get_colors(len(sites), cmap_name="clist")

    fig, ax = plt.subplots(figsize=(config.HALF_W, config.HALF_W * 0.78))

    for i, site in enumerate(sites):
        df = loader.load_origin(site, add_derived=False)
        # 설비용량으로 정규화하지 않은 원 스케일 (사이트 간 용량 차이 그대로 표현)
        hourly_avg = df.groupby(df["Time"].dt.hour)[config.POWER_COL].mean()
        ax.plot(hourly_avg.index, hourly_avg.values,
                color=colors[i], label=f"Site {site}", linewidth=1.0)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Power [MW]")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4))
    config.set_nature_ticks(ax)
    # 오른쪽에 세로 1열로 나열
    ax.legend(loc="upper right", ncol=1)

    plt.tight_layout(pad=0.25)
    plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "3_a.png"))
    plt.close(fig)
    print("saved: 3_a.png")


if __name__ == "__main__":
    main()
