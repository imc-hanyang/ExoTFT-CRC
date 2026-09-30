import os
import pandas as pd
import matplotlib.pyplot as plt

import config


def main():
    site_numbers = [1, 2, 4, 5, 6, 7, 8]

    colors = config.get_colors(len(site_numbers), cmap_name="clist")

    fig, ax = plt.subplots(figsize=(config.SINGLE_COL, config.SINGLE_COL * 0.75))

    TIME_COL = 'Time(year-month-day h:m:s)'
    POWER_COL = 'Power (MW)'

    for i, site in enumerate(site_numbers):
        filename = f"site{site}.xlsx"
        filepath = os.path.join(config.DATA_ROOT+'/solar_stations', filename)

        try:
            df = pd.read_excel(filepath)

            df[TIME_COL] = pd.to_datetime(df[TIME_COL])

            df['Hour'] = df[TIME_COL].dt.hour

            hourly_avg = df.groupby('Hour')[POWER_COL].mean()

            x = hourly_avg.index
            y = hourly_avg.values

            ax.plot(x, y, color=colors[i], label=f'Site {site}', linewidth=1.0)

        except FileNotFoundError:
            print(f"경고: {filepath} 파일을 찾을 수 없어 그래프에서 제외합니다.")
        except KeyError as e:
            print(f"경고: 데이터프레임에서 {e} 컬럼을 찾을 수 없습니다. (TIME_COL, POWER_COL 변수명을 확인하세요.)")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Power [MW]")

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4))

    config.set_nature_ticks(ax)

    ax.legend(loc='upper right')

    plt.tight_layout()

    save_path = os.path.join(config.SAVE_DIR_MAIN, "3_a.png")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()