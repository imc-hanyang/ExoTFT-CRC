import os
from idlelib.browser import file_open

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import config

df_list = []
sites = [1, 2, 4, 5, 6, 7, 8]

for site in sites:
  filename = f"site{site}.xlsx"
  file_path = os.path.join(config.DATA_ROOT + '/solar_stations', filename)

  df = pd.read_excel(file_path)

  copy_df = df[['Power (MW)', 'Time(year-month-day h:m:s)']].copy()
  copy_df['prev_Power'] = copy_df['Power (MW)'].shift(1)
  df_list.append(df)

box_data = []

for i, df in enumerate(df_list):
    df_filtered = df[df['Power (MW)'] != 0].copy()
    df_filtered['Time'] = pd.to_datetime(df_filtered['Time(year-month-day h:m:s)'])

    minutes = df_filtered['Time'].dt.hour * 60 + df_filtered['Time'].dt.minute
    box_data.append(minutes)

def minutes_to_hhmm(x, pos):
    h = int(x // 60)
    m = int(x % 60)
    return f"{h:02d}:{m:02d}"

fig, ax = plt.subplots(figsize=(config.SINGLE_COL, config.SINGLE_COL * 0.8))

colors = config.get_colors(len(sites), cmap_name="clist")

bplot = ax.boxplot(box_data, tick_labels='1245678', patch_artist=True,
                   boxprops=dict(linewidth=0.8),       # 선 굵기 저널 규격화
                   whiskerprops=dict(linewidth=0.8),
                   capprops=dict(linewidth=0.8),
                   medianprops=dict(linewidth=1.0, color='black'))

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Site')
ax.set_ylabel('Time of Day (HH:MM)')

ax.set_ylim(bottom=0, top=1440)
ax.yaxis.set_major_locator(ticker.MultipleLocator(180))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(minutes_to_hhmm))

config.set_nature_ticks(ax)

plt.tight_layout()

save_path = os.path.join(config.SAVE_DIR_MAIN, "3_b.png")
plt.savefig(save_path)
plt.show()
