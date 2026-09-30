import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm, ListedColormap
import config  # 설정 파일


def process_time(df):
    df['Time'] = pd.to_datetime(df['Time'])
    df['time_index'] = df['Time'].dt.hour * 4 + df['Time'].dt.minute // 15
    df['dayofyear'] = df['Time'].dt.dayofyear
    return df


reforecast_data_path = config.DATA_ROOT + '/reforecast_results'
best_model_combos = [
    ('svr', 'svr'), ('lgb', 'svr'), ('mlr', 'svr'), ('mlr', 'svr'),
    ('mlr', 'svr'), ('mlr', 'svr'), ('lgb', 'svr')
]
sites = [1, 2, 4, 5, 6, 7, 8]

# 1. Figure 생성
fig, axes = plt.subplots(7, 1, figsize=(config.SINGLE_COL, config.SINGLE_COL * 4.0), sharex=False)

last_cmap, last_norm = None, None

# 2. 루프를 통한 플로팅
for i, ax in enumerate(axes):
    site_num = sites[i]
    forecast_model = best_model_combos[i][0]
    target = f'error_{forecast_model}'

    # 데이터 로드 및 전처리 (첫 번째 코드와 동일한 로직)
    df = process_time(
        pd.read_csv(os.path.join(config.DATA_ROOT,
                                 f"forecast_results/residuals/final_result_{forecast_model}_{site_num}_.csv")).copy())
    pivot = df.groupby(['dayofyear', 'time_index'])[target].mean().reset_index()

    time = pivot['time_index']
    doy = pivot['dayofyear']
    residual = pivot[target].mul(-1)

    # [데이터 처리 일치] 격자 데이터 생성 및 선형 보간 (fill_value=np.nan 추가)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 95, 96), np.linspace(1, 365, 365))
    grid_z = griddata((time, doy), residual, (grid_x, grid_y), method='linear', fill_value=np.nan)

    # [데이터 처리 일치] abs_max 기반 levels 및 step 설정 (step = 0.5 고정)
    abs_max = np.nanmax(np.abs(grid_z))
    step = 0.5
    bound = np.ceil(abs_max / step) * step
    levels = np.arange(-bound, bound + step, step)

    # [데이터 처리 일치] 컬러맵 정중앙을 흰색으로 지정하는 인덱스 수정
    colors = plt.cm.seismic(np.linspace(0, 1, len(levels) - 1))
    colors[len(colors) // 2] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    if i == 6:
        last_cmap, last_norm = cmap, norm

    # [데이터 처리 일치] extend='both' 제거하여 첫 번째 코드와 완벽히 동일하게 플로팅
    ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm)
    config.set_nature_ticks(ax)
    ax.set_ylabel("Day of Year")

# 3. 공통 축 설정
axes[-1].set_xticks([0, 24, 48, 72, 95])
axes[-1].set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:45'])
axes[-1].set_xlabel("Time")
config.label_panels(axes, fontsize=8)

# 4. 컬러바 배치 (Low/0/High가 위, Residuals [MW]가 그 아래)
plt.tight_layout(rect=[0.05, 0.11, 1, 1], h_pad=0.5)

cbar_ax = fig.add_axes([0.17, 0.07, 0.75, 0.012])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=last_norm, cmap=last_cmap),
                    cax=cbar_ax, orientation='horizontal')

cbar.set_ticks([])  # 눈금 제거

# 컬러바 텍스트 및 레이블 설정
cbar.ax.text(0.5, -1.5, '0', ha='center', va='top', fontsize=8, transform=cbar.ax.transAxes)
cbar.ax.text(0.0, -1.5, 'Low', ha='center', va='top', fontsize=8, transform=cbar.ax.transAxes)
cbar.ax.text(1.0, -1.5, 'High', ha='center', va='top', fontsize=8, transform=cbar.ax.transAxes)
cbar.set_label('Residuals [MW]', fontsize=9, labelpad=25)

plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "8_b.png"), dpi=300)
plt.show()