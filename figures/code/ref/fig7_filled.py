import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors

# 설정 파일 임포트
import config

forecast_models = ['svr', 'lgb', 'mlr', 'mlr', 'mlr', 'mlr', 'lgb']
files = []
data_path = config.DATA_ROOT + '/reforecast_results'
sites = [1, 2, 4, 5, 6, 7, 8]

for site, forecast_model in zip(sites, forecast_models):
    files.append(os.path.join(data_path, f'final_result_{forecast_model}_{site}_.csv'))

df_list = [pd.read_csv(file) for file in files]

# 1. Figure 생성
fig = plt.figure(figsize=(config.DOUBLE_COL, config.DOUBLE_COL * 0.75))
ax = fig.add_subplot(111, projection='3d')

colors = config.get_colors(len(sites), cmap_name="clist")

verts = []
face_colors = []
y_ticks = []
y_labels = []
max_z = 0

# 2. 데이터 처리 및 다각형 버텍스(Vertex) 생성
for idx, (df, site, forecast_model) in enumerate(zip(df_list, sites, forecast_models)):
    data = df[f'error_{forecast_model}'].dropna().mul(-1)
    kde = gaussian_kde(data)

    x = np.linspace(-1.2, 1.2, 200)
    z = kde(x)

    # Z축 최대값 갱신 (나중에 축 범위 설정을 위해)
    if z.max() > max_z:
        max_z = z.max()

    # 면적을 칠하기 위해 다각형의 시작과 끝을 바닥(z=0)으로 닫아줌
    vertex = [(x[0], 0)] + list(zip(x, z)) + [(x[-1], 0)]
    verts.append(vertex)

    # 색상에 투명도(Alpha) 추가
    rgba = mcolors.to_rgba(colors[idx], alpha=0.7)
    face_colors.append(rgba)

    y_ticks.append(idx)
    y_labels.append(str(site))

# PolyCollection 생성 (면적 채우기 및 테두리 선)
poly = PolyCollection(verts, facecolors=face_colors, edgecolors='black', linewidths=0.5)

# 3D 축에 추가 (y축을 기준으로 배치)
ax.add_collection3d(poly, zs=np.arange(len(sites)), zdir='y')

# 3. 축 한계값 및 라벨 설정 (add_collection3d 사용 시 x, y, z lim 수동 설정 필수)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(0, len(sites) - 1)
ax.set_zlim(0, max_z * 1.1)  # 최대값에 약간의 여유(10%)를 둠

ax.set_xlabel('Relative error')
ax.set_ylabel('Site index')
ax.set_zlabel('Probability density')

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# 4. 3D 그래프 스타일 클리닝
# 4. 3D 그래프 스타일 클리닝
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# 격자선(Grid) 켜고 약하게 설정하기
ax.grid(True)

# 3D 축의 격자선 스타일을 옅게 커스터마이징 (회색, 점선, 얇은 두께, 투명도 30%)
grid_style = {"color": "gray", "linestyle": "--", "linewidth": 0.5, "alpha": 0.3}
ax.xaxis._axinfo["grid"].update(grid_style)
ax.yaxis._axinfo["grid"].update(grid_style)
ax.zaxis._axinfo["grid"].update(grid_style)

# 카메라 시점 조정
ax.view_init(elev=25, azim=-50)

# plt.tight_layout()
fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)

# 저장 및 출력
save_path = os.path.join(config.SAVE_DIR_MAIN, "7.png")
# plt.savefig(save_path, dpi=300)
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()