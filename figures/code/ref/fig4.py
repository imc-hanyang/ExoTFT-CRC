import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 우리가 만든 설정 파일 임포트 (자동으로 기본 폰트, 선 굵기 등 적용됨)
import config

data_path = config.DATA_ROOT + '/reforecast_forecast_eval_results'

df_forecast = pd.read_excel(os.path.join(data_path, 'forecast_results.xlsx'))
df_reforecast = pd.read_excel(os.path.join(data_path, 'reforecast_results.xlsx'))

sites = [1, 2, 4, 5, 6, 7, 8]
models = ['MLR', 'SVR', 'LGB', 'MLP']

b0_prev_legend_name = 'BO: Current Observation'
b1_forecast_legend_name = 'B1: Base Forecast'
reforecast_legend_name = 'Proposed: Error-Corrected Forecast'

best_model_combos = [
    ('svr', 'svr'),
    ('lgb', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('lgb', 'svr')
]

compare_with_prev_15min_df = pd.read_csv(os.path.join(data_path,'lag1_pred_results.csv'))

fig, axes = plt.subplots(3, 1, figsize=(config.DOUBLE_COL, config.DOUBLE_COL * 1.2))

errors = ['NMAE', 'NRMSE', 'R^2']
# units = {'NAME': 'MW', 'MSE': 'MW²', 'RMSE': 'MW'}

# 2. 색상 팔레트 적용: 하드코딩된 red/blue/green 대신 config의 색약자 친화 팔레트 3개 사용
palette = config.get_colors(3, cmap_name="clist")
colors = {
    b0_prev_legend_name: palette[0],
    b1_forecast_legend_name: palette[1],
    reforecast_legend_name: palette[2]
}

bar_width = 0.25  # 세 막대가 겹치지 않도록 너비 설정

for idx, graph_error in enumerate(errors):
    site_labels = []
    compare_vals = []
    forecast_vals = []
    reforecast_vals = []

    for site_num, (best_forecast_model, best_reforecast_model) in zip(sites, best_model_combos):
        prev_pv_diff = compare_with_prev_15min_df.loc[
            compare_with_prev_15min_df['site'] == site_num, graph_error
        ].values[0]

        forecast_error = df_forecast[
            (df_forecast['site'] == site_num) &
            (df_forecast['forecast'] == best_forecast_model.upper())
            ][graph_error].iloc[0]

        reforecast_error = df_reforecast[
            (df_reforecast['site'] == site_num) &
            (df_reforecast['forecast'] == best_forecast_model.upper()) &
            (df_reforecast['reforecast'] == best_reforecast_model.upper())
            ][graph_error].iloc[0]

        # print(f"Site {site_num} | Forecast: {best_forecast_model.upper()} | Reforecast: {best_reforecast_model.upper()} | {graph_error}: {reforecast_error:.4f}")

        site_labels.append(f'Site {site_num}')
        compare_vals.append(prev_pv_diff)
        forecast_vals.append(forecast_error)
        reforecast_vals.append(reforecast_error)

    compare_vals = np.array(compare_vals)
    forecast_vals = np.array(forecast_vals)
    reforecast_vals = np.array(reforecast_vals)
    x = np.arange(len(site_labels))
    ax = axes[idx]

    # 막대 테두리를 얇게(linewidth=0.5) 주고 검은색(edgecolor)을 추가해 형태를 명확히 함
    for i in range(len(x)):
        ax.bar(x[i] - bar_width, compare_vals[i], width=bar_width, color=colors[b0_prev_legend_name],
               edgecolor='black', linewidth=0.5, label=b0_prev_legend_name if i == 0 else "")
        ax.bar(x[i], forecast_vals[i], width=bar_width, color=colors[b1_forecast_legend_name],
               edgecolor='black', linewidth=0.5, label=b1_forecast_legend_name if i == 0 else "")
        ax.bar(x[i] + bar_width, reforecast_vals[i], width=bar_width, color=colors[reforecast_legend_name],
               edgecolor='black', linewidth=0.5, label=reforecast_legend_name if i == 0 else "")

    # 3. 라벨 설정 (fontsize 하드코딩 제거)
    ax.set_xticks(x)
    ax.set_xticklabels(site_labels)
    ax.set_ylabel(f'{graph_error}')
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # 🚨 ax.grid() 제거 (저널 규격상 금지)

    # 4. Nature/Science 스타일 눈금 적용
    config.set_nature_ticks(ax)

# 5. (선택사항) 논문용 패널 라벨 (A, B, C) 자동 추가
config.label_panels(axes, uppercase=True)

# 6. 범례 및 레이아웃 설정 (fontsize 하드코딩 제거)
fig.legend(
    [b0_prev_legend_name, b1_forecast_legend_name, reforecast_legend_name],
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, 0.0),
    frameon=False  # 테두리 없음
)

# 하단 범례가 잘리지 않도록 여백 조정
fig.tight_layout()

save_path = os.path.join(config.SAVE_DIR_MAIN, "4.png")
plt.savefig(save_path)
plt.show()