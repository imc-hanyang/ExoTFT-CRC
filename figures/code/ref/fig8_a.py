import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import time

# 설정 파일 임포트
import config

# 데이터 경로 및 변수 설정
train_size = 22 / 24
reforecast_data_path = config.DATA_ROOT + '/reforecast_results'
forecast_data_path = config.DATA_ROOT + '/forecast_results'

best_model_combos = [
    ('svr', 'svr'),
    ('lgb', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('lgb', 'svr')
]
operation_hours_array = [
    ('06:00', '21:30'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('06:00', '21:00'),
    ('06:00', '21:00'),
    ('06:00', '19:00')
]
sites = [1, 2, 4, 5, 6, 7, 8]

# 1. Figure 생성: 가로 너비 SINGLE_COL, 세로 7칸 이어붙이기
fig, axes = plt.subplots(7, 1, figsize=(config.SINGLE_COL, config.SINGLE_COL * 4.0), sharex=False)

for ax, (forecast_model, reforecast_model), (start_str, end_str), site_num in zip(
        axes, best_model_combos, operation_hours_array, sites
):
    # 파일 경로 생성 및 로드
    reforecast_file = os.path.join(reforecast_data_path, f"final_result_{forecast_model}_{site_num}_.csv")
    forecast_22mon_file = os.path.join(forecast_data_path, f'{str(site_num)}_test.csv')

    reforecast_df = pd.read_csv(reforecast_file)
    forecast_22mon_df = pd.read_csv(forecast_22mon_file)
    df_list = [reforecast_df, forecast_22mon_df]

    # 데이터 전처리
    for i in range(2):
        split_index = int(len(df_list[i]) * train_size)
        df_list[i] = df_list[i].iloc[:split_index].copy()

        df = df_list[i]
        df['Time'] = pd.to_datetime(df['Time'])
        start_time, end_time = time.fromisoformat(start_str), time.fromisoformat(end_str)
        df_list[i] = df[df['Time'].dt.time.between(start_time, end_time)]

    # 데이터 병합
    merged = pd.merge(df_list[0][['Time', 'Power (MW)', f'pred_error_{reforecast_model}']],
                      df_list[1][['Time', f'pred_{forecast_model}']], on='Time', how='inner').dropna()

    # 계산
    slope, intercept = np.polyfit(merged['Power (MW)'], merged[f'pred_{forecast_model}'], 1)
    adjusted_y_pred = slope * merged['Power (MW)'] + intercept + merged[f'pred_error_{reforecast_model}']

    # 2. 각 사이트별 축 범위 설정 (y=x 45도 유지를 위해 x, y 범위를 동일하게)
    all_vals = np.concatenate([merged['Power (MW)'], adjusted_y_pred])
    lower_bound = all_vals.min() * 0.95
    upper_bound = all_vals.max() * 1.05
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)

    # 산점도 및 선 플롯
    ax.plot(merged['Power (MW)'], adjusted_y_pred, 'o', markersize=1.5, color='black', alpha=0.3, label='Reforecasted')
    ax.plot(merged['Power (MW)'], slope * merged['Power (MW)'] + intercept, '-', color='green', linewidth=1.0,
            label='Base trend')

    # y=x 기준선
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'r-', linewidth=1.0, label='y=x')

    # 스타일 적용
    ax.text(0.05, 0.9, f'Site {site_num}', transform=ax.transAxes, fontweight='bold')
    config.set_nature_ticks(ax)

    # Y축 라벨
    ax.set_ylabel('Predicted Power')
    ax.set_xlabel('Actual Power')

# 3. 레이아웃 및 범례
config.label_panels(axes, uppercase=True)

# 범례 설정: ncol=3으로 원복
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1], h_pad=0.5)
plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "8_a.png"), dpi=300)
plt.show()