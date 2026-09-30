import os
import pandas as pd
import matplotlib.pyplot as plt

# 설정 파일 임포트
import config

# 사용할 색상 추출
palette = config.get_colors(8, cmap_name="clist")
color_base = palette[1]  # 하늘색 (Sky Blue)
color_residual = palette[2]  # 빨강 (Muted Red)

# best_model_combos = [
#     ('svr', 'svr'),
#     ('lgb', 'svr'),
#     ('mlr', 'svr'),
#     ('mlr', 'svr'),
#     ('mlr', 'svr'),
#     ('mlr', 'svr'),
#     ('lgb', 'svr')
# ]
best_model_combos = [
    ('mlr', 'svr'),
    ('mlr', 'svr')
]


data_path = config.DATA_ROOT + '/reforecast_results'
# sites = [1, 2, 4, 5, 6, 7, 8]
sites = [4, 6]

# [데이터 처리 일치] 데이터를 분할하기 위한 train_size 설정
train_size = 22 / 24

fig, axes = plt.subplots(2, 1, figsize=(config.SINGLE_COL, config.SINGLE_COL), sharex=True)

for i, ax in enumerate(axes):
    site_index = i
    site_num = sites[site_index]

    best_forecast_model, best_reforecast_model = best_model_combos[i]

    # [데이터 처리 일치] CSV 파일을 읽어온 뒤, split_index를 기준으로 뒷부분(Test)만 잘라냅니다.
    file_path = os.path.join(data_path, f'best/final_result_{best_forecast_model}_{site_num}_.csv')
    df = pd.read_csv(file_path)
    split_index = int(len(df) * train_size)
    df = df.iloc[split_index:].copy()

    df['Time'] = pd.to_datetime(df['Time'])
    df['hour_min'] = df['Time'].dt.strftime('%H:%M')

    # 잔차(Residual) 계산
    df['residual_forecast'] = df[f'pred_{best_forecast_model}'] - df['Power (MW)']
    df['residual_reforecast'] = df[f'reforecasted_PV_{best_reforecast_model}'] - df['Power (MW)']

    # 시간대별 평균 및 표준편차 계산
    agg = df.groupby('hour_min')[['residual_forecast', 'residual_reforecast']].agg(['mean', 'std'])
    agg = agg.sort_index()

    mean_forecast = agg['residual_forecast']['mean']
    std_forecast = agg['residual_forecast']['std']
    mean_reforecast = agg['residual_reforecast']['mean']
    std_reforecast = agg['residual_reforecast']['std']

    x = list(range(len(mean_forecast)))

    # ① Forecast (Base model) - 하늘색
    ax.plot(x, mean_forecast, color=color_base, label='Base model', linewidth=1.0, linestyle='--')
    ax.fill_between(x,
                    mean_forecast - std_forecast,
                    mean_forecast + std_forecast,
                    color=color_base,
                    alpha=0.3,  # 투명도를 낮춰 겹치는 부분 시야 확보
                    edgecolor='none')

    # ② Reforecast (Residual model) - 빨강
    ax.plot(x, mean_reforecast, color=color_residual, label='Residual model', linewidth=1.0, linestyle='-')
    ax.fill_between(x,
                    mean_reforecast - std_reforecast,
                    mean_reforecast + std_reforecast,
                    color=color_residual,
                    alpha=0.4,
                    edgecolor='none')

    # ③ 기준선 (오차 0 지점)
    # 데이터를 가리지 않도록 선 굵기를 0.5로 매우 얇게, 색상은 연한 회색/검정 점선으로 처리
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.6)

    # Y축 라벨 설정
    ax.set_ylabel('Error [MW]')

    # 4. x축 눈금 처리 로직 (맨 마지막 서브플롯인 i == 6 일 때만 라벨 설정)
    if i == len(sites) - 1:
        index_list = list(agg.index)
        even_positions = []
        even_labels = []
        seen_hours = set()

        for idx, label in enumerate(index_list):
            hour = int(label[:2])
            if hour % 2 == 0 and hour not in seen_hours:
                even_positions.append(idx)
                even_labels.append(str(hour))
                seen_hours.add(hour)

        ax.set_xticks(even_positions)
        ax.set_xticklabels(even_labels, ha='center')
        ax.set_xlabel('Hour')

    # Nature/Science 스타일 눈금 적용
    config.set_nature_ticks(ax)

# =============================================================================
# 루프 종료 후 전체 레이아웃 설정
# =============================================================================

# 패널 라벨 (A~G) 자동 추가
config.label_panels(axes, uppercase=True)

# 통합 범례 추가 (맨 아래 중앙 배치, 가로 폭이 좁으므로 ncol=2로 설정하여 줄바꿈 유도)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)

# 레이아웃 조정
plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.5)

# 저장 및 출력
save_path = os.path.join(config.SAVE_DIR_MAIN, "6_b.png")
plt.savefig(save_path, dpi=300)

plt.show()