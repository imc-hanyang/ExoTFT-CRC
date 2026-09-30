import os.path
import pandas as pd
import matplotlib.pyplot as plt

# 우리가 만든 설정 파일 임포트 (폰트, 기본 스타일 자동 적용)
import config

# 사용할 색상 추출
palette = config.get_colors(8, cmap_name="clist")
color_base = palette[1]  # 하늘색 (Sky Blue)
color_residual = palette[2]  # 빨강 (Muted Red)
color_actual = palette[7]  # 회색 (Grey)

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

# 1. Figure 생성: 가로 너비를 절반(SINGLE_COL)으로 축소
# 세로가 찌그러지지 않도록 높이 비율을 기존 1.4에서 2.8로 증가시켜 절대적 높이 유지
fig, axes = plt.subplots(2, 1, figsize=(config.SINGLE_COL, config.SINGLE_COL), sharex=True)

# 사이트 루프 시작
for i, ax in enumerate(axes):
    site_num = sites[i]
    best_forecast_model, best_reforecast_model = best_model_combos[i]

    # csv 파일 로드
    df = pd.read_csv(os.path.join(data_path, f'final_result_{best_forecast_model}_{site_num}_.csv'))

    # ▶ 선택 날짜만 필터링
    selected_date = '2020-12-24'
    df['Time'] = pd.to_datetime(df['Time'])
    df = df[df['Time'].dt.date == pd.to_datetime(selected_date).date()]
    df['hour_min'] = df['Time'].dt.strftime('%H:%M')

    # ▶ 시간(HH:MM)별 평균
    hourly_avg = df.groupby('hour_min')[
        ['Power (MW)', f'pred_{best_forecast_model}', f'reforecasted_PV_{best_reforecast_model}']
    ].mean()

    # ① Measurement (Actual) - 회색 배경 형광펜 효과
    ax.plot(hourly_avg.index,
            hourly_avg['Power (MW)'],
            label='Actual',
            color=color_actual,
            linestyle='-',
            linewidth=0.7,
            alpha=0.3,
            zorder=1)

    # ② Forecast (Base) - 하늘색
    ax.plot(hourly_avg.index,
            hourly_avg[f'pred_{best_forecast_model}'],
            label='Base model',
            color=color_base,
            linestyle='--',
            linewidth=0.5,
            marker='o',
            markersize=0.3,
            zorder=2)

    # ③ Reforecast (Residual) - 빨강
    ax.plot(hourly_avg.index,
            hourly_avg[f'reforecasted_PV_{best_reforecast_model}'],
            label='Residual model',
            color=color_residual,
            linestyle=':',
            linewidth=0.5,
            marker='s',
            markersize=0.3,
            zorder=3)

    # Y축 라벨 및 범위 설정
    ax.set_ylabel('Power [MW]')
    ax.set_ylim(bottom=0)

    # 각 그래프 좌측 상단에 사이트 번호 표기
    # 3. x축 눈금 처리 로직 (맨 마지막 서브플롯인 i == 6 일 때만 라벨 설정)
    if i == len(sites) - 1:
        index_list = list(hourly_avg.index)
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

# 통합 범례 추가 (그림 전체 맨 아래 중앙에 1개만 배치)
handles, labels = axes[0].get_legend_handles_labels()
# 가로가 좁아졌으므로 범례가 너무 길어지지 않게 ncol을 3에서 2 또는 1로 줄일 수도 있지만,
# 텍스트 길이를 고려해 일단 3을 유지하되 글씨가 겹치면 ncol=2로 수정하는 것을 권장합니다.
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False)

# 레이아웃 조정
plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=0.5)

# 저장 및 출력
save_path = os.path.join(config.SAVE_DIR_MAIN, "6_a.png")
plt.savefig(save_path, dpi=300)
plt.show()