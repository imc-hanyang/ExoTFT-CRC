import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 1. 만들어두신 config 파일 불러오기 (내부 rcParams 자동 세팅됨)
import config

# 2. 데이터 로드
data_path = config.DATA_ROOT + '/shap'
df = pd.read_csv(os.path.join(data_path, 'shap_reforecast.csv'))

# 3. Y축(Feature) 공유를 위한 전체 평균 랭크 계산 및 정렬 기준 생성
feature_order = df.groupby('feature')['importance_rank'].mean().sort_values().index

# 4. 세 가지 기준(site, forecast_model, reforecast_model)의 피벗 테이블 생성
pivots = {
    'Site': df.groupby(['feature', 'site'])['importance_rank'].mean().unstack(level='site').reindex(feature_order),
    'Base Forecasting Model': df.groupby(['feature', 'forecast_model'])['importance_rank'].mean().unstack(
        level='forecast_model').reindex(feature_order),
    'Residual Forecasting Model': df.groupby(['feature', 'reforecast_model'])['importance_rank'].mean().unstack(
        level='reforecast_model').reindex(feature_order)
}

# 5. Figure 설정 (★ 세로 길이를 7.5로 추가 축소, layout='constrained' 유지)
fig, axes = plt.subplots(1, 3, figsize=(config.DOUBLE_COL, 7.5),
                         sharey=True, layout='constrained')

# 6. 컬러맵 설정
cmap = config.get_cmap("RdBu")

# 7. 3개의 히트맵을 반복문으로 그리기
for i, (title, pivot) in enumerate(pivots.items()):
    ax = axes[i]

    # 개별 히트맵 안의 cbar는 모두 False로 꺼둡니다.
    heatmap = sns.heatmap(pivot,
                          cmap=cmap,
                          ax=ax,
                          cbar=False,
                          linewidths=0.5,
                          linecolor='lightgray')

    # 축 및 틱 스타일 지정
    config.set_nature_ticks(ax)

    # 제목 및 X/Y축 라벨 설정
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("")

    if i == 0:
        ax.set_ylabel("Feature", fontweight='bold')
    else:
        ax.set_ylabel("")
        # Y축이 공유되므로 2, 3번째 그래프는 y축 틱(선)을 안보이게 숨김
        ax.tick_params(axis='y', which='both', length=0)

# 8. 공통 컬러바 추가
cb = fig.colorbar(axes[2].collections[0], ax=axes, shrink=0.8, pad=0.02, aspect=40)
cb.ax.invert_yaxis()
cb.ax.yaxis.set_major_locator(MultipleLocator(5))
cb.set_label('Average Importance Rank')
config.thin_spines(cb.ax)

# 9. 레이아웃 조정 및 저장
save_path = os.path.join(config.SAVE_DIR, "9.png")

# ★ 왼쪽 라벨이 잘리지 않도록 bbox_inches='tight' 옵션 추가
plt.savefig(save_path, bbox_inches='tight')
plt.show()