import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 만들어두신 config 파일 불러오기
import config

# 2. 데이터 로드 (경로는 기존 코드 환경에 맞게 유지)
data_path = config.DATA_ROOT + '/shap'
df_base = pd.read_csv(os.path.join(data_path, 'forecast_shap.csv'))
df_resid = pd.read_csv(os.path.join(data_path, 'shap_reforecast.csv'))

# 3. 사이트 및 피처별로 중요도 순위 평균 집계
base_agg = df_base.groupby(['feature', 'site'])['importance_rank'].mean().reset_index()
resid_agg = df_resid.groupby(['feature', 'site'])['importance_rank'].mean().reset_index()

# 4. 데이터 병합 및 차이(Difference) 계산
# 차이 = Residual 순위 - Base 순위
# - Residual에서 더 중요하면 음수(-) -> RdBu 컬러맵에서 빨강(Red)
# - Base에서 더 중요하면 양수(+) -> RdBu 컬러맵에서 파랑(Blue)
df_merge = pd.merge(base_agg, resid_agg, on=['feature', 'site'], suffixes=('_base', '_resid'))
df_merge['rank_diff'] = df_merge['importance_rank_resid'] - df_merge['importance_rank_base']

# 5. 히트맵을 위한 피벗 테이블 생성 (가로형이므로 index=site, columns=feature)
pivot_diff = df_merge.pivot(index='site', columns='feature', values='rank_diff')

# (선택) 피처 정렬: 차이가 큰 것부터 작은 순서 등 원하는 순서대로 정렬 가능합니다.
# 여기서는 전체 사이트 평균 차이값을 기준으로 정렬하여 그라데이션이 예쁘게 보이도록 했습니다.
feature_order = pivot_diff.mean(axis=0).sort_values().index
pivot_diff = pivot_diff[feature_order]

# 6. Figure 설정 (가로로 긴 2단 너비 사용)
# 가로로 기니까 높이는 4~5 정도로 슬림하게 잡아줍니다.
fig, ax = plt.subplots(figsize=(config.DOUBLE_COL, 5), layout='constrained')

# 7. 컬러맵 설정 (RdBu)
cmap = config.get_cmap("RdBu")

# 8. 히트맵 그리기
# center=0 을 명시하여 차이가 없는 0 지점이 완벽한 흰색이 되도록 고정합니다.
heatmap = sns.heatmap(pivot_diff,
                      cmap=cmap,
                      center=0,
                      ax=ax,
                      cbar_kws={'orientation': 'horizontal',
                                'shrink': 0.5, # 컬러바 가로 길이 조정
                                'aspect': 40},
                      linewidths=0.5,
                      linecolor='lightgray')

# 축 및 틱 스타일 지정
config.set_nature_ticks(ax)

# X축 피처 이름 45도 회전 (ha='right'를 주어야 글씨 끝이 틱에 정확히 맞물림)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# 축 제목
ax.set_ylabel("Site", fontweight='bold')
ax.set_xlabel("Feature", fontweight='bold')

# 컬러바 세부 설정
cb = ax.collections[0].colorbar
cb.set_label('Difference in Rank (Residual - Base)')
config.thin_spines(cb.ax)

# 9. 레이아웃 저장
save_path = os.path.join(config.SAVE_DIR, "10.png")
plt.savefig(save_path, bbox_inches='tight')
plt.show()