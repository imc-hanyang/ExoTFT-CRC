import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import config  # 이제 이 파일 안의 label_panels가 'Site X'를 출력합니다.

best_model_combos = [
    ('svr', 'svr'),
    ('lgb', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('mlr', 'svr'),
    ('lgb', 'svr')
]
reforecast_data_path = config.DATA_ROOT + '/reforecast_results'


def plot_mean_variance_band_over_c(
        axes,
        df: pd.DataFrame,
        forecast_model: str,
        var_names: list,
        legend_labels: list,
        n_bins: int = 60,
        smooth_window: int = 3,
):
    c = -df[f'error_{forecast_model}'].shift(-1)
    c_clean = pd.to_numeric(c, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    bins = np.linspace(c_clean.min(), c_clean.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    for i, name in enumerate(var_names):
        if name not in df.columns: continue
        y = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if name == f'error_{forecast_model}': y = -y
        xy = pd.DataFrame({"c": c, "y": y}).dropna()
        if xy.empty: continue

        mu, sd = xy["y"].mean(), xy["y"].std(ddof=0)
        if sd > 0: xy["y"] = (xy["y"] - mu) / sd

        cats = pd.cut(xy["c"], bins=bins, include_lowest=True, right=False)
        grouped = xy.groupby(cats)["y"]
        y_mean = grouped.mean().reindex(pd.Categorical(cats.cat.categories, ordered=True))

        if smooth_window and smooth_window > 1:
            y_mean = y_mean.rolling(smooth_window, min_periods=1, center=True).mean()

        mask = ~y_mean.isna()
        axes.plot(bin_centers[mask.to_numpy()], y_mean[mask].to_numpy(), label=legend_labels[i], linewidth=1.0)

    # 개별 축 범위 설정
    axes.set_xlim(c_clean.min(), c_clean.max())

    axes.set_ylabel("Normalized")
    config.set_nature_ticks(axes)


# 1. Figure 생성: 가로 1/3 (SINGLE_COL), 세로 7칸
fig, axes = plt.subplots(7, 1, figsize=(config.SINGLE_COL, config.SINGLE_COL * 4.0), sharex=False)

legend_labels = ['total_solar', 'power_t-8', 'hour_sin', 'diff_t-1', 'residual_t']

for i, ax in enumerate(axes):
    site_num = [1, 2, 4, 5, 6, 7, 8][i]
    forecast_model = best_model_combos[i][0]
    df = pd.read_csv(os.path.join(reforecast_data_path, f"final_result_{forecast_model}_{site_num}_.csv"))
    var_names = ['Total solar irradiance (W/m2)', 'lag_8', 'hour_sin', 'diff_15min', f'error_{forecast_model}']

    plot_mean_variance_band_over_c(ax, df, forecast_model, var_names, legend_labels)

    if i == 6: ax.set_xlabel("residual_t+1")

# 2. 패널 라벨 호출 (config에서 수정된 로직에 의해 'Site 1', 'Site 2'...가 출력됨)
config.label_panels(axes, fontsize=8)

# 3. 범례 설정
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=7)

# 4. 레이아웃 조정
# plt.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.5)
plt.tight_layout(rect=[-0.02, 0.07, 1, 1], h_pad=0.5)
plt.savefig(os.path.join(config.SAVE_DIR_MAIN, "8_c.png"), dpi=300)
plt.show()