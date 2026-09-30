"""
[3/7] init_alpha — OOF 기반 초기 앙상블 비율(alpha) 탐색

  - 입력   : site{N}_oof_val_fold{1..5}.csv (res_model_5_fold) — 5개 모두 있어야 진행
  - 채점   : Site 7 이상 구간 제외 + 가동 시간 필터 (threshold 없음), 분모 = capacity
  - 탐색   : alpha ∈ [0.0, 2.0], 0.01 간격 (201개),  ŷ = Base_Pred + alpha · Res_Pred
  - 기준   : NRMSE 최소
  - 저장   : site{N}_best_alpha_value.csv (Site / Best_Alpha / OOF_NRMSE / OOF_NMAE / OOF_R2)
  → create_rf (고정 alpha), crc_create_N_results (adaptive alpha 초기값) 에서 사용
"""

import os
import numpy as np
import pandas as pd

from common import (
    TARGET_SITES, setup_logging, print_stage, print_site, site_paths, result_path,
    get_capacity, filter_eval_window, score,
)

LOG_NAME = 'local_cell3_log.txt'
N_FOLDS = 5


def get_metrics(y_true, y_pred, capacity):
    """(NRMSE %, NMAE %, R²). 샘플이 없으면 0"""
    if len(y_true) == 0: return 0.0, 0.0, 0.0
    return score(y_true, y_pred, capacity)


def run_site(site_idx, alphas):
    print_site(site_idx, 'Alpha Sweep')
    p = site_paths(site_idx)
    if not os.path.exists(p['csv']):
        print(f'   ⚠️  원본 CSV 없음 ({os.path.basename(p["csv"])}), 스킵'); return
    capacity = get_capacity(pd.read_csv(p['csv']))

    fold_files = [result_path(site_idx, f'oof_val_fold{fold}.csv') for fold in range(1, N_FOLDS + 1)]
    oof_dfs = [pd.read_csv(f) for f in fold_files if os.path.exists(f)]
    if len(oof_dfs) != N_FOLDS:
        missing = [os.path.basename(f) for f in fold_files if not os.path.exists(f)]
        print(f'   ⚠️  fold 결과 부족 ({len(oof_dfs)}/{N_FOLDS}), 없음: {missing}, 스킵'); return

    valid_df = filter_eval_window(pd.concat(oof_dfs, ignore_index=True), site_idx)
    y_true, y_base, y_res = valid_df['True_Target'].values, valid_df['Base_Pred'].values, valid_df['Res_Pred'].values
    print(f'   ➤ Capacity: {capacity:.4f} MW | 채점 샘플: {len(y_true)}')

    best_alpha, best_nrmse, best_metrics = 0.0, float('inf'), {}
    for alpha in alphas:
        y_final = y_base + (alpha * y_res)
        nrmse, nmae, r2 = get_metrics(y_true, y_final, capacity)
        if nrmse < best_nrmse: best_nrmse, best_alpha, best_metrics = nrmse, alpha, {'NRMSE': nrmse, 'NMAE': nmae, 'R2': r2}

    base_nrmse, base_nmae, base_r2 = get_metrics(y_true, y_base, capacity)
    print(f'\n   🏆 [Site {site_idx}] Best Alpha = {best_alpha:.2f}')
    print(f'      {"":<18} {"NRMSE":>8} {"NMAE":>8} {"R²":>8}')
    print(f'      {"alpha = 0 (Base)":<18} {base_nrmse:>7.3f}% {base_nmae:>7.3f}% {base_r2:>8.4f}')
    print(f'      {f"alpha = {best_alpha:.2f}":<18} {best_metrics["NRMSE"]:>7.3f}% {best_metrics["NMAE"]:>7.3f}% {best_metrics["R2"]:>8.4f}')

    pd.DataFrame([{
        'Site': site_idx, 'Best_Alpha': best_alpha,
        'OOF_NRMSE': best_metrics['NRMSE'], 'OOF_NMAE': best_metrics['NMAE'], 'OOF_R2': best_metrics['R2'],
    }]).to_csv(p['best_alpha'], index=False)
    print(f'      ✔ 저장: {os.path.basename(p["best_alpha"])}')


def main():
    setup_logging(LOG_NAME)
    print_stage('init_alpha — OOF 기반 앙상블 비율(alpha) 탐색', [
        '입력: oof_val_fold1~5.csv',
        '탐색: alpha 0.00 ~ 2.00 (0.01 간격), NRMSE 최소',
        '채점: 가동 시간 기준 (threshold 없음)',
    ])
    alphas = np.linspace(0.0, 2.0, 201)
    for site_idx in TARGET_SITES:
        run_site(site_idx, alphas)


if __name__ == '__main__':
    main()
