"""
[7/7] crc_create_N_results — Rolling Base + 잔차 모델 (Adaptive Alpha) 최종 테스트 + 채점

  - 비교군 : base_rolling_m22 단독
  - 실험군 : base_rolling_m22 + adaptive_alpha × ê
  - 잔차 모델 : M13~M22 재학습 (create_rf 와 같은 방식이지만 별도로 새로 학습)
                frozen encoder = base_trained.pt                 → site{N}_res_final_adaptive.pt
  - Adaptive Alpha (초기값 = init_alpha 의 best_alpha)
      96스텝마다 직전 96스텝의 |Base 오차| vs |앙상블 오차| paired t-test
        p < 0.05  &  앙상블 오차 > Base 오차 → alpha × 0.5   (유의한 역효과)
        p < 0.05  &  앙상블 오차 < Base 오차 → alpha × 1.2   (유의한 개선)
        그 외                                 → 유지
      범위 [0.0, 2.0]. Alpha_Used 는 해당 스텝 예측에 실제 사용된 값 (갱신 전)
      참조: Student (1908). Biometrika 6(1); Fisher (1925) 유의수준 0.05 관례
  - 채점 : Site 7 이상 구간 제외 + 가동 시간 필터 (threshold 없음), 분모 = capacity
           비교군 / 실험군 NRMSE·NMAE·R² + NRMSE 개선폭(비교군 − 실험군, %p)
  - 저장 : site{N}_res_final_adaptive.csv
           (Time / True_Target / Rolling_Base_Pred / Final_Ensemble_Pred / Alpha_Used, MW — 스케일러 역변환)
"""

import os, gc
import numpy as np
import pandas as pd
import torch
from scipy import stats

from common import (
    TARGET_SITES, Config, ResidualDataset,
    setup_logging, print_stage, print_site, print_score_table, site_paths, missing_files, load_npy, load_oof_tensors,
    result_path, steps_per_month, get_capacity, fit_power_scaler, load_frozen_base, train_residual_model,
    warmup_residual_window, predict_residual, push_residual, to_device, point_forecast, decode_time,
    filter_eval_window, score,
)

LOG_NAME = 'local_cell4_adaptive_v3_log.txt'
REQUIRED = ['csv', 'X', 'base_trained', 'base_rolling', 'oof_preds', 'residuals', 'esvd_windows', 'best_alpha']

# Adaptive Alpha 하이퍼파라미터
ADAPT_WINDOW = 96      # 재평가 주기 / t-test 윈도우 (스텝)
DECAY        = 0.5     # 유의한 역효과 시 곱하는 값
RECOVER      = 1.2     # 유의한 개선 시 곱하는 값
MIN_ALPHA    = 0.0
MAX_ALPHA    = 2.0
P_THRESHOLD  = 0.05


@torch.no_grad()
def sequential_test_adaptive_v3(rolling_base_model, residual_model, dataset,
                                test_indices, cfg, best_alpha,
                                adapt_window=96, decay=0.5, recover=1.2,
                                min_alpha=0.0, max_alpha=2.0, p_threshold=0.05):
    """Sequential Test + Paired t-test 기반 Adaptive Alpha (규칙은 모듈 docstring 참조)"""
    rolling_base_model.eval(); residual_model.eval()
    R = cfg.residual_lookback

    print(f'      ➤ 잔차 윈도우 워밍업 (rolling_base 로 test 직전 {R}스텝 재추론)...')
    current_res_raw = warmup_residual_window(rolling_base_model, dataset, test_indices[0], R)
    print('      ➤ 워밍업 완료')

    adaptive_alpha = best_alpha
    y_true_list, base_pred_list, ensemble_pred_list = [], [], []
    alpha_log, pvalue_log = [], []
    buf_base_err, buf_ens_err = [], []

    print(f'      ➤ Adaptive Alpha 시작 | 초기 alpha {adaptive_alpha:.3f} | window {adapt_window} | '
          f'decay ×{decay} | recover ×{recover} | 범위 [{min_alpha}, {max_alpha}] | p < {p_threshold}')
    print(f'      {"Step":>11}  {"Alpha 변화":<18}  판정')

    for step, idx in enumerate(test_indices):
        batch = to_device(dataset.collate([idx]))

        # [비교군] rolling_base 단독
        y_hat_b = point_forecast(rolling_base_model(batch))
        # [실험군] 실시간 ESVD → 잔차 모델 → 앙상블
        e_hat = predict_residual(residual_model, batch, current_res_raw, y_hat_b, cfg)
        final_ensemble = y_hat_b + (adaptive_alpha * e_hat)
        target = batch['target']

        buf_base_err.append(abs((target - y_hat_b).item()))
        buf_ens_err.append(abs((target - final_ensemble).item()))
        y_true_list.append(target.cpu().item())
        base_pred_list.append(y_hat_b.cpu().item())
        ensemble_pred_list.append(final_ensemble.cpu().item())
        alpha_log.append(adaptive_alpha)

        # ── adapt_window 스텝마다 Paired t-test 로 alpha 재평가 ──────────
        if (step + 1) % adapt_window == 0:
            window_base_arr = np.array(buf_base_err[-adapt_window:])
            window_ens_arr  = np.array(buf_ens_err[-adapt_window:])
            mean_base, mean_ens = window_base_arr.mean(), window_ens_arr.mean()
            _, p_value = stats.ttest_rel(window_base_arr, window_ens_arr)
            pvalue_log.append(p_value)
            prev_alpha = adaptive_alpha

            if p_value < p_threshold and mean_ens > mean_base:
                adaptive_alpha = max(min_alpha, adaptive_alpha * decay)
                direction = f'↓ 유의한 역효과 (p={p_value:.4f}, base={mean_base:.4f} < ens={mean_ens:.4f})'
            elif p_value < p_threshold and mean_ens < mean_base:
                adaptive_alpha = min(max_alpha, adaptive_alpha * recover)
                direction = f'↑ 유의한 개선   (p={p_value:.4f}, base={mean_base:.4f} > ens={mean_ens:.4f})'
            else:
                direction = f'→ 유의하지 않음 (p={p_value:.4f}), 유지'

            print(f'      [{step + 1:5d}/{len(test_indices)}]  {prev_alpha:.4f} → {adaptive_alpha:.4f}  {direction}')

        # 잔차 윈도우 업데이트 (rolling_base 기준 실제 오차)
        current_res_raw = push_residual(current_res_raw, target - y_hat_b)

    scaler = dataset.scaler
    y_true_mw    = scaler.inverse_transform(np.array(y_true_list).reshape(-1, 1)).flatten()
    base_pred_mw = scaler.inverse_transform(np.array(base_pred_list).reshape(-1, 1)).flatten()
    ens_pred_mw  = scaler.inverse_transform(np.array(ensemble_pred_list).reshape(-1, 1)).flatten()
    time_str_list = [decode_time(dataset.Y_time[i]) for i in test_indices]

    print(f'      ➤ Alpha 요약  : 초기 {best_alpha:.3f} | 최종 {adaptive_alpha:.3f} | '
          f'min {min(alpha_log):.3f} | max {max(alpha_log):.3f} | mean {np.mean(alpha_log):.3f}')
    if pvalue_log:
        print(f'      ➤ p-value 요약: mean {np.mean(pvalue_log):.4f} | '
              f'p < {p_threshold} 비율 {np.mean(np.array(pvalue_log) < p_threshold) * 100:.1f}%')

    return pd.DataFrame({
        'Time':                time_str_list,
        'True_Target':         y_true_mw,
        'Rolling_Base_Pred':   base_pred_mw,   # 비교군
        'Final_Ensemble_Pred': ens_pred_mw,    # 실험군
        'Alpha_Used':          alpha_log,
    })


def run_site(site_idx, cfg):
    print_site(site_idx, '최종 테스트 (Adaptive Alpha)')
    p = site_paths(site_idx)
    missing = missing_files(p, REQUIRED)
    if missing:
        print(f'   ⚠️  필요 파일 없음 {missing}, 스킵'); return

    df = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)
    saved_oof_preds, saved_residuals, saved_esvd_windows = load_oof_tensors(p)
    best_alpha = pd.read_csv(p['best_alpha'])['Best_Alpha'].iloc[0]
    capacity   = get_capacity(df)

    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    scaler_train_idx = list(range(0, spm * 12))                        # M1~M12
    res_train_idx    = list(range(spm * 12, min(spm * 22, n_total)))   # M13~M22
    test_idx         = list(range(spm * 22, n_total))                  # M23~M24

    scaler = fit_power_scaler(df, scaler_train_idx)
    dataset = ResidualDataset(df, npy_X, npy_Y, npy_Y_time, saved_oof_preds, saved_residuals, saved_esvd_windows, cfg, scaler)

    # ── [1/4] base_rolling_m22 로드 — 비교군·실험군 공통 Base ────────────
    rolling_base_model = load_frozen_base(p['base_rolling'], cfg)
    print(f'   [1/4] base_rolling_m22 로드 완료 | 초기 alpha (best_alpha) = {best_alpha:.2f}')

    # ── [2/4] 잔차 모델 M13~M22 재학습 (frozen encoder = base_trained) ────
    print(f'\n   [2/4] 잔차 모델 재학습 (M13~M22, {len(res_train_idx)} 샘플, frozen encoder = base_trained)')
    base_trained_model = load_frozen_base(p['base_trained'], cfg)
    torch.cuda.empty_cache()
    res_model = train_residual_model(
        base_trained_model, dataset, res_train_idx, cfg,
        save_path=result_path(site_idx, 'res_final_adaptive.pt'),
        desc='Adaptive Residual', num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── [3/4] Sequential Test ─────────────────────────────────────────────
    print(f'\n   [3/4] Sequential Test (M23~M24, {len(test_idx)}스텝)')
    test_df = sequential_test_adaptive_v3(
        rolling_base_model, res_model, dataset, test_idx, cfg, best_alpha,
        adapt_window=ADAPT_WINDOW, decay=DECAY, recover=RECOVER,
        min_alpha=MIN_ALPHA, max_alpha=MAX_ALPHA, p_threshold=P_THRESHOLD,
    )
    save_csv = result_path(site_idx, 'res_final_adaptive.csv')
    test_df.to_csv(save_csv, index=False)
    print(f'      ✔ 저장: {os.path.basename(save_csv)}')

    # ── [4/4] 채점: 가동 시간 기준, threshold 없음 ─────────────────────────
    valid_df = filter_eval_window(test_df, site_idx)
    yt      = valid_df['True_Target'].values
    y_base  = valid_df['Rolling_Base_Pred'].values
    y_final = valid_df['Final_Ensemble_Pred'].values

    b_nrmse, b_nmae, b_r2 = score(yt, y_base,  capacity)
    f_nrmse, f_nmae, f_r2 = score(yt, y_final, capacity)
    delta = b_nrmse - f_nrmse   # 양수 = 잔차 보정으로 개선

    print_score_table(
        f'[Site {site_idx}] 최종 결과 (M23~M24) | capacity {capacity:.4f} MW | 채점 샘플 {len(yt)}',
        [('[비교군] Rolling Base 단독',               b_nrmse, b_nmae, b_r2),
         ('[실험군] Rolling Base + 잔차 (Adaptive α)', f_nrmse, f_nmae, f_r2)],
    )
    print(f'   ▶ NRMSE 개선: {delta:+.3f}%p {"✅" if delta > 0 else "❌"}')

    del rolling_base_model, base_trained_model, res_model, dataset
    torch.cuda.empty_cache()
    gc.collect()
    print(f'\n   ✅ Site {site_idx} 완료')


def main():
    setup_logging(LOG_NAME)
    print_stage('crc_create_N_results — Rolling Base + 잔차 모델 (Adaptive Alpha) 최종 테스트', [
        f'Alpha 조정: {ADAPT_WINDOW}스텝마다 Paired t-test (p < {P_THRESHOLD}), ×{DECAY} / ×{RECOVER}',
        f'Alpha 범위: {MIN_ALPHA} ~ {MAX_ALPHA}',
        '채점: 가동 시간 기준 (threshold 없음)',
        '모델: site{N}_res_final_adaptive.pt',
        '결과: site{N}_res_final_adaptive.csv',
    ])
    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        run_site(site_idx, cfg)


if __name__ == '__main__':
    main()
