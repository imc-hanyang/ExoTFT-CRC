"""
[4/7] create_rf — Rolling Base + 잔차 모델 (고정 alpha) 최종 테스트

  - 비교군 : base_rolling_m22 단독
  - 실험군 : base_rolling_m22 + best_alpha × ê   (best_alpha: init_alpha 결과, 고정)
             → 두 군의 차이는 잔차 모델 유/무뿐
  - 잔차 모델 : M13~M22 전체로 재학습 (앞 90% 학습 / 뒤 10% 검증)
                frozen encoder = base_trained.pt              → site{N}_res_final.pt
  - Sequential Test (M23~M24, 1스텝씩)
      · 워밍업: test 직전 96스텝을 rolling_base 로 재추론해 잔차 윈도우 초기화
      · 매 스텝: 잔차 윈도우 VMD → ê → ŷ = ŷ_base + alpha·ê
                 → 실제 오차 (y − ŷ_base) 를 윈도우에 push
  - 저장 : site{N}_final_test_results.csv
           (Time / True_Target / Rolling_Base_Pred / Final_Ensemble_Pred, MW — 스케일러 역변환)
  * 채점은 하지 않음 (crc_create_N_results 에서 adaptive alpha 버전을 채점)
"""

import os
import numpy as np
import pandas as pd
import torch

from common import (
    TARGET_SITES, Config, ResidualDataset,
    setup_logging, print_stage, print_site, site_paths, missing_files, load_npy, load_oof_tensors, result_path,
    steps_per_month, fit_power_scaler, load_frozen_base, train_residual_model,
    warmup_residual_window, predict_residual, push_residual, to_device, point_forecast, decode_time,
)

LOG_NAME = 'local_cell4_log.txt'
REQUIRED = ['csv', 'X', 'base_trained', 'base_rolling', 'oof_preds', 'residuals', 'esvd_windows', 'best_alpha']


@torch.no_grad()
def sequential_test(rolling_base_model, residual_model, dataset, test_indices, cfg, best_alpha):
    """비교군(rolling_base 단독)과 실험군(rolling_base + alpha·ê)을 한 번에 1스텝씩 추론"""
    rolling_base_model.eval(); residual_model.eval()
    R = cfg.residual_lookback

    print(f'      ➤ 잔차 윈도우 워밍업 (rolling_base 로 test 직전 {R}스텝 재추론)...')
    current_res_raw = warmup_residual_window(rolling_base_model, dataset, test_indices[0], R)
    print('      ➤ 워밍업 완료')

    y_true_list, base_pred_list, ensemble_pred_list, time_str_list = [], [], [], []
    for idx in test_indices:
        batch = to_device(dataset.collate([idx]))

        # [비교군] rolling_base 단독
        y_hat_b = point_forecast(rolling_base_model(batch))
        # [실험군] 실시간 ESVD → 잔차 모델 → 앙상블
        e_hat = predict_residual(residual_model, batch, current_res_raw, y_hat_b, cfg)
        final_ensemble = y_hat_b + (best_alpha * e_hat)

        target = batch['target']
        y_true_list.append(target.cpu().item())
        base_pred_list.append(y_hat_b.cpu().item())
        ensemble_pred_list.append(final_ensemble.cpu().item())
        time_str_list.append(decode_time(dataset.Y_time[idx]))

        # 잔차 윈도우 업데이트 (rolling_base 기준 실제 오차)
        current_res_raw = push_residual(current_res_raw, target - y_hat_b)

    scaler = dataset.scaler
    return pd.DataFrame({
        'Time':                time_str_list,
        'True_Target':         scaler.inverse_transform(np.array(y_true_list).reshape(-1, 1)).flatten(),
        'Rolling_Base_Pred':   scaler.inverse_transform(np.array(base_pred_list).reshape(-1, 1)).flatten(),      # 비교군
        'Final_Ensemble_Pred': scaler.inverse_transform(np.array(ensemble_pred_list).reshape(-1, 1)).flatten(),  # 실험군
    })


def run_site(site_idx, cfg):
    print_site(site_idx, '최종 테스트 (고정 alpha)')
    p = site_paths(site_idx)
    missing = missing_files(p, REQUIRED)
    if missing:
        print(f'   ⚠️  필요 파일 없음 {missing}, 스킵'); return

    df = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)
    saved_oof_preds, saved_residuals, saved_esvd_windows = load_oof_tensors(p)
    best_alpha = pd.read_csv(p['best_alpha'])['Best_Alpha'].iloc[0]

    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    scaler_train_idx = list(range(0, spm * 12))                                # M1~M12
    res_train_idx    = list(range(spm * 12, min(spm * 22, n_total)))           # M13~M22
    test_idx         = list(range(spm * 22, n_total))                          # M23~M24

    scaler = fit_power_scaler(df, scaler_train_idx)
    dataset = ResidualDataset(df, npy_X, npy_Y, npy_Y_time, saved_oof_preds, saved_residuals, saved_esvd_windows, cfg, scaler)

    # ── [1/3] base_rolling_m22 로드 — 비교군·실험군 공통 Base ────────────
    rolling_base_model = load_frozen_base(p['base_rolling'], cfg)
    print(f'   [1/3] base_rolling_m22 로드 완료 | best_alpha = {best_alpha:.2f}')

    # ── [2/3] 잔차 모델 M13~M22 전체 재학습 (frozen encoder = base_trained) ─
    print(f'\n   [2/3] 잔차 모델 재학습 (M13~M22, {len(res_train_idx)} 샘플, frozen encoder = base_trained)')
    base_trained_model = load_frozen_base(p['base_trained'], cfg)
    torch.cuda.empty_cache()
    res_model = train_residual_model(
        base_trained_model, dataset, res_train_idx, cfg,
        save_path=result_path(site_idx, 'res_final.pt'),
        desc='Final Residual', num_workers=4, pin_memory=True,
    )

    # ── [3/3] Sequential Test ─────────────────────────────────────────────
    print(f'\n   [3/3] Sequential Test (M23~M24, {len(test_idx)}스텝)')
    test_df = sequential_test(rolling_base_model, res_model, dataset, test_idx, cfg, best_alpha)

    save_csv = result_path(site_idx, 'final_test_results.csv')
    test_df.to_csv(save_csv, index=False)
    print(f'      ✔ 저장: {os.path.basename(save_csv)}')
    print(f'\n   ✅ Site {site_idx} 완료')


def main():
    setup_logging(LOG_NAME)
    print_stage('create_rf — Rolling Base + 잔차 모델 (고정 alpha) 최종 테스트', [
        '비교군: base_rolling_m22 단독',
        '실험군: base_rolling_m22 + best_alpha × 잔차 모델',
        '두 군의 차이 = 잔차 모델 유/무',
        '잔차 모델: M13~M22 재학습, frozen encoder = base_trained.pt',
    ])
    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        run_site(site_idx, cfg)


if __name__ == '__main__':
    main()
