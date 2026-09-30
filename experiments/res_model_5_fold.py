"""
[2/7] res_model_5_fold — 잔차 모델 5-Fold OOF 교차 검증

  - Frozen encoder     : site{N}_base_rolling_m22.pt
  - 입력 ŷ_base / 정답 : base_N_oof 의 OOF 값 (oof_preds / residuals / esvd_windows) → 누수 방지
  - 스케일러           : M1~M12
  - Fold k (k=1..5)
      · 검증 구간: M(11+2k) ~ M(12+2k)  →  M13~M14, M15~M16, ..., M21~M22
      · 학습 구간: M1 ~ 검증 직전  +  (검증 끝 + lookback 672스텝) ~ M22
                   (검증 직후 lookback 만큼 비워서 입력 윈도우 겹침 차단)
      · 학습 구간 앞 90% 학습 / 뒤 10% early stopping 검증
  - 저장: site{N}_res_fold{k}.pt
          site{N}_oof_val_fold{k}.csv (Time / True_Target / Base_Pred / Res_Pred, MW)
            Base_Pred = OOF Base 예측, Res_Pred = ê × scale
  → init_alpha 에서 5개 fold csv 를 합쳐 alpha 탐색
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from common import (
    TARGET_SITES, Config, ResidualDataset,
    setup_logging, print_stage, print_site, site_paths, missing_files, load_npy, load_oof_tensors, result_path,
    steps_per_month, fit_power_scaler, load_frozen_base, train_residual_model, to_device, point_forecast, decode_time,
)

LOG_NAME = 'local_cell2_log.txt'
N_FOLDS = 5


@torch.no_grad()
def predict_fold(res_model, dataset, val_idx):
    """검증 구간 ê 추론 → MW 단위 DataFrame (ŷ_base 는 OOF 값 그대로)"""
    res_model.eval()
    loader = DataLoader(Subset(dataset, val_idx), batch_size=512, shuffle=False)
    y_true_list, base_pred_list, res_pred_list, time_str_list = [], [], [], []
    for batch in loader:
        batch = to_device(batch)
        y_hat_b = batch['oof_pred']
        e_hat = point_forecast(res_model(batch, y_hat_b))
        y_true_list.append(batch['target'].cpu().numpy())
        base_pred_list.append(y_hat_b.cpu().numpy())
        res_pred_list.append(e_hat.cpu().numpy())
        time_str_list.extend([decode_time(dataset.Y_time[i]) for i in batch['idx'].cpu().numpy().flatten()])

    y_true_np    = np.concatenate(y_true_list).flatten()
    base_pred_np = np.concatenate(base_pred_list).flatten()
    res_pred_np  = np.concatenate(res_pred_list).flatten()

    scaler = dataset.scaler
    return pd.DataFrame({
        'Time':        time_str_list,
        'True_Target': scaler.inverse_transform(y_true_np.reshape(-1, 1)).flatten(),
        'Base_Pred':   scaler.inverse_transform(base_pred_np.reshape(-1, 1)).flatten(),
        'Res_Pred':    res_pred_np * scaler.scale_[0],   # 잔차는 차이값이므로 scale 만 곱함
    })


def run_site(site_idx, cfg):
    print_site(site_idx, '잔차 모델 5-Fold 교차 검증')
    p = site_paths(site_idx)
    missing = missing_files(p, ['csv', 'X', 'base_rolling', 'oof_preds', 'residuals', 'esvd_windows'])
    if missing:
        print(f'   ⚠️  필요 파일 없음 {missing}, 스킵'); return

    df = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)
    saved_oof_preds, saved_residuals, saved_esvd_windows = load_oof_tensors(p)
    base_model = load_frozen_base(p['base_rolling'], cfg)
    print(f'   ➤ frozen encoder: {os.path.basename(p["base_rolling"])}')

    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    train_end_idx = spm * 22   # M22 끝

    scaler = fit_power_scaler(df, list(range(0, spm * 12)))
    dataset = ResidualDataset(df, npy_X, npy_Y, npy_Y_time, saved_oof_preds, saved_residuals, saved_esvd_windows, cfg, scaler)

    for fold in range(1, N_FOLDS + 1):
        val_start_idx = spm * (12 + (fold - 1) * 2)
        val_end_idx   = min(spm * (12 + fold * 2), train_end_idx)
        safe_train_restart_idx = val_end_idx + cfg.lookback

        train_idx = list(range(0, val_start_idx))
        if safe_train_restart_idx < train_end_idx:
            train_idx += list(range(safe_train_restart_idx, train_end_idx))
        val_idx = list(range(val_start_idx, val_end_idx))

        m_s = 13 + (fold - 1) * 2
        print(f'\n   ── [Fold {fold}/{N_FOLDS}] Val M{m_s}~M{m_s + 1} | 학습 {len(train_idx)} / 검증 {len(val_idx)} 샘플 ──')
        torch.cuda.empty_cache()
        res_model = train_residual_model(
            base_model, dataset, train_idx, cfg,
            save_path=result_path(site_idx, f'res_fold{fold}.pt'),
            patience=10, desc=f'Fold {fold} Residual',
        )

        fold_df = predict_fold(res_model, dataset, val_idx)
        save_csv = result_path(site_idx, f'oof_val_fold{fold}.csv')
        fold_df.to_csv(save_csv, index=False)
        print(f'      ✔ 저장: {os.path.basename(save_csv)}')

    print(f'\n   ✅ Site {site_idx} 완료')


def main():
    setup_logging(LOG_NAME)
    print_stage('res_model_5_fold — 잔차 모델 5-Fold OOF 교차 검증', [
        'frozen encoder: base_rolling_m22.pt',
        '입력 ŷ_base / 정답 잔차: base_N_oof 의 OOF 값 (누수 방지)',
        'Fold 검증 구간: M13~M14 / M15~M16 / M17~M18 / M19~M20 / M21~M22',
        '저장: res_fold{k}.pt, oof_val_fold{k}.csv → init_alpha',
    ])
    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        run_site(site_idx, cfg)


if __name__ == '__main__':
    main()
