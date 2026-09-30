"""
[IG] R-ECO Adaptive(CRC) — Integrated Gradients 피처 중요도

  - 예측 구조 : final = rolling_base(pv, weather, temporal) + α · residual_model(residual_hist, ...)
  - 모델      : site{N}_base_rolling_m22.pt  (Rolling Base, IG wrapper 의 Base)
                site{N}_base_trained.pt      (잔차 모델의 frozen encoder 기준)
                site{N}_res_final_adaptive.pt (crc_create_N_results 의 잔차 모델)
  - α         : site{N}_best_alpha_value.csv 의 Best_Alpha (상수로 고정)
  - 스케일러  : M1~M12 기준 (crc_create_N_results 와 동일)
  - 잔차 입력 : base_N_oof 가 저장한 OOF 잔차 (residuals.pt / esvd_windows.pt)
  - IG 대상   : M23~M24 테스트 구간, 입력 (pv, weather, temporal, residual_hist) / baseline = 0
  - 저장      : site{N}_ig_feature_importance_adaptive_reco.csv
"""

import os, sys, gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/
from common import (
    TARGET_SITES, SAVE_PATH, Config, PVDataset, ResidualModel, device,
    setup_logging, site_paths, missing_files, load_npy, result_path,
    steps_per_month, fit_power_scaler, load_frozen_base, point_forecast,
)

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("⚠️ captum 미설치. '!pip install captum' 실행 필요.")

LOG_NAME = 'local_adaptive_ig_log.txt'


# =========================================================================== #
# IG Wrapper
# =========================================================================== #
class AdaptiveForecastingWrapper(nn.Module):
    """
    R-ECO 최종 예측: final = base(pv, weather, temporal) + alpha * residual_model(...)
    alpha 는 상수로 고정 (IG 는 입력 텐서에 대한 gradient 만 계산)
    """
    def __init__(self, base_model, residual_model, alpha):
        super().__init__()
        self.base_model     = base_model
        self.residual_model = residual_model
        self.alpha          = alpha

    def forward(self, pv, weather, temporal, residual_hist):
        batch = {'pv': pv, 'weather': weather, 'temporal': temporal, 'residual_hist': residual_hist}
        y_hat_b = point_forecast(self.base_model(batch))
        e_hat   = point_forecast(self.residual_model(batch, y_hat_b))
        return y_hat_b + self.alpha * e_hat


# =========================================================================== #
# Dataset
# =========================================================================== #
class AdaptivePVDataset(PVDataset):
    """
    PVDataset item 에 residual_hist 추가 (저장된 OOF 산출물 사용).
      - residual_hist : [residuals[idx−R:idx], esvd_windows[idx]] → (R, 1+K). idx < R 이면 0
    common.ResidualDataset 과 같지만 IG 에는 oof_preds 가 필요 없어 로드하지 않는다.
    """
    def __init__(self, df, npy_X, npy_Y, npy_Y_time, saved_residuals, saved_esvd_windows, cfg, scaler):
        super().__init__(df, npy_X, npy_Y, npy_Y_time, cfg, scaler)
        self.saved_residuals    = saved_residuals.cpu()
        self.saved_esvd_windows = saved_esvd_windows.cpu()
        self.R, self.C = cfg.residual_lookback, cfg.n_res_channels

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        if idx >= self.R:
            batch['residual_hist'] = torch.cat([self.saved_residuals[idx - self.R:idx], self.saved_esvd_windows[idx]], dim=-1)
        else:
            batch['residual_hist'] = torch.zeros(self.R, self.C)
        return batch


# =========================================================================== #
# IG 실행
# =========================================================================== #
def run_integrated_gradients_adaptive(base_model, residual_model, dataset,
                                      test_indices, cfg, site_idx, alpha):
    if not HAS_CAPTUM:
        print('   ❌ Captum 미설치, IG 건너뜀'); return

    print(f'   🔍 Integrated Gradients 계산 중 (Site {site_idx}, alpha={alpha:.3f}, {len(test_indices)}스텝)...')

    wrapper = AdaptiveForecastingWrapper(base_model, residual_model, alpha).to(device)
    wrapper.eval()
    ig = IntegratedGradients(wrapper)

    loader = DataLoader(Subset(dataset, test_indices), batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)

    # 입력 그룹별 채널 중요도 (샘플 수 가중 합 → 마지막에 평균)
    importances = {
        'PV_Sequence':   np.zeros(cfg.n_endo_channels),
        'Weather':       np.zeros(len(cfg.weather_vars)),
        'Temporal':      np.zeros(len(cfg.temporal_vars)),
        'Residual_Hist': np.zeros(cfg.n_res_channels),
    }
    total_samples = 0

    for i, batch in enumerate(loader):
        pv            = batch['pv'].to(device)
        weather       = batch['weather'].to(device)
        temporal      = batch['temporal'].to(device)
        residual_hist = batch['residual_hist'].to(device)
        b_size = pv.size(0); total_samples += b_size

        attr_pv, attr_w, attr_t, attr_res = ig.attribute(
            inputs=(pv, weather, temporal, residual_hist),
            baselines=(torch.zeros_like(pv), torch.zeros_like(weather),
                       torch.zeros_like(temporal), torch.zeros_like(residual_hist)),
            target=0
        )

        # (B, L, C) → 시간축 |attr| 합 → 배치 평균 → (C,)
        importances['PV_Sequence']   += attr_pv.abs().sum(dim=1).mean(dim=0).cpu().numpy()  * b_size
        importances['Weather']       += attr_w.abs().sum(dim=1).mean(dim=0).cpu().numpy()   * b_size
        importances['Temporal']      += attr_t.abs().sum(dim=1).mean(dim=0).cpu().numpy()   * b_size
        importances['Residual_Hist'] += attr_res.abs().sum(dim=1).mean(dim=0).cpu().numpy() * b_size

        sys.stdout.write(f'\r      ➤ 배치 {i+1}/{len(loader)} ({total_samples}샘플)')
        sys.stdout.flush()
    print()

    for k in importances: importances[k] /= total_samples

    # ── 결과 표 (PV 채널 0 = 원값 / 1~ = ESVD, 잔차 채널 0 = 원 잔차 / 1~ = VMD IMF) ──
    results_list = []
    results_list.append({'Group': 'Endogenous', 'Feature': 'PV_Raw', 'Importance': importances['PV_Sequence'][0]})
    for i in range(1, cfg.n_endo_channels):
        results_list.append({'Group': 'Endogenous', 'Feature': f'PV_ESVD_{i}', 'Importance': importances['PV_Sequence'][i]})
    for i, v in enumerate(cfg.weather_vars):
        results_list.append({'Group': 'Weather', 'Feature': v, 'Importance': importances['Weather'][i]})
    for i, v in enumerate(cfg.temporal_vars):
        results_list.append({'Group': 'Temporal', 'Feature': v, 'Importance': importances['Temporal'][i]})
    results_list.append({'Group': 'Residual', 'Feature': 'Raw_Error', 'Importance': importances['Residual_Hist'][0]})
    for i in range(1, cfg.n_res_channels):
        results_list.append({'Group': 'Residual', 'Feature': f'IMF_{i}', 'Importance': importances['Residual_Hist'][i]})

    df_ig = pd.DataFrame(results_list)
    total_imp = df_ig['Importance'].sum()
    df_ig['Percentage (%)'] = (df_ig['Importance'] / total_imp) * 100
    df_ig = df_ig.sort_values('Importance', ascending=False).reset_index(drop=True)

    print(f'\n      [Site {site_idx} IG Feature Importance — R-ECO Adaptive]')
    print('      ' + '-'*55)
    print(f"      {'Group':<15} | {'Feature':<20} | {'Ratio (%)':>10}")
    print('      ' + '-'*55)
    for _, row in df_ig.iterrows():
        print(f"      {row['Group']:<15} | {row['Feature']:<20} | {row['Percentage (%)']:>9.2f}%")
    print('      ' + '-'*55)

    save_csv = os.path.join(SAVE_PATH, f'site{site_idx}_ig_feature_importance_adaptive_reco.csv')
    df_ig.to_csv(save_csv, index=False)
    print(f'      ➤ IG 결과 저장: {os.path.basename(save_csv)}')


# =========================================================================== #
# 사이트 단위 실행
# =========================================================================== #
def run_site(site_idx, cfg):
    print(f'\n{"="*70}\n📊 [Site {site_idx}] R-ECO Adaptive IG 시작')

    p = site_paths(site_idx)
    p['res_model'] = result_path(site_idx, 'res_final_adaptive.pt')  # crc_create_N_results 산출물
    missing = missing_files(p, ['csv', 'X', 'Y', 'T', 'base_trained', 'base_rolling', 'res_model',
                                'residuals', 'esvd_windows', 'best_alpha'])
    if missing:
        print(f'   ⚠️  파일 없음: {missing}, 스킵'); return

    df = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)
    saved_residuals    = torch.load(p['residuals'],    map_location='cpu')
    saved_esvd_windows = torch.load(p['esvd_windows'], map_location='cpu')
    best_alpha         = pd.read_csv(p['best_alpha'])['Best_Alpha'].iloc[0]

    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    scaler_train_idx = list(range(0, spm * 12))       # M1~M12 (crc_create_N_results 와 동일)
    test_idx         = list(range(spm * 22, n_total))  # M23~M24

    print(f'   ➤ 테스트 스텝: {len(test_idx)} | alpha: {best_alpha:.3f}')
    scaler = fit_power_scaler(df, scaler_train_idx)
    print(f'      ➤ 스케일러 mean: {scaler.mean_[0]:.4f}, scale: {scaler.scale_[0]:.4f}')
    dataset = AdaptivePVDataset(df, npy_X, npy_Y, npy_Y_time,
                                saved_residuals, saved_esvd_windows, cfg, scaler)

    # 잔차 모델의 frozen encoder 기준 (학습 때와 동일하게 base_trained 인코더 공유)
    base_trained_model = load_frozen_base(p['base_trained'], cfg)

    # IG wrapper 의 Base 예측용
    rolling_base_model = load_frozen_base(p['base_rolling'], cfg)
    print(f'   ➤ base_rolling_m22 로드 완료')

    res_model = ResidualModel(cfg, stage1=base_trained_model).to(device)
    res_model.load_state_dict(torch.load(p['res_model'], map_location=device))
    res_model.eval()
    for param in res_model.parameters(): param.requires_grad = False
    print(f'   ➤ res_final_adaptive 로드 완료')

    run_integrated_gradients_adaptive(rolling_base_model, res_model, dataset,
                                      test_idx, cfg, site_idx, best_alpha)

    # 메모리 정리
    del rolling_base_model, base_trained_model, res_model, dataset
    torch.cuda.empty_cache()
    gc.collect()
    print(f'   ✅ Site {site_idx} 완료\n')


def main():
    setup_logging(LOG_NAME)
    print(f"\n🚀 Using Device: {device}")
    print('\n' + '='*80)
    print('🌍 R-ECO Adaptive(CRC) IG 분석')
    print('  - Base:     site{N}_base_rolling_m22.pt')
    print('  - Residual: site{N}_res_final_adaptive.pt (frozen encoder = site{N}_base_trained.pt)')
    print('  - 스케일러: M1~M12 기준 (crc_create_N_results 와 동일)')
    print('  - OOF 잔차: 저장된 residuals.pt / esvd_windows.pt 사용')
    print('  - Alpha:    site{N}_best_alpha_value.csv 에서 로드')
    print('  - 저장:     site{N}_ig_feature_importance_adaptive_reco.csv')
    print('='*80)

    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        run_site(site_idx, cfg)


if __name__ == '__main__':
    main()
