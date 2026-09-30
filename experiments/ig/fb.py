"""
[IG] 통학습(Full-Batch) Base — Integrated Gradients 피처 중요도

  - 모델      : create_fb 가 저장한 site{N}_base_fullbatch_m22.pt (새로 학습 없음)
  - 스케일러  : create_fb 와 동일하게 M1~M22 로 fit
  - IG 대상   : M23~M24 테스트 구간, 입력 (pv, weather, temporal) / baseline = 0
  - 중요도    : |attribution| 을 시간축으로 합산 → 샘플 평균 → 전체 합 대비 비율(%)
  - 저장      : site{N}_ig_feature_importance_fullbatch_base.csv
"""

import os, sys, gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/
from common import (
    TARGET_SITES, SAVE_PATH, Config, PVDataset, device,
    setup_logging, site_paths, missing_files, load_npy, result_path,
    steps_per_month, fit_power_scaler, load_frozen_base, point_forecast,
)

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("⚠️ captum 미설치. '!pip install captum' 실행 필요.")

LOG_NAME = 'local_fullbatch_ig_log.txt'


# =========================================================================== #
# IG Wrapper
# =========================================================================== #
class BaseForecastingWrapper(nn.Module):
    """captum 은 텐서 입력만 받으므로 (pv, weather, temporal) → batch dict 로 묶어 Base 호출"""
    def __init__(self, base_model):
        super().__init__(); self.base_model = base_model

    def forward(self, pv, weather, temporal):
        batch = {'pv': pv, 'weather': weather, 'temporal': temporal}
        return point_forecast(self.base_model(batch))


# =========================================================================== #
# IG 실행
# =========================================================================== #
def run_integrated_gradients_fullbatch(base_model, dataset, test_indices, cfg, site_idx):
    if not HAS_CAPTUM:
        print('   ❌ Captum 미설치, IG 건너뜀'); return

    print(f'   🔍 Integrated Gradients 계산 중 (Site {site_idx}, {len(test_indices)}스텝)...')

    wrapper = BaseForecastingWrapper(base_model).to(device)
    wrapper.eval()
    ig = IntegratedGradients(wrapper)

    loader = DataLoader(Subset(dataset, test_indices), batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)

    # 입력 그룹별 채널 중요도 (샘플 수 가중 합 → 마지막에 평균)
    importances = {
        'PV_Sequence': np.zeros(cfg.n_endo_channels),
        'Weather':     np.zeros(len(cfg.weather_vars)),
        'Temporal':    np.zeros(len(cfg.temporal_vars)),
    }
    total_samples = 0

    for i, batch in enumerate(loader):
        pv       = batch['pv'].to(device)
        weather  = batch['weather'].to(device)
        temporal = batch['temporal'].to(device)
        b_size   = pv.size(0)
        total_samples += b_size

        attr_pv, attr_w, attr_t = ig.attribute(
            inputs=(pv, weather, temporal),
            baselines=(torch.zeros_like(pv), torch.zeros_like(weather), torch.zeros_like(temporal)),
            target=0
        )

        # (B, L, C) → 시간축 |attr| 합 → 배치 평균 → (C,)
        importances['PV_Sequence'] += attr_pv.abs().sum(dim=1).mean(dim=0).cpu().numpy() * b_size
        importances['Weather']     += attr_w.abs().sum(dim=1).mean(dim=0).cpu().numpy() * b_size
        importances['Temporal']    += attr_t.abs().sum(dim=1).mean(dim=0).cpu().numpy() * b_size

        sys.stdout.write(f'\r      ➤ 배치 {i+1}/{len(loader)} ({total_samples}샘플)')
        sys.stdout.flush()
    print()

    for k in importances: importances[k] /= total_samples

    # ── 결과 표 (채널 0 = PV 원값, 1~ = ESVD 성분) ────────────────────────
    results_list = []
    results_list.append({'Group': 'Endogenous', 'Feature': 'PV_Raw', 'Importance': importances['PV_Sequence'][0]})
    for i in range(1, cfg.n_endo_channels):
        results_list.append({'Group': 'Endogenous', 'Feature': f'PV_ESVD_{i}', 'Importance': importances['PV_Sequence'][i]})
    for i, v in enumerate(cfg.weather_vars):
        results_list.append({'Group': 'Weather', 'Feature': v, 'Importance': importances['Weather'][i]})
    for i, v in enumerate(cfg.temporal_vars):
        results_list.append({'Group': 'Temporal', 'Feature': v, 'Importance': importances['Temporal'][i]})

    df_ig = pd.DataFrame(results_list)
    total_imp = df_ig['Importance'].sum()
    df_ig['Percentage (%)'] = (df_ig['Importance'] / total_imp) * 100
    df_ig = df_ig.sort_values('Importance', ascending=False).reset_index(drop=True)

    print(f'\n      [Site {site_idx} IG Feature Importance — 통학습 Base]')
    print('      ' + '-'*55)
    print(f"      {'Group':<15} | {'Feature':<20} | {'Ratio (%)':>10}")
    print('      ' + '-'*55)
    for _, row in df_ig.iterrows():
        print(f"      {row['Group']:<15} | {row['Feature']:<20} | {row['Percentage (%)']:>9.2f}%")
    print('      ' + '-'*55)

    save_csv = os.path.join(SAVE_PATH, f'site{site_idx}_ig_feature_importance_fullbatch_base.csv')
    df_ig.to_csv(save_csv, index=False)
    print(f'      ➤ IG 결과 저장: {os.path.basename(save_csv)}')


# =========================================================================== #
# 사이트 단위 실행
# =========================================================================== #
def run_site(site_idx, cfg):
    print(f'\n{"="*70}\n📊 [Site {site_idx}] 통학습 Base IG 시작')

    p = site_paths(site_idx)
    p['model'] = result_path(site_idx, 'base_fullbatch_m22.pt')  # create_fb 산출물
    missing = missing_files(p, ['csv', 'X', 'Y', 'T', 'model'])
    if missing:
        print(f'   ⚠️  파일 없음: {missing}, 스킵'); return

    df = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)

    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    scaler_train_idx = list(range(0, min(spm * 22, n_total)))  # M1~M22 (create_fb 와 동일)
    test_idx         = list(range(spm * 22, n_total))          # M23~M24

    print(f'   ➤ 테스트 스텝: {len(test_idx)} | 스케일러 구간: M1~M22')
    scaler = fit_power_scaler(df, scaler_train_idx)
    print(f'      ➤ 스케일러 mean: {scaler.mean_[0]:.4f}, scale: {scaler.scale_[0]:.4f}')
    dataset = PVDataset(df, npy_X, npy_Y, npy_Y_time, cfg, scaler)

    base_model = load_frozen_base(p['model'], cfg)
    print(f'   ➤ 가중치 로드 완료: {os.path.basename(p["model"])}')

    run_integrated_gradients_fullbatch(base_model, dataset, test_idx, cfg, site_idx)

    # 메모리 정리
    del base_model, dataset
    torch.cuda.empty_cache()
    gc.collect()
    print(f'   ✅ Site {site_idx} 완료\n')


def main():
    setup_logging(LOG_NAME)
    print(f"\n🚀 Using Device: {device}")
    print('\n' + '='*80)
    print('🌍 통학습 Base IG 분석 (site{N}_base_fullbatch_m22.pt 로드)')
    print('  - 새로 학습 없음, 저장된 가중치 그대로 사용')
    print('  - 스케일러: create_fb 와 동일한 M1~M22 기준')
    print('  - 저장: site{N}_ig_feature_importance_fullbatch_base.csv')
    print('='*80)

    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        run_site(site_idx, cfg)


if __name__ == '__main__':
    main()
