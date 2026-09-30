"""
[1/7] base_N_oof — Base 모델 학습 + 롤링 OOF 잔차 추출

  1. 초기 Base 학습   : M1~M10 Train / M11~M12 Val (early stopping)       → site{N}_base_trained.pt
  2. 롤링 OOF 잔차    : M11~M22 를 2개월 단위로 반복
                         (a) 현재 모델로 해당 2개월 예측 (OOS) → 잔차 기록
                         (b) M1~직전 구간 Train / 방금 예측한 2개월 Val 로 파인튜닝 (lr=1e-4)
                         마지막 반복까지 끝난 모델                          → site{N}_base_rolling_m22.pt
                       M1~M10 은 초기 Base 의 in-sample 예측으로 채움 (잔차 윈도우 초기화용)
  3. ESVD(VMD) 피처   : 잔차 윈도우(직전 96스텝)마다 VMD 분해 (K=4)
  4. 저장             : site{N}_oof_preds.pt / site{N}_residuals.pt / site{N}_esvd_windows.pt

  * 스케일러: M1~M12 구간 Power 로 fit, 이후 모든 단계(통학습 Base 제외)에서 동일
  * 산출물 사용처
      base_trained.pt       → create_rf, crc_create_N_results 잔차 모델의 frozen encoder
      base_rolling_m22.pt   → res_model_5_fold 의 frozen encoder,
                              create_rf, crc_create_N_results 의 Base 예측 (비교군/실험군 공통)
      oof_preds / residuals / esvd_windows → 잔차 모델 학습 입력 (ResidualDataset)
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from joblib import Parallel, delayed

from common import (
    HAS_VMDPY, TARGET_SITES, Config, BaseModel, PVDataset, device,
    setup_logging, print_stage, print_site, site_paths, missing_files, load_npy,
    steps_per_month, fit_power_scaler, train_base_model, extract_vmd_features, to_device, point_forecast,
)

LOG_NAME = 'local_cell1_log.txt'


# =========================================================================== #
# ESVD(VMD) 피처 — 잔차 윈도우별 VMD 분해 (병렬)
# =========================================================================== #
def process_single_vmd(i, oof_np, R, K):
    """샘플 i 의 잔차 윈도우 oof[i−R:i] → (R, K) VMD 모드"""
    window_signal = oof_np[i - R:i]
    res = extract_vmd_features(window_signal, K) if np.abs(window_signal).max() > 1e-5 else np.zeros((R, K))
    return i, res


def prepare_oof_esvd(oof_res: torch.Tensor, R: int, K: int) -> torch.Tensor:
    """OOF 잔차 (N, 1) → esvd_windows (N, R, K). i < R 인 샘플은 0."""
    N = len(oof_res)
    esvd_windows = torch.zeros(N, R, K)
    oof_np = oof_res.numpy().flatten()
    total_tasks, chunk_size = N - R, 1000
    with Parallel(n_jobs=-1) as parallel:
        for chunk_start in range(R, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            results = parallel(delayed(process_single_vmd)(i, oof_np, R, K) for i in range(chunk_start, chunk_end))
            for i, res_np in results: esvd_windows[i] = torch.tensor(res_np, dtype=torch.float32)
            processed = chunk_end - R
            sys.stdout.write(f'\r      - ESVD 추출 진행률: {(processed / total_tasks) * 100:5.1f}%')
            sys.stdout.flush()
    print()
    return esvd_windows


# =========================================================================== #
# 추론
# =========================================================================== #
@torch.no_grad()
def predict_chunk(model, dataset, start_idx, end_idx):
    """[start_idx, end_idx) 구간 예측 → (targets, preds), scaled"""
    model.eval()
    loader = DataLoader(Subset(dataset, range(start_idx, end_idx)), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    preds, targets = [], []
    for batch in loader:
        batch = to_device(batch)
        y_hat = point_forecast(model(batch))
        preds.append(y_hat.cpu().squeeze()); targets.append(batch['target'].cpu().squeeze())
    return torch.cat(targets), torch.cat(preds)


# =========================================================================== #
# 사이트 단위 실행
# =========================================================================== #
def run_site(site_idx, cfg):
    print_site(site_idx, 'Base 학습 + OOF 잔차 추출')
    p = site_paths(site_idx)
    missing = missing_files(p, ['csv', 'X'])
    if missing:
        print(f'   ⚠️  필요 파일 없음 {missing}, 스킵'); return

    df = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)
    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    print(f'   ➤ 전체 샘플: {n_total} | 1개월 = {spm} 스텝')

    # 스케일러 기준: M1~M12
    scaler = fit_power_scaler(df, list(range(0, spm * 12)))
    dataset = PVDataset(df, npy_X, npy_Y, npy_Y_time, cfg, scaler)
    oof_preds = torch.zeros(n_total, 1)
    oof_res   = torch.zeros(n_total, 1)

    # ── [1/4] 초기 Base 학습: M1~M10 Train / M11~M12 Val ─────────────────
    print('\n   [1/4] 초기 Base 모델 학습 (M1~M10 Train / M11~M12 Val)')
    base_model = BaseModel(cfg).to(device)
    base_model = train_base_model(
        base_model, dataset,
        train_idx=list(range(0, spm * 10)),
        val_idx  =list(range(spm * 10, spm * 12)),
        desc='초기 Base',
    )
    torch.save(base_model.state_dict(), p['base_trained'])
    print(f'      ✔ 저장: {os.path.basename(p["base_trained"])}')

    # ── [2/4] 롤링 OOF 잔차 추출 (M11~M22, 2개월 단위) ─────────────────────
    print('\n   [2/4] 롤링 OOF 잔차 추출 (M11~M22, 2개월 단위)')

    # M1~M10: 초기 Base 의 in-sample 예측 (잔차 윈도우 초기화 목적)
    t_m1_10, p_m1_10 = predict_chunk(base_model, dataset, 0, spm * 10)
    oof_preds[0:spm * 10, 0] = p_m1_10
    oof_res  [0:spm * 10, 0] = t_m1_10 - p_m1_10
    print('      ➤ M1~M10 초기 Base 예측으로 채움')

    # M11~M12, M13~M14, ..., M21~M22
    for m in range(10, 22, 2):
        start_idx = m * spm
        end_idx   = min((m + 2) * spm, n_total)
        label     = f'M{m + 1}~M{m + 2}'

        # (a) 해당 구간을 보기 전 모델로 OOS 예측
        targets_m, preds_m = predict_chunk(base_model, dataset, start_idx, end_idx)
        oof_preds[start_idx:end_idx, 0] = preds_m
        oof_res  [start_idx:end_idx, 0] = targets_m - preds_m
        print(f'      ➤ {label} OOS 예측 완료')

        # (b) M1~직전 구간 Train / 방금 예측한 구간 Val 로 파인튜닝
        base_model = train_base_model(
            base_model, dataset,
            train_idx=list(range(0, start_idx)),
            val_idx  =list(range(start_idx, end_idx)),
            epochs=30, patience=5, lr=1e-4,
            desc=f'{label} 롤링',
        )

    torch.save(base_model.state_dict(), p['base_rolling'])
    print(f'      ✔ 저장: {os.path.basename(p["base_rolling"])}')

    # ── [3/4] ESVD(VMD) 피처 생성 ──────────────────────────────────────────
    print(f'\n   [3/4] ESVD(VMD) 피처 생성 (R={cfg.residual_lookback}, K={cfg.k_imfs})')
    esvd_windows = prepare_oof_esvd(oof_res, cfg.residual_lookback, cfg.k_imfs)

    # ── [4/4] 산출물 저장 ──────────────────────────────────────────────────
    print('\n   [4/4] 산출물 저장')
    torch.save(oof_preds,    p['oof_preds'])
    torch.save(oof_res,      p['residuals'])
    torch.save(esvd_windows, p['esvd_windows'])
    for key in ['oof_preds', 'residuals', 'esvd_windows']:
        print(f'      ✔ 저장: {os.path.basename(p[key])}')
    print(f'\n   ✅ Site {site_idx} 완료')


def main():
    setup_logging(LOG_NAME)
    print_stage('base_N_oof — Base 모델 학습 + 롤링 OOF 잔차 추출', [
        '스케일러: M1~M12 고정',
        '초기 Base: M1~M10 Train / M11~M12 Val → base_trained.pt',
        '롤링 OOF 잔차: M11~M22, 2개월 단위 (OOS 예측 → 파인튜닝) → base_rolling_m22.pt',
        'ESVD(VMD): 잔차 96스텝 윈도우, K=4 → esvd_windows.pt',
    ])
    if not HAS_VMDPY:
        print('   ⚠️  vmdpy 미설치 — ESVD 피처가 모두 0 으로 채워집니다')

    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        run_site(site_idx, cfg)


if __name__ == '__main__':
    main()
