"""
[5/7] create_fb — 통학습(Full-Batch) Base 비교군 학습 / 테스트

  - 학습      : M1~M22 전체로 새 BaseModel 학습 (앞 90% 학습 / 뒤 10% early stopping 검증)
  - 스케일러  : M1~M22 로 별도 fit (통학습 Base 전용, 다른 단계의 M1~M12 스케일러와 다름)
  - Sequential Test (M23~M24)
      · 예측값: 모델 출력 → M1~M22 스케일러 역변환 (MW)
      · 정답값: 원본 CSV Power (MW) 를 시간으로 매핑 (스케일러 무관, 매핑 실패 시 NaN)
  - 저장 : site{N}_base_fullbatch_m22.pt
           site{N}_base_fullbatch_test_results.csv (Time / True_Target / Base_Pred)
"""

import os, gc
import numpy as np
import pandas as pd
import torch

from common import (
    TARGET_SITES, Config, BaseModel, PVDataset, device,
    setup_logging, print_stage, print_site, site_paths, missing_files, load_npy, result_path,
    steps_per_month, get_capacity, fit_power_scaler, train_base_model, to_device, point_forecast, decode_time,
)

LOG_NAME = 'local_cell4b_log.txt'


@torch.no_grad()
def sequential_test_base(model, dataset, test_indices, df_origin, npy_Y_time):
    """
    예측값: 모델 출력 → scaler.inverse_transform → MW
    정답값: 원본 CSV 의 Power (MW) 를 시간 매핑으로 직접 사용
    """
    model.eval()
    pred_list, time_str_list = [], []
    for idx in test_indices:
        batch = to_device(dataset.collate([idx]))
        y_hat = point_forecast(model(batch))
        pred_list.append(y_hat.cpu().item())
        time_str_list.append(decode_time(npy_Y_time[idx]))

    pred_mw = dataset.scaler.inverse_transform(np.array(pred_list).reshape(-1, 1)).flatten()

    # 정답값: 원본 CSV 첫 번째 컬럼(시간) 기준 매핑
    df_origin['Parsed_Time'] = pd.to_datetime(df_origin.iloc[:, 0])
    true_mw = []
    for t in pd.to_datetime(time_str_list):
        row = df_origin[df_origin['Parsed_Time'] == t]
        true_mw.append(row['Power (MW)'].values[0] if len(row) > 0 else np.nan)

    return pd.DataFrame({
        'Time':        time_str_list,
        'True_Target': np.array(true_mw),
        'Base_Pred':   pred_mw,
    })


def run_site(site_idx, cfg):
    """
    통학습 Base 학습 + Sequential Test + 결과 저장.
    반환: (test_df, capacity) / 입력 파일이 없으면 None
    """
    print_site(site_idx, '통학습 Base 비교군')
    p = site_paths(site_idx)
    missing = missing_files(p, ['csv', 'X', 'Y', 'T'])
    if missing:
        print(f'   ⚠️  필요 파일 없음 {missing}, 스킵'); return None

    df_origin = pd.read_csv(p['csv'])
    npy_X, npy_Y, npy_Y_time = load_npy(p)

    n_total = len(npy_X)
    spm = steps_per_month(n_total)
    train_end_idx = min(spm * 22, n_total)
    train_idx_all = list(range(0, train_end_idx))    # M1~M22
    test_idx      = list(range(spm * 22, n_total))   # M23~M24
    print(f'   ➤ 전체 샘플: {n_total} | 학습(M1~M22): {train_end_idx} | 테스트(M23~M24): {len(test_idx)}')

    # ── [1/3] 스케일러 (M1~M22) / capacity ────────────────────────────────
    # 샘플 인덱스 구간 [0, train_end_idx) 를 df 행 인덱스에 그대로 적용 (다른 단계의 스케일러와 같은 방식)
    print('\n   [1/3] 스케일러 생성 (M1~M22)')
    scaler_m22 = fit_power_scaler(df_origin, range(0, train_end_idx))
    capacity = get_capacity(df_origin)
    print(f'      ➤ scaler mean: {scaler_m22.mean_[0]:.4f} | scale: {scaler_m22.scale_[0]:.4f}')
    print(f'      ➤ capacity (채점 분모): {capacity:.4f} MW')

    dataset = PVDataset(df_origin, npy_X, npy_Y, npy_Y_time, cfg, scaler_m22)

    # ── [2/3] 통학습 Base 학습 (M1~M22, 90/10 split) ───────────────────────
    print('\n   [2/3] 통학습 Base 모델 학습 (M1~M22)')
    split = int(len(train_idx_all) * 0.9)
    tr, val = train_idx_all[:split], train_idx_all[split:]
    base_model = BaseModel(cfg).to(device)
    torch.cuda.empty_cache()
    base_model = train_base_model(base_model, dataset, tr, val, epochs=100, patience=15, lr=1e-3,
                                  desc=f'Site{site_idx} 통학습', persistent_workers=True)

    save_pt = result_path(site_idx, 'base_fullbatch_m22.pt')
    torch.save(base_model.state_dict(), save_pt)
    print(f'      ✔ 저장: {os.path.basename(save_pt)}')

    # ── [3/3] Sequential Test ─────────────────────────────────────────────
    print(f'\n   [3/3] Sequential Test (M23~M24, {len(test_idx)}스텝)')
    torch.cuda.empty_cache()
    test_df = sequential_test_base(base_model, dataset, test_idx, df_origin.copy(), npy_Y_time)

    save_csv = result_path(site_idx, 'base_fullbatch_test_results.csv')
    test_df.to_csv(save_csv, index=False)
    print(f'      ✔ 저장: {os.path.basename(save_csv)}')

    # 메모리 정리
    del base_model, dataset
    torch.cuda.empty_cache()
    gc.collect()
    return test_df, capacity


def main():
    setup_logging(LOG_NAME)
    print_stage('create_fb — 통학습 Base 비교군 학습 / 테스트', [
        '학습: M1~M22 전체 통학습',
        '스케일러: M1~M22 (통학습 Base 전용)',
        '정답값: 원본 CSV Power (MW) 시간 매핑',
    ])
    cfg = Config(d_model=16, n_heads=1)
    for site_idx in TARGET_SITES:
        if run_site(site_idx, cfg) is not None:
            print(f'\n   ✅ Site {site_idx} 완료')


if __name__ == '__main__':
    main()
