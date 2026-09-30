import os
import sys
import time
import builtins
import warnings
import numpy as np
import pandas as pd
from PyEMD import EMD
from vmdpy import VMD
import antropy as ant
from sklearn.preprocessing import StandardScaler
import glob
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# 백그라운드 로깅 설정
log_file_path = '/content/gdrive/MyDrive/reforecast/timexer/preprocessing_log.txt'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

if not hasattr(builtins, '_real_original_print'):
    builtins._real_original_print = builtins.print

def dual_print(*args, **kwargs):
    builtins._real_original_print(*args, **kwargs)
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    text = sep.join(str(arg) for arg in args) + end
    with open(log_file_path, "a", encoding='utf-8') as f:
        f.write(text)

builtins.print = dual_print
print("🚀 최종 ESVD 병렬 전처리 및 로깅이 시작되었습니다.")

# ESVD 파이프라인
def esvd_pipeline(pv_power, vmd_k=4):
    noise_amplitude = 1e-4 * np.std(pv_power)
    if noise_amplitude == 0:
        noise_amplitude = 1e-4
    jitter = np.random.normal(0, noise_amplitude, size=pv_power.shape)
    pv_power_jittered = pv_power + jitter

    emd = EMD()
    emd.MAX_ITERATIONS = 2000
    imfs = emd.emd(pv_power_jittered)

    se_values = [ant.sample_entropy(imf) for imf in imfs]
    valid_se = [se for se in se_values if not np.isinf(se) and not np.isnan(se)]
    threshold = np.mean(valid_se) if len(valid_se) > 0 else 0

    high_comp, low_comp = [], []
    for i, se in enumerate(se_values):
        if np.isinf(se) or np.isnan(se) or se > threshold:
            high_comp.append(imfs[i])
        else:
            low_comp.append(imfs[i])

    if len(high_comp) == 0:
        high_comp = low_comp
        low_comp = []

    recon_high_comp = np.sum(high_comp, axis=0)
    recon_low_comp = np.sum(low_comp, axis=0) if len(low_comp) > 0 else np.zeros_like(pv_power)

    alpha = 2000; tau = 0; DC = 0; init = 1; tol = 1e-7
    u, _, _ = VMD(recon_high_comp, alpha, tau, vmd_k, DC, init, tol)

    features = [pv_power, recon_low_comp] + list(u)
    min_len = min(len(f) for f in features)
    features = [f[:min_len] for f in features]

    return np.vstack(features).T

# 윈도우 생성 (시간 추적 추가)
def process_single_window(i, pv_series, time_series, seq_len, vmd_k):
    window_features = esvd_pipeline(pv_series[i : i + seq_len], vmd_k=vmd_k)
    target_power = pv_series[i + seq_len]
    target_time = time_series[i + seq_len] # 해당 타겟의 원본 시간 추출
    return window_features, target_power, target_time

def process_all_windows_parallel(pv_series, time_series, seq_len=672, vmd_k=4, n_jobs=-1):
    total_len = len(pv_series)
    print(f"🔥 CPU 멀티프로세싱 가동 (할당 코어: {'최대치' if n_jobs == -1 else n_jobs})")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_window)(i, pv_series, time_series, seq_len, vmd_k)
        for i in range(total_len - seq_len)
    )

    X = [res[0] for res in results]
    Y = [res[1] for res in results]
    Y_time = [res[2] for res in results]

    print(f"✅ 데이터 변환 완료 (소요 시간: {time.time() - start_time:.1f}초)")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(Y_time, dtype='str')

# 메인 실행부
def main():
    target_sites = [1, 2, 4, 5, 6, 7, 8]
    base_path = "/content/gdrive/MyDrive/reforecast/dataset/solar_stations"
    save_path = "/content/gdrive/MyDrive/reforecast/timexer/data/esvd_features"

    SEQ_LEN = 672
    VMD_K = 4

    os.makedirs(save_path, exist_ok=True)

    for site_idx in target_sites:
        search_pattern = os.path.join(base_path, f"*site {site_idx}*.xlsx")
        matched_files = glob.glob(search_pattern)

        print(f"\n==========================================================")
        print(f"  [START] 사이트 {site_idx} 전체 윈도우 ESVD 전처리 시작 (순수 원본)")
        print(f"==========================================================")

        if not matched_files:
            continue

        file_path = matched_files[0]

        df = pd.read_excel(file_path)
        df = df.dropna(subset=['Power (MW)'])

        # 🚨 [수정 1] 0값 필터링 로직 제거! 밤 시간대(0)를 그대로 살려 시간 연속성 유지
        df = df.reset_index(drop=True)

        # Power 배열과 Time 배열을 각각 추출
        # 🚨 [수정 2] 스케일링 로직을 없애고 순수 발전량(MW)을 그대로 사용
        pv_power = df['Power (MW)'].values.astype(np.float32)
        time_series = df['Time(year-month-day h:m:s)'].astype(str).values

        # 윈도우별 ESVD 병렬 처리 (스케일링 안 된 pv_power 원본 투입)
        X, Y, Y_time = process_all_windows_parallel(pv_power, time_series, seq_len=SEQ_LEN, vmd_k=VMD_K, n_jobs=-1)

        # 저장 (하나의 파일로 통째로 저장)
        np.save(os.path.join(save_path, f'site_{site_idx}_X_esvd.npy'), X)
        np.save(os.path.join(save_path, f'site_{site_idx}_Y.npy'), Y)
        np.save(os.path.join(save_path, f'site_{site_idx}_Y_time.npy'), Y_time)

        print(f"✅ 사이트 {site_idx} 저장 완료!")
        print(f"   - X shape : {X.shape}")
        print(f"   - Y shape : {Y.shape}")
        print(f"   - Y_time shape : {Y_time.shape} (Time Mapping Array)")

    print("\n🎉 모든 사이트의 무결성 ESVD 전처리가 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    main()