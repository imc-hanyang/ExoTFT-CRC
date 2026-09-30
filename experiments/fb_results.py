"""
[6/7] fb_results — 통학습 Base 비교군 학습 / 테스트 + 채점

  - create_fb 와 동일한 학습 / Sequential Test 를 다시 수행해
    site{N}_base_fullbatch_m22.pt / site{N}_base_fullbatch_test_results.csv 를 덮어쓴 뒤 채점
  - 채점 : Site 7 이상 구간 제외 + 가동 시간 필터 + NaN(시간 매핑 실패) 제거
           + True_Target > 0.0001 (주간만), 분모 = capacity
"""

from common import TARGET_SITES, Config, setup_logging, print_stage, print_score_table, filter_eval_window, score
from create_fb import LOG_NAME, run_site as run_fullbatch_site


def evaluate(test_df, site_idx, capacity):
    """채점 → (NRMSE, NMAE, R², 샘플 수). 유효 샘플이 없으면 None"""
    valid_df = filter_eval_window(test_df, site_idx)
    valid_df = valid_df.dropna(subset=['True_Target', 'Base_Pred'])
    mask = valid_df['True_Target'].values > 0.0001
    yt = valid_df['True_Target'].values[mask]
    yp = valid_df['Base_Pred'].values[mask]
    if len(yt) == 0: return None
    return (*score(yt, yp, capacity), len(yt))


def main():
    setup_logging(LOG_NAME)
    print_stage('fb_results — 통학습 Base 비교군 학습 / 테스트 + 채점', [
        '학습: M1~M22 전체 통학습 (create_fb 와 동일, 결과 덮어씀)',
        '스케일러: M1~M22 (통학습 Base 전용)',
        '채점: 원본 CSV True_Target, 가동 시간 + True_Target > 0.0001',
        'capacity: 사이트별 설비 용량 (없으면 최대 실제 발전량)',
    ])
    cfg = Config(d_model=16, n_heads=1)

    for site_idx in TARGET_SITES:
        result = run_fullbatch_site(site_idx, cfg)
        if result is None: continue
        test_df, capacity = result

        metrics = evaluate(test_df, site_idx, capacity)
        if metrics is None:
            print('   ⚠️  유효 샘플 없음, 채점 스킵'); continue
        nrmse, nmae, r2, n_valid = metrics

        print_score_table(f'[Site {site_idx}] 통학습 Base 비교군 (M23~M24) | capacity {capacity:.4f} MW | 유효 샘플 {n_valid}',
                          [('[통학습 Base] M1~M22 Full-Batch', nrmse, nmae, r2)])
        print(f'\n   ✅ Site {site_idx} 완료')


if __name__ == '__main__':
    main()
