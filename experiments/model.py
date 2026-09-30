"""
R-ECO 실험 파이프라인 전체 실행 (Colab)

실행 순서 — 각 단계는 앞 단계 산출물(SAVE_PATH)을 사용한다.
  1. base_N_oof        Base 학습 + 롤링 OOF 잔차 + ESVD 피처
  2. res_model_5_fold  잔차 모델 5-Fold OOF 교차 검증
  3. init_alpha        OOF 기반 best alpha 탐색
  4. create_rf         Rolling Base + 잔차 (고정 alpha) 최종 테스트
  5. create_fb         통학습 Base 비교군 학습 / 테스트
  6. fb_results        통학습 Base 비교군 학습 / 테스트 + 채점
  7. crc_create_N_results  Rolling Base + 잔차 (Adaptive alpha) 최종 테스트 + 채점

단계별 로그는 common.LOG_DIR 아래 각 모듈의 LOG_NAME 파일에 기록된다.
"""

from google.colab import drive
drive.mount('/content/gdrive')

import time

import base_N_oof, res_model_5_fold, init_alpha, create_rf, create_fb, fb_results, crc_create_N_results
from common import device, setup_logging

PIPELINE = [
    ('base_N_oof',       base_N_oof),
    ('res_model_5_fold', res_model_5_fold),
    ('init_alpha',       init_alpha),
    ('create_rf',        create_rf),
    ('create_fb',        create_fb),
    ('fb_results',       fb_results),
    ('crc_create_N_results', crc_create_N_results),
]
LOG_NAME = 'local_pipeline_log.txt'


def _fmt_elapsed(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}h {m:02d}m {s:02d}s'


def main():
    setup_logging(LOG_NAME)
    n = len(PIPELINE)
    print('\n' + '#' * 80)
    print(f'🚀 R-ECO 파이프라인 시작 | device: {device}')
    for i, (name, _) in enumerate(PIPELINE, 1):
        print(f'   {i}. {name}')
    print('#' * 80)

    t_total = time.time()
    elapsed = []
    for i, (name, module) in enumerate(PIPELINE, 1):
        setup_logging(module.LOG_NAME)   # 단계 배너도 해당 단계 로그에 기록
        print('\n\n' + '#' * 80)
        print(f'▶ [{i}/{n}] {name}')
        print('#' * 80)
        t0 = time.time()
        module.main()
        elapsed.append((name, time.time() - t0))
        print(f'\n⏱  [{i}/{n}] {name} 완료 — {_fmt_elapsed(elapsed[-1][1])}')

    setup_logging(LOG_NAME)
    print('\n' + '#' * 80)
    print('🏁 R-ECO 파이프라인 완료')
    for name, sec in elapsed:
        print(f'   {name:<22} {_fmt_elapsed(sec)}')
    print(f'   {"total":<22} {_fmt_elapsed(time.time() - t_total)}')
    print('#' * 80)


if __name__ == '__main__':
    main()
