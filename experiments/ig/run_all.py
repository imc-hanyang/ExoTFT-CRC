"""
IG 분석 일괄 실행

  1. fb  : 통학습 Base IG      (선행: create_fb)
  2. crc : R-ECO Adaptive IG  (선행: base_N_oof, init_alpha, crc_create_N_results)

사용: python experiments/ig/run_all.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # experiments/ig/
import fb
import crc


def main():
    fb.main()
    crc.main()


if __name__ == '__main__':
    main()
