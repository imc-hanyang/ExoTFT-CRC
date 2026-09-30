"""
Table. 사이트별 Adaptive α 구조 요약 (5_c 윗패널의 표 버전).

α 는 하루 단위로만 갱신되므로 5_c 윗패널의 '시간대별 평균 α' 는 사실상
사이트별 평균 α 와 같습니다. 그 값을 표로 옮기고, α 가 어떻게 움직였는지를
시작값/종료값/변동 횟수로 함께 보여줍니다.

  alpha_start / alpha_end   테스트 첫날 · 마지막날의 α (일 블록 기준)
  alpha_mean  / alpha_sd    전체 테스트일에 대한 평균 · 표준편차
  up / down / hold          전일 대비 α 가 오른 · 내린 · 그대로인 날의 수
                            (합 = 테스트일 수 − 1. 첫날은 비교 대상이 없어 제외)

α 는 0 에 붙어도 정확한 0 이 아니라 1e-6 수준까지 감쇠합니다(사이트 4·5·8).
반올림해 0.000 으로 적으면 '보정이 정확히 꺼졌다' 로 읽히므로 유효숫자
6 자리(%.6g)로 그대로 씁니다.
"""
import os

import numpy as np
import pandas as pd

import config
import alpha_association as aa

OUT = "alpha_structure.csv"
TOL = 1e-12          # α 변동으로 볼 최소 차이


def build() -> pd.DataFrame:
    rows = []
    for site in aa.SITES:
        try:
            t = aa.daily_table(site)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] site {site}: {e}")
            continue
        a = t["alpha"]
        d = a.diff()                       # 첫날은 NaN → up/down/hold 어디에도 안 들어감
        rows.append({
            "site": site,
            "alpha_start": float(a.iloc[0]),
            "alpha_end": float(a.iloc[-1]),
            "alpha_mean": float(a.mean()),
            "alpha_sd": float(a.std()),
            "up": int((d > TOL).sum()),
            "down": int((d < -TOL).sum()),
            "hold": int((d.abs() <= TOL).sum()),
        })
    return pd.DataFrame(rows)


def main():
    df = build()
    out = os.path.join(config.SAVE_DIR_MAIN, OUT)
    df.to_csv(out, index=False, float_format="%.6g")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
