"""
loader.py
=========
사이트별 결과 CSV(모델 N개) + 원본 기상 데이터를 Time 기준으로 병합하고
모델별 잔차 파생 변수를 만들어 주는 공통 로더.

핵심 아이디어
------------
config.MODELS 에 등록된 모델만큼 자동으로 컬럼이 생깁니다.

    pred_<key>       예측값 [MW]
    resid_<key>      True_Target - pred          (+ 과소예측 / - 과대예측)
    abs_resid_<key>  |resid|

ExoTFT-CRC 전용 부가 컬럼
    Rolling_Base_Pred, Alpha_Used, Correction( = Final - Rolling = α·ê )

주의
----
* True_Target 은 base 파일 기준(원본 CSV 시간 매핑값)으로 통일합니다.
  adaptive 파일의 True_Target(스케일러 역변환값)은 버립니다.
* 채점 필터는 operation_hours 기준만 적용 (threshold 없음).
* Site 7 은 2020-12-14 08:00 ~ 2020-12-31 23:45 구간을 반드시 제외합니다.
"""

import os
import numpy as np
import pandas as pd

import config

# 원본에서 함께 실어올 기상/시간 피처
ORIGIN_FEATURES = [
    "GHI", "DNI", "TSI", "Temperature", "Atmospheric pressure",
    "hour", "sin(hour)", "cos(hour)", "month", "sin(month)", "cos(month)",
    "site Nominal capacity", "site Maximum capacity", config.POWER_COL,
]


# ══════════════════════════════════════════════════════════════════════════════
# 원본 데이터
# ══════════════════════════════════════════════════════════════════════════════
def load_origin(site: int, add_derived: bool = True) -> pd.DataFrame:
    """dataset/origin/site_{site}.csv 로드 (전 기간 2019~2020)."""
    df = pd.read_csv(os.path.join(config.ORIGIN_DIR, f"site_{site}.csv"))
    df["Time"] = pd.to_datetime(df[config.ORIGIN_TIME_COL])
    df = df.drop(columns=[config.ORIGIN_TIME_COL]).sort_values("Time").reset_index(drop=True)

    if add_derived:
        p = df[config.POWER_COL]
        df["persistence"] = p.shift(1)          # B0: 직전 15분 관측치
        df["diff_15min"] = p.diff()             # 15분 변화량
        for lag in (1, 4, 8, 96):
            df[f"lag_{lag}"] = p.shift(lag)
        # 청천지수 대용: 발전량 / 설비용량
        df["capacity_factor"] = p / df["site Nominal capacity"]
    return df


def nominal_capacity(site: int) -> float:
    df = pd.read_csv(os.path.join(config.ORIGIN_DIR, f"site_{site}.csv"),
                     usecols=["site Nominal capacity"], nrows=1)
    return float(df.iloc[0, 0])


# ══════════════════════════════════════════════════════════════════════════════
# 결과 병합
# ══════════════════════════════════════════════════════════════════════════════
def load_site(site: int,
              order=None,
              apply_operation_hours: "bool | None" = None,
              with_origin: bool = True,
              verbose: bool = True) -> pd.DataFrame:
    """
    등록된 모델 결과 + 원본 피처를 Time 기준 inner join 후 잔차 파생 변수 생성.

    apply_operation_hours=None 이면 config.APPLY_OPERATION_HOURS (기본 False) 를 따릅니다.

    Returns
    -------
    DataFrame  (Time, True_Target, pred_*, resid_*, abs_resid_*, 기상 피처 …)
    """
    if apply_operation_hours is None:
        apply_operation_hours = config.APPLY_OPERATION_HOURS

    display_specs = config.model_specs(order=order, site=site, available_only=False)
    display_keys = {s.key for s in display_specs}

    # 병합 순서 규칙
    #  1) True_Target 은 반드시 **ExoTFT-FB 파일** 기준이어야 하므로 맨 앞에 둡니다
    #     (adaptive 파일의 True_Target 은 스케일러 역변환값이라 섞으면 안 됨).
    #  2) base / rolling 두 기준선 컬럼은 스위치와 무관하게 항상 만들어 둡니다.
    #     USE_ROLLING_AS_BASE 로 표시 대상만 바뀌고, 다른 스크립트가 참조하는
    #     abs_resid_base / abs_resid_rolling 이 사라지지 않도록.
    always = [config.FULLBATCH_BASE_KEY, "rolling"]
    specs = [config.MODELS[k] for k in always if k in config.MODELS]
    specs += [s for s in display_specs if s.key not in {sp.key for sp in specs}]

    merged = None
    used = []
    for spec in specs:
        if not spec.exists(site):
            if verbose and spec.file is not None:
                print(f"   [skip] site{site} / {spec.label}: 결과 파일 없음 → "
                      f"{os.path.basename(spec.path(site))}")
            continue
        path = spec.path(site)

        df = pd.read_csv(path)
        df["Time"] = pd.to_datetime(df["Time"])

        cols = ["Time", spec.pred_col] + [c for c in spec.extra_cols if c in df.columns]
        sub = df[cols].rename(columns={spec.pred_col: spec.pred})

        if merged is None:
            # 첫 모델 파일에서 True_Target 기준값을 가져옴
            true_src = df[["Time", "True_Target"]] if "True_Target" in df.columns else None
            merged = sub if true_src is None else true_src.merge(sub, on="Time", how="inner")
        else:
            merged = merged.merge(sub, on="Time", how="inner")
        used.append(spec)

    if merged is None:
        raise FileNotFoundError(f"site {site}: 사용할 수 있는 결과 파일이 하나도 없습니다.")

    # ── 원본 기상 피처 병합 (True_Target 도 원본 Power 기준으로 보정) ──────────
    if with_origin:
        origin = load_origin(site)
        keep = ["Time"] + [c for c in ORIGIN_FEATURES + ["persistence", "diff_15min",
                                                         "lag_1", "lag_4", "lag_8", "lag_96",
                                                         "capacity_factor"]
                           if c in origin.columns]
        merged = merged.merge(origin[keep], on="Time", how="inner")
        if "True_Target" not in merged.columns:
            merged["True_Target"] = merged[config.POWER_COL]

    # ── Site 7 예외 구간 제거 ────────────────────────────────────────────────
    if site == 7:
        s, e = pd.to_datetime(config.SITE7_EXCLUDE[0]), pd.to_datetime(config.SITE7_EXCLUDE[1])
        merged = merged[~merged["Time"].between(s, e)]

    # ── 가조시간 필터 ────────────────────────────────────────────────────────
    if apply_operation_hours:
        st, et = config.OPERATION_HOURS[site]
        hhmm = merged["Time"].dt.strftime("%H:%M")
        merged = merged[(hhmm >= st) & (hhmm <= et)]

    merged = merged.sort_values("Time").reset_index(drop=True)

    # ── 모델별 잔차 파생 변수 ────────────────────────────────────────────────
    for spec in used:
        merged[spec.resid] = merged["True_Target"] - merged[spec.pred]
        merged[spec.abs_resid] = merged[spec.resid].abs()

    # ── ExoTFT-CRC 전용 진단 변수 ─────────────────────────────────────────────────
    reco = config.MODELS[config.RECO_KEY]
    if reco.pred in merged.columns and "Rolling_Base_Pred" in merged.columns:
        merged["Correction"] = merged[reco.pred] - merged["Rolling_Base_Pred"]
        merged["resid_rolling"] = merged["True_Target"] - merged["Rolling_Base_Pred"]
        merged["abs_resid_rolling"] = merged["resid_rolling"].abs()

    base = config.MODELS[config.BASE_KEY]
    if base.abs_resid in merged.columns and reco.abs_resid in merged.columns:
        merged["Improvement"] = merged[base.abs_resid] - merged[reco.abs_resid]
        merged["Improved"] = (merged["Improvement"] > 0).astype(int)

    merged.attrs["site"] = site
    # 컬럼은 다 만들어 두되, 그림에 그릴 목록은 스위치가 정한 표시 순서만.
    merged.attrs["specs"] = [s for s in used if s.key in display_keys]
    merged.attrs["all_specs"] = used
    merged.attrs["capacity"] = float(merged["site Nominal capacity"].iloc[0]) \
        if "site Nominal capacity" in merged.columns else nominal_capacity(site)
    return merged


def available_specs(df: pd.DataFrame):
    """load_site 결과에 실제로 들어간 ModelSpec 리스트."""
    return df.attrs.get("specs", [])


# ══════════════════════════════════════════════════════════════════════════════
# 평가 지표
# ══════════════════════════════════════════════════════════════════════════════
def metrics(y_true, y_pred, capacity: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    err = y_true - y_pred
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "MAE":   float(np.abs(err).mean()),
        "RMSE":  float(np.sqrt((err ** 2).mean())),
        "NMAE":  float(np.abs(err).mean() / capacity * 100.0),          # [%] of capacity
        "NRMSE": float(np.sqrt((err ** 2).mean()) / capacity * 100.0),  # [%] of capacity
        "R^2":   float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def metric_table(sites=None, order=None, include_persistence: bool = False,
                 apply_operation_hours: bool = True) -> pd.DataFrame:
    """사이트 × 모델 성능 표.

    채점용이므로 가조시간 필터를 기본 적용합니다 (config.APPLY_OPERATION_HOURS 와 무관).
    fig4 는 dataset/results/model_total_results.csv 를 쓰므로 이 함수를 사용하지 않습니다.
    """
    sites = sites or config.sites
    rows = []
    for site in sites:
        df = load_site(site, order=order, verbose=False,
                       apply_operation_hours=apply_operation_hours)
        cap = df.attrs["capacity"]
        cand = []
        if include_persistence and "persistence" in df.columns:
            cand.append(("B0: Persistence", "persistence"))
        cand += [(s.plain, s.pred) for s in available_specs(df)]
        for label, col in cand:
            r = metrics(df["True_Target"], df[col], cap)
            r.update(site=site, model=label, n=len(df), capacity=cap)
            rows.append(r)
    return pd.DataFrame(rows)[["site", "model", "n", "capacity",
                               "MAE", "RMSE", "NMAE", "NRMSE", "R^2"]]


if __name__ == "__main__":
    print(metric_table(include_persistence=True).to_string(index=False))
