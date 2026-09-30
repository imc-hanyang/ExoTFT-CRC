"""
Adaptive α 와 기상 / PV power 입력의 연관성 검정 (figure 아님, 콘솔 표 출력).

배경 — 반드시 알고 분석해야 하는 α 의 구조
-------------------------------------------
Alpha_Used 는 15분 스텝별로 움직이는 값이 아닙니다. 실측 확인 결과:

  * 하루(96 스텝) 단위 **계단 함수**. 갱신 시각은 사이트별 고정
    (site 4 = 12:00, site 8 = 08:00, 그 외 = 16:00 — 롤링 재학습 경계).
  * 갱신은 **곱셈형(AIMD)**: 직전 값 ×1.2 (성공) 또는 ×0.5 (실패), 상·하한 clip.
    (0.31 → 0.372 → 0.4464 → 0.53568 → cap 0.62 처럼 등비수열로 나타남)

따라서
  (1) 분석 단위는 **일(day)** 이어야 합니다. 스텝 단위로 α 와 GHI 를 상관내면
      α 는 하루 내 상수인데 GHI 는 일주기로 크게 변하므로, α 가 반응할 수 없는
      하루 내 변동이 상관계수를 지배하는 허위 상관이 됩니다.
  (2) α_d 는 갱신 시점 이전 정보로 정해지므로 **전일(prev-day)** 조건과 대응시킵니다.
  (3) 테스트 구간이 11~12월이라 기온·일사가 단조 하강합니다
      (ρ(Temp, 경과일) = -0.37 ~ -0.92). **경과일 추세를 반드시 통제**해야 합니다.

출력 항목
---------
  0. 추세 진단        : 각 변수 vs 테스트 경과일 Spearman ρ
  1. 부분 상관        : ρ(α, 전일 피처 | 경과일)
  2. 상승일 vs 하락일 : 전일 조건 Mann-Whitney (rank-biserial 효과크기)
  3. 통합 로지스틱    : 사이트 내 순위 정규화 + 사이트 고정효과 + 경과일 통제.
                        기상 변수의 효과가 '전일 예측 난이도(ExoTFT-RF NRMSE)' 를 넣으면
                        사라지는지(매개) 확인하는 것이 핵심.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, norm

import config
import loader

SITES = config.sites

# 기상 입력 / PV power 입력 / 대조군(예측 난이도·보정 성과)
WEATHER = ["GHI", "DNI", "TSI", "Temp", "Pres"]
PVPOWER = ["cf_mean", "cf_max", "clearness", "vol", "vol_std"]
CONTROL = ["nmae_base", "nmae_roll", "nrmse_base", "nrmse_roll", "improv", "hit_rate"]
FEATS = WEATHER + PVPOWER + CONTROL

# '전일 예측 난이도' 대표 지표 — **ExoTFT-RF(보정 전) 기준**.
#
# α 갱신 규칙이 평가하는 대상은 "ExoTFT-RF 잔차를 α·ê 로 얼마나 잘 상쇄했나"
# 입니다. 따라서 '전일이 얼마나 어려운 하루였나' 도 보정이 실제로 올라타는 기준선,
# 즉 ExoTFT-RF 의 오차로 재야 대조군의 층위가 α 의 판정 층위와 일치합니다.
# ExoTFT-FB 오차를 쓰면 롤링 재학습 이전 모델의 난이도라 α 가 마주한 잔차와
# 한 단계 어긋나고, 그림(5_e)의 라벨과 본문 서술도 기준선이 뒤바뀝니다.
# (config.USE_ROLLING_AS_BASE = True 와도 이쪽이 일관됩니다. 나머지 세 조합
#  nmae_base / nmae_roll / nrmse_base 는 FEATS 에 그대로 있어 콘솔 표에서
#  민감도를 함께 확인할 수 있습니다.)
# 지표 선택: "nmae_roll"(평균 절대) 또는 "nrmse_roll"(제곱평균, 극단 오차 가중)
DIFFICULTY = "nrmse_roll"

MIN_STEPS = 90          # 잘린 첫/마지막 블록 제외 (96 스텝 블록 기준)

# α 가 감쇠한 구간의 α* 신뢰성에 대하여
# ------------------------------------
# α ≈ 1e-6 이면 Correction 도 1e-5 MW 수준까지 작아지지만, 결과 CSV 가 float64
# 를 전체 유효자릿수로 저장하므로 Σ(eC)/ΣC² 의 상대 정밀도는 유지됩니다.
# (실측: 붕괴 구간에서도 블록당 40~96 스텝의 Correction 이 0 이 아니고,
#  α* 가 −1.3 ~ +1.8 범위에서 매끄럽게 변합니다 — 반올림 잡음이 아님.)
# 따라서 크기 기준으로 블록을 버리지 않습니다. 분모가 정확히 0 인 블록,
# 즉 그날 보정이 한 번도 일어나지 않은 경우만 _alpha_opt 가 NaN 을 냅니다.


def _alpha_opt(x: pd.DataFrame) -> float:
    """블록 하나의 사후 최적 α. 자세한 정의는 daily_table 주석 참조."""
    c = x["Correction"].to_numpy(dtype=float)
    e = x["resid_rolling"].to_numpy(dtype=float)
    a = float(x["Alpha_Used"].iloc[0])
    den = float((c ** 2).sum())
    if not np.isfinite(den) or den <= 0.0:
        return np.nan
    return a * float((e * c).sum()) / den


# ══════════════════════════════════════════════════════════════════════════════
# 일(day) 단위 집계
# ══════════════════════════════════════════════════════════════════════════════
def alpha_blocks(df: pd.DataFrame) -> pd.Series:
    """α 갱신 시각을 하루의 시작으로 보고 24h 블록 id 를 부여."""
    ch = df.loc[df["Alpha_Used"].diff().abs() > 1e-12, "Time"]
    off = ch.dt.strftime("%H:%M").mode()[0] if len(ch) else "00:00"
    h, m = map(int, off.split(":"))
    return (df["Time"] - pd.Timedelta(hours=h, minutes=m)).dt.floor("D")


def daily_table(site: int) -> pd.DataFrame:
    df = loader.load_site(site, verbose=False)
    cap = df.attrs["capacity"]
    d = df.assign(_b=alpha_blocks(df)).copy()
    d["cf"] = d[config.POWER_COL] / cap             # capacity factor
    d["dvol"] = d["diff_15min"].abs() / cap         # 15분 변동폭 (간헐성)

    g = d.groupby("_b")
    gd = d[d["GHI"] > 10].groupby("_b")             # 주간만 쓰는 피처
    t = pd.DataFrame({
        "alpha": g["Alpha_Used"].first(),
        "n": g.size(),
        # ── 기상 입력 ──
        "GHI": gd["GHI"].mean(),
        "DNI": gd["DNI"].mean(),
        "TSI": gd["TSI"].mean(),
        "Temp": g["Temperature"].mean(),
        "Pres": g["Atmospheric pressure"].mean(),
        # ── PV power 입력 ──
        "cf_mean": gd["cf"].mean(),
        "cf_max": g["cf"].max(),
        # 일사량 대비 발전 효율 (청천지수 대용)
        "clearness": gd.apply(lambda x: x["cf"].mean() / max(x["GHI"].mean(), 1e-9) * 1000,
                              include_groups=False),
        "vol": gd["dvol"].mean(),
        "vol_std": gd["dvol"].std(),
        # ── 대조군: 전일 예측 난이도 / 보정 성과 ──
        "nmae_base": g["abs_resid_base"].mean() / cap * 100,
        "nmae_roll": g["abs_resid_rolling"].mean() / cap * 100,
        # NRMSE — 큰 오차에 제곱 가중이 붙어 '하루 중 크게 틀린 순간' 을 더 반영.
        # NMAE 와 같은 난이도 지표지만 극단 오차에 민감한 변형입니다.
        "nrmse_base": g["resid_base"].apply(lambda x: (x ** 2).mean() ** 0.5) / cap * 100,
        "nrmse_roll": g["resid_rolling"].apply(lambda x: (x ** 2).mean() ** 0.5) / cap * 100,
        # ExoTFT-CRC(보정 후) 오차. α 가 곱해진 결과물이라 α 방향의 '설명변수'로 쓰면
        # 순환이 되므로 FEATS 에는 넣지 않고 그림 표시용으로만 둡니다.
        "nmae_reco": g["abs_resid_reco"].mean() / cap * 100,
        "improv": g["Improvement"].mean() / cap * 100,
        "hit_rate": g["Improved"].mean(),
        # ── 사후 최적 α (오라클) ──────────────────────────────────────────
        # 그날의 ExoTFT-RF 잔차 e 를 예측 잔차 ê 로 가장 잘 상쇄했을 계수.
        #   α* = argmin_α Σ(e − α·ê)²  =  Σ(e·ê) / Σ(ê²)
        # ê = Correction / α_used 이므로 α 로 나누지 않는 등가식
        #   α* = α_used · Σ(e·C) / Σ(C²)      (C = Correction = α·ê)
        # 을 씁니다. α 가 0 에 가까울 때 1/α 를 태우지 않아 조건수가 낫습니다.
        "alpha_opt": g.apply(_alpha_opt, include_groups=False),
    })
    t = t[t["n"] >= MIN_STEPS].reset_index(drop=True)
    t["site"] = site
    t["day_idx"] = np.arange(len(t), dtype=float)
    return t


def _fmt(r, p, thr=0.05):
    if not np.isfinite(r):
        return "  -  "
    return f"{r:+.2f}{'*' if p < thr else ' '}"


# ══════════════════════════════════════════════════════════════════════════════
# 0. α 구조 요약
# ══════════════════════════════════════════════════════════════════════════════
def report_structure(tabs):
    rows = []
    for s, t in tabs.items():
        d = t["alpha"].diff()
        a = t["alpha"]
        rows.append({
            "site": s, "days": len(t),
            "α min": round(a.min(), 3), "α max": round(a.max(), 3),
            "α 상한 체류율": round((a >= a.max() - 1e-9).mean(), 2),
            "상승": int((d > 1e-12).sum()), "하락": int((d < -1e-12).sum()),
            "유지": int((d.abs() <= 1e-12).sum()),
        })
    print("### α 구조 (일 단위)")
    print(pd.DataFrame(rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 1. 추세 진단 + 부분 상관
# ══════════════════════════════════════════════════════════════════════════════
def report_trend(tabs):
    rows = []
    for s, t in tabs.items():
        idx = t["day_idx"].values
        r = {"site": s, "alpha": f"{spearmanr(idx, t['alpha'])[0]:+.2f}"}
        for f in ["GHI", "Temp", "cf_mean", "vol", DIFFICULTY]:
            m = np.isfinite(t[f].values)
            r[f] = f"{spearmanr(idx[m], t[f].values[m])[0]:+.2f}"
        rows.append(r)
    print("\n### 0. 추세 진단 — Spearman ρ(·, 테스트 경과일)")
    print("    |ρ| 이 큰 변수는 계절 추세와 얽혀 있어 통제 없이는 해석 불가")
    print(pd.DataFrame(rows).to_string(index=False))


def partial_rho(x, y, z):
    """순위 변환 후 z 를 회귀로 제거한 잔차 간 상관 (rank-based partial)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10:
        return np.nan, np.nan
    rx, ry, rz = (pd.Series(v[m]).rank().values for v in (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    if np.std(resid(rx, rz)) == 0 or np.std(resid(ry, rz)) == 0:
        return np.nan, np.nan
    return spearmanr(resid(rx, rz), resid(ry, rz))


def report_partial(tabs):
    rows = []
    for s, t in tabs.items():
        r = {"site": s}
        for f in FEATS:
            r[f] = _fmt(*partial_rho(t[f].shift(1).values, t["alpha"].values,
                                     t["day_idx"].values))
        rows.append(r)
    print("\n### 1. 부분 Spearman ρ(α, 전일 피처 | 경과일)     * p<0.05")
    print(pd.DataFrame(rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 2. 상승일 vs 하락일
# ══════════════════════════════════════════════════════════════════════════════
def report_updown(tabs, min_group=5):
    rows = []
    for s, t in tabs.items():
        d = t["alpha"].diff()
        up, dn = (d > 1e-12).values, (d < -1e-12).values
        r = {"site": s, "n_up": int(up.sum()), "n_dn": int(dn.sum())}
        if up.sum() >= min_group and dn.sum() >= min_group:
            for f in FEATS:
                x = t[f].shift(1).values
                a, b = x[up & np.isfinite(x)], x[dn & np.isfinite(x)]
                if len(a) < min_group or len(b) < min_group:
                    r[f] = "  -  "
                    continue
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                r[f] = _fmt(2 * u / (len(a) * len(b)) - 1, p)   # rank-biserial
        rows.append(r)
    print("\n### 2. α 상승일 vs 하락일 — 전일 조건 비교")
    print("    rank-biserial 효과크기, + 면 상승일에서 그 값이 큼     * p<0.05")
    print(pd.DataFrame(rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 3. 통합 로지스틱 (매개 검정)
# ══════════════════════════════════════════════════════════════════════════════
def pooled_design(tabs):
    """사이트 내 순위 정규화 → 전일 피처로 시프트 → 상승/하락 라벨."""
    out = []
    for s, t in tabs.items():
        z = t.copy()
        for c in FEATS:
            z[c] = z[c].rank(pct=True)
        z["day_idx"] = z["day_idx"].rank(pct=True)
        prev = z[FEATS].shift(1)
        prev["day_idx"] = z["day_idx"]          # 추세는 당일 기준
        prev["y"] = np.sign(z["alpha"].diff())
        prev["site"] = s
        out.append(prev)
    P = pd.concat(out).dropna()
    P = P[P["y"] != 0].copy()
    P["y"] = (P["y"] > 0).astype(int)
    return P


def logit(P, cols):
    """사이트 고정효과 포함 로지스틱 (Newton-Raphson, Wald z)."""
    X = [np.ones(len(P))] + [P[c].values for c in cols]
    for s in sorted(P["site"].unique())[1:]:
        X.append((P["site"] == s).values.astype(float))
    X = np.column_stack(X)
    y = P["y"].values
    b = np.zeros(X.shape[1])
    for _ in range(300):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(len(b)),
                                X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    W = np.clip(p * (1 - p), 1e-9, None)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(len(b)))))
    k = len(cols)
    z = b[1:1 + k] / se[1:1 + k]
    return pd.DataFrame({"var": cols, "coef": b[1:1 + k], "se": se[1:1 + k],
                         "z": z,
                         "p": [2 * (1 - norm.cdf(abs(v))) for v in z]})


def _logit_str(res: pd.DataFrame) -> str:
    """logit() 결과를 콘솔 표로."""
    out = res.copy()
    out["coef"] = out["coef"].round(2)
    out["z"] = out["z"].round(2)
    out["p"] = out["p"].map(lambda v: f"{v:.3f}")
    return out[["var", "coef", "z", "p"]].to_string(index=False)


MODELS_TO_FIT = [
    ["Temp", "day_idx"],
    ["vol", "day_idx"],
    [DIFFICULTY, "day_idx"],
    ["GHI", "Temp", "cf_mean", "vol", "day_idx"],
    ["vol", DIFFICULTY, "day_idx"],                       # ← 매개 검정 핵심
    ["GHI", "Temp", "cf_mean", "vol", DIFFICULTY, "day_idx"],
]


def report_logit(tabs):
    P = pooled_design(tabs)
    print(f"\n### 3. 통합 로지스틱 — α 상승(1) vs 하락(0)")
    print(f"    n = 상승 {int(P['y'].sum())}일 / 하락 {int((1 - P['y']).sum())}일, "
          f"사이트 고정효과 + 경과일 통제, 모든 피처는 사이트 내 백분위 순위")
    for cols in MODELS_TO_FIT:
        print(f"\n  -- 설명변수 {cols}")
        print(_logit_str(logit(P, cols)))


def main():
    tabs = {}
    for s in SITES:
        try:
            tabs[s] = daily_table(s)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] site {s}: {e}")
    if not tabs:
        print("사용 가능한 사이트가 없습니다.")
        return
    report_structure(tabs)
    report_trend(tabs)
    report_partial(tabs)
    report_updown(tabs)
    report_logit(tabs)


if __name__ == "__main__":
    main()
