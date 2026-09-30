"""
R-ECO 실험 공통 모듈 — experiments/*.py 가 모두 공유

  - 경로 / 사이트 설정
  - 로그(dual print) 설정 + 출력 포맷 헬퍼
  - Config, 모델 구조 (BaseModel, ResidualModel)
  - Dataset (PVDataset, ResidualDataset)
  - 학습 / 추론 / 채점 유틸

데이터 인덱스 규칙
  - 샘플 idx 의 입력 = npy_X[idx] (lookback 672스텝), 타깃 = npy_Y[idx]
  - 기상/시간 피처는 df 의 idx + lookback 시점에서 가져온다
  - 전체 샘플을 24등분해 1개월(M) 단위로 사용: M1~M22 학습용, M23~M24 테스트
"""

import os, sys, builtins, unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from vmdpy import VMD
    HAS_VMDPY = True
except ImportError:
    HAS_VMDPY = False


# =========================================================================== #
# 경로 / 사이트 설정 (Colab 기준)
# =========================================================================== #
BASE_DIR    = '/content/gdrive/MyDrive/reforecast'
ORIGIN_PATH = os.path.join(BASE_DIR, 'timexer', 'data', 'origin')         # site_{N}.csv
ESVD_PATH   = os.path.join(BASE_DIR, 'timexer', 'data', 'esvd_features')  # site_{N}_X_esvd.npy 등 (preprocess 산출물)
SAVE_PATH   = os.path.join(BASE_DIR, 'timexer', 'data', 'results')        # 모든 실험 산출물
LOG_DIR     = os.path.join(BASE_DIR, 'timexer')                           # 단계별 로그 txt

TARGET_SITES = [1, 2, 4, 5, 6, 7, 8]

# 채점용 사이트별 가동 시간 (HH:MM 문자열 비교)
OPERATION_HOURS = {
    1: ('06:00', '21:30'), 2: ('00:00', '23:59'), 4: ('00:00', '23:59'),
    5: ('00:00', '23:59'), 6: ('06:00', '21:00'), 7: ('06:00', '21:00'), 8: ('06:00', '19:00'),
}
# Site 7 데이터 이상 구간 (채점에서 제외)
SITE7_EXCLUDE = ('2020-12-14 08:00:00', '2020-12-31 23:45:00')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def result_path(site_idx, suffix):
    """SAVE_PATH/site{N}_{suffix}"""
    return os.path.join(SAVE_PATH, f'site{site_idx}_{suffix}')


def site_paths(site_idx):
    """사이트별 입력 데이터 + 단계 간 공유 산출물 경로"""
    return {
        'csv':          os.path.join(ORIGIN_PATH, f'site_{site_idx}.csv'),
        'X':            os.path.join(ESVD_PATH,   f'site_{site_idx}_X_esvd.npy'),
        'Y':            os.path.join(ESVD_PATH,   f'site_{site_idx}_Y.npy'),
        'T':            os.path.join(ESVD_PATH,   f'site_{site_idx}_Y_time.npy'),
        'base_trained': result_path(site_idx, 'base_trained.pt'),       # base_N_oof
        'base_rolling': result_path(site_idx, 'base_rolling_m22.pt'),   # base_N_oof
        'oof_preds':    result_path(site_idx, 'oof_preds.pt'),          # base_N_oof
        'residuals':    result_path(site_idx, 'residuals.pt'),          # base_N_oof
        'esvd_windows': result_path(site_idx, 'esvd_windows.pt'),       # base_N_oof
        'best_alpha':   result_path(site_idx, 'best_alpha_value.csv'),  # init_alpha
    }


def missing_files(paths, keys):
    """paths[key] 중 존재하지 않는 파일의 basename 목록"""
    return [os.path.basename(paths[k]) for k in keys if not os.path.exists(paths[k])]


def load_npy(paths):
    """(npy_X, npy_Y, npy_Y_time)"""
    return np.load(paths['X']), np.load(paths['Y']), np.load(paths['T'], allow_pickle=True)


def load_oof_tensors(paths):
    """base_N_oof 산출물 (oof_preds, residuals, esvd_windows) — CPU 로드"""
    return (torch.load(paths['oof_preds'],    map_location='cpu'),
            torch.load(paths['residuals'],    map_location='cpu'),
            torch.load(paths['esvd_windows'], map_location='cpu'))


def steps_per_month(n_total):
    """전체 샘플을 24개월로 나눈 1개월당 스텝 수"""
    return max(1, n_total // 24)


def get_capacity(df):
    """채점 분모: 설비 용량 컬럼이 있으면 사용, 없으면 실제 최대 발전량"""
    if 'site Maximum capacity' in df.columns: return df['site Maximum capacity'].iloc[0]
    elif 'Maximum capacity (MW)' in df.columns: return df['Maximum capacity (MW)'].iloc[0]
    else: return df['Power (MW)'].max()


# =========================================================================== #
# 로그 / 출력 포맷
# =========================================================================== #
_log_file_path = None


def _dual_print(*args, **kwargs):
    """콘솔 출력 + 로그 파일 기록. '\\r' 이 포함된 진행률 출력은 파일에 남기지 않는다."""
    builtins._original_print_backup(*args, **kwargs)
    text = kwargs.get('sep', ' ').join(str(arg) for arg in args) + kwargs.get('end', '\n')
    if '\r' in text: return
    with open(_log_file_path, 'a', encoding='utf-8') as f: f.write(text); f.flush()


def setup_logging(log_name):
    """이후 print 를 LOG_DIR/log_name 에도 기록. 단계마다 호출해 로그 파일을 전환한다."""
    global _log_file_path
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)
    _log_file_path = os.path.join(LOG_DIR, log_name)
    if not hasattr(builtins, '_original_print_backup'):
        builtins._original_print_backup = builtins.print
    builtins.print = _dual_print


def print_stage(title, details=()):
    """단계 헤더"""
    print('\n' + '=' * 80)
    print(f'🌍 {title}')
    for line in details:
        print(f'  - {line}')
    print('=' * 80)


def print_site(site_idx, title):
    """사이트 헤더"""
    print(f'\n{"-" * 70}\n📊 [Site {site_idx}] {title}\n{"-" * 70}')


def _pad(text, width):
    """한글(전각) 폭을 고려한 왼쪽 정렬"""
    w = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)
    return text + ' ' * max(0, width - w)


def print_score_table(title, rows):
    """rows: [(모델명, NRMSE, NMAE, R²), ...]"""
    print(f'\n   🏆 {title}')
    print(f'   {_pad("모델", 42)} {"NRMSE":>8} {"NMAE":>8} {"R²":>8}')
    print(f'   {"-" * 70}')
    for name, nrmse, nmae, r2 in rows:
        print(f'   {_pad(name, 42)} {nrmse:>7.3f}% {nmae:>7.3f}% {r2:>8.4f}')
    print(f'   {"-" * 70}')


def _print_epoch(desc, epoch, val_loss, counter, patience):
    """epoch 진행률 (한 줄 덮어쓰기, 로그 파일에는 남지 않음)"""
    sys.stdout.write(f'\r      - [{desc}] Epoch {epoch + 1:03d} | Val Loss: {val_loss:.5f} (Pat: {counter}/{patience})')
    sys.stdout.flush()


def _print_train_summary(desc, best_loss, best_epoch, n_epochs):
    print(f'      ➤ [{desc}] 학습 종료 | best epoch {best_epoch:03d}/{n_epochs:03d} | best val loss {best_loss:.5f}')


# =========================================================================== #
# Config
# =========================================================================== #
@dataclass
class Config:
    lookback: int = 672            # Base 입력 길이 (15분 × 672 = 7일)
    horizon: int = 1               # 1스텝 예측
    residual_lookback: int = 96    # 잔차 윈도우 길이 R
    k_imfs: int = 4                # VMD 모드 수 K
    n_endo_channels: int = 6       # npy_X 채널 수 (0: Power, 1~: ESVD 성분)
    patch_len: int = 16; patch_stride: int = 8
    weather_vars: List[str] = field(default_factory=lambda: ['GHI', 'DNI', 'TSI', 'Temperature', 'Atmospheric pressure'])
    temporal_vars: List[str] = field(default_factory=lambda: ['hour', 'sin(hour)', 'cos(hour)', 'month', 'sin(month)', 'cos(month)'])
    d_model: int = 16; n_heads: int = 1; n_fusion_layers: int = 2; dropout: float = 0.1
    quantiles: Optional[List[float]] = None

    @property
    def weather_len(self) -> int: return 1                 # 현재 시점 기상값 1개
    @property
    def temporal_len(self) -> int: return self.horizon     # 예측 시점 시간 피처
    @property
    def n_outputs(self) -> int: return 1 if self.quantiles is None else len(self.quantiles)
    @property
    def n_res_channels(self) -> int: return 1 + self.k_imfs  # 원 잔차 1 + VMD 성분 K


# =========================================================================== #
# 모델 구성 요소 (TFT 스타일)
# =========================================================================== #
class GLU(nn.Module):
    """Gated Linear Unit: sigmoid(a) * b"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return torch.sigmoid(a) * b


class GateAddNorm(nn.Module):
    """LayerNorm(GLU(Dropout(x)) + skip)"""
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, skip):
        return self.norm(self.glu(self.dropout(x)) + skip)


class GRN(nn.Module):
    """Gated Residual Network. d_context / c 는 인터페이스만 있고 사용하지 않음."""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1, d_context=None):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.gate_add_norm = GateAddNorm(d_hidden, d_out, dropout)

    def forward(self, a, c=None):
        return self.gate_add_norm(self.fc2(F.elu(self.fc1(a))), self.skip(a))


class VariableSelectionNetwork(nn.Module):
    """변수별 GRN 출력을 softmax 가중치로 합산. (합산값, 변수 가중치) 반환"""
    def __init__(self, var_names, d_model, dropout=0.1, d_context=None):
        super().__init__()
        self.var_names = var_names
        self.flat_grn = GRN(d_model * len(var_names), d_model, len(var_names), dropout, d_context)
        self.var_grns = nn.ModuleDict({v: GRN(d_model, d_model, d_model, dropout) for v in var_names})

    def forward(self, inputs, context=None):
        flat = torch.cat([inputs[v] for v in self.var_names], dim=-1)
        weights = torch.softmax(self.flat_grn(flat, context), dim=-1)
        processed = torch.stack([self.var_grns[v](inputs[v]) for v in self.var_names], dim=-1)
        return (processed * weights.unsqueeze(-2)).sum(dim=-1), weights


class InterpretableMultiHeadAttention(nn.Module):
    """헤드별 Q/K, 헤드 간 공유 V → 헤드 평균 (TFT interpretable attention)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, self.d_head)
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape
        qh = self.q(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        kh = self.k(k).view(B, k.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        vs = self.v(v).unsqueeze(1)
        scores = qh @ kh.transpose(-2, -1) / (self.d_head ** 0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ vs.expand(-1, self.n_heads, -1, -1)).mean(dim=1)
        return self.out(self.dropout(out)), attn.mean(dim=1)


class EndogenousPatchEncoder(nn.Module):
    """PV 입력 (B, lookback, C) → 패치 토큰 (B, n_patch, d) + global 토큰 1개"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_patch = (cfg.lookback - cfg.patch_len) // cfg.patch_stride + 1
        self.proj = nn.Linear(cfg.patch_len * cfg.n_endo_channels, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patch, cfg.d_model) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.grn = GRN(cfg.d_model, cfg.d_model, cfg.d_model, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B = x.shape[0]
        patches = x.unfold(1, self.cfg.patch_len, self.cfg.patch_stride).permute(0, 1, 3, 2).reshape(B, self.n_patch, -1)
        tokens = self.grn(self.dropout(self.proj(patches)) + self.pos)
        return torch.cat([tokens, self.global_token.expand(B, -1, -1)], dim=1)


class ExogenousVariateEncoder(nn.Module):
    """외생 변수별 1토큰 (B, n_vars, d). VSN 가중치로 토큰 크기를 조정한다."""
    def __init__(self, var_names, seq_len, cfg):
        super().__init__()
        self.var_names = var_names
        self.embed = nn.ModuleDict({v: nn.Linear(seq_len, cfg.d_model) for v in var_names})
        self.vsn = VariableSelectionNetwork(var_names, cfg.d_model, cfg.dropout)
        self.grn = GRN(cfg.d_model, cfg.d_model, cfg.d_model, cfg.dropout)

    def forward(self, x):
        tokens = {v: self.embed[v](x[..., i].contiguous()) for i, v in enumerate(self.var_names)}
        _, weights = self.vsn(tokens, None)
        stacked = torch.stack([tokens[v] for v in self.var_names], dim=1)
        return self.grn(stacked) * weights.unsqueeze(-1) * len(self.var_names), weights


class FusionLayer(nn.Module):
    """self-attention(내생 토큰) → global 토큰만 외생 토큰에 cross-attention → feed-forward"""
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = InterpretableMultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.gan_self = GateAddNorm(cfg.d_model, cfg.d_model, cfg.dropout)
        self.cross_attn = InterpretableMultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.gan_cross = GateAddNorm(cfg.d_model, cfg.d_model, cfg.dropout)
        self.ff = GRN(cfg.d_model, cfg.d_model * 2, cfg.d_model, cfg.dropout)
        self.gan_ff = GateAddNorm(cfg.d_model, cfg.d_model, cfg.dropout)

    def forward(self, endo, exog):
        h, _ = self.self_attn(endo, endo, endo)
        endo = self.gan_self(h, endo)
        g = endo[:, -1:, :]
        h_cross, _ = self.cross_attn(g, exog, exog)
        endo = torch.cat([endo[:, :-1, :], self.gan_cross(h_cross, g)], dim=1)
        return self.gan_ff(self.ff(endo), endo), {}


class ForecastHead(nn.Module):
    """전체 토큰 flatten → MLP → (B, horizon)"""
    def __init__(self, n_tokens, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Sequential(
            nn.Linear(n_tokens * cfg.d_model, cfg.d_model * 2), nn.ELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.horizon * cfg.n_outputs),
        )

    def forward(self, tokens):
        return self.fc(tokens.reshape(tokens.shape[0], -1)).view(tokens.shape[0], self.cfg.horizon)


class BaseModel(nn.Module):
    """Base 예측 모델: PV 패치 토큰 ↔ (기상 + 시간) 토큰 융합 → ŷ_base"""
    def __init__(self, cfg):
        super().__init__()
        self.endo_encoder = EndogenousPatchEncoder(cfg)
        self.weather_encoder = ExogenousVariateEncoder(cfg.weather_vars, cfg.weather_len, cfg)
        self.temporal_encoder = ExogenousVariateEncoder(cfg.temporal_vars, cfg.temporal_len, cfg)
        self.layers = nn.ModuleList([FusionLayer(cfg) for _ in range(cfg.n_fusion_layers)])
        self.head = ForecastHead(self.endo_encoder.n_patch + 1, cfg)

    def forward(self, batch):
        w_tok, _ = self.weather_encoder(batch['weather'])
        t_tok, _ = self.temporal_encoder(batch['temporal'])
        exog = torch.cat([w_tok, t_tok], dim=1)
        endo = self.endo_encoder(batch['pv'])
        for layer in self.layers: endo, _ = layer(endo, exog)
        return self.head(endo)


class ResidualSequenceEncoder(nn.Module):
    """잔차 윈도우 (B, R, 1+K) → 패치 토큰 + global 토큰"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_patch = (cfg.residual_lookback - cfg.patch_len) // cfg.patch_stride + 1
        self.proj = nn.Linear(cfg.patch_len * cfg.n_res_channels, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patch, cfg.d_model) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.grn = GRN(cfg.d_model, cfg.d_model, cfg.d_model, cfg.dropout)

    def forward(self, e_seq):
        B = e_seq.shape[0]
        patches = e_seq.unfold(1, self.cfg.patch_len, self.cfg.patch_stride).permute(0, 1, 3, 2).reshape(B, self.n_patch, -1)
        tokens = self.grn(self.proj(patches) + self.pos)
        return torch.cat([tokens, self.global_token.expand(B, -1, -1)], dim=1)


class BasePredictionEncoder(nn.Module):
    """Base 예측값 ŷ_base → 토큰 1개"""
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.horizon, cfg.d_model)
        self.grn = GRN(cfg.d_model, cfg.d_model, cfg.d_model, cfg.dropout)

    def forward(self, y_hat_b):
        return self.grn(self.proj(y_hat_b)).unsqueeze(1)


class ResidualModel(nn.Module):
    """
    잔차 예측 모델 ê.
      - 내생: 잔차 윈도우 토큰 (원 잔차 + VMD 성분)
      - 외생: 기상 / 시간 / PV 토큰 (stage1 BaseModel 인코더 공유, frozen) + ŷ_base 토큰
    """
    def __init__(self, cfg, stage1):
        super().__init__()
        self.res_encoder = ResidualSequenceEncoder(cfg)
        self.base_pred_encoder = BasePredictionEncoder(cfg)
        self.weather_encoder = stage1.weather_encoder
        self.temporal_encoder = stage1.temporal_encoder
        self.endo_encoder = stage1.endo_encoder
        for m in [self.weather_encoder, self.temporal_encoder, self.endo_encoder]:
            for param in m.parameters(): param.requires_grad = False
        self.layers = nn.ModuleList([FusionLayer(cfg) for _ in range(cfg.n_fusion_layers)])
        self.head = ForecastHead(self.res_encoder.n_patch + 1, cfg)

    def forward(self, batch, y_hat_b):
        endo = self.res_encoder(batch['residual_hist'])
        base_pred_tok = self.base_pred_encoder(y_hat_b.detach())
        w_tok, _ = self.weather_encoder(batch['weather'])
        t_tok, _ = self.temporal_encoder(batch['temporal'])
        pv_tok = self.endo_encoder(batch['pv'])
        exog = torch.cat([w_tok, t_tok, pv_tok, base_pred_tok], dim=1)
        for layer in self.layers: endo, _ = layer(endo, exog)
        return self.head(endo)


def load_frozen_base(path, cfg):
    """저장된 BaseModel 가중치 로드 → eval + requires_grad=False"""
    model = BaseModel(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    return model


# =========================================================================== #
# VMD (ESVD) 피처
# =========================================================================== #
def extract_vmd_features(signal_1d: np.ndarray, K: int = 4) -> np.ndarray:
    """1D 신호 → (len, K) VMD 모드. vmdpy 미설치 / 신호가 거의 0 / 분해 실패 시 0 행렬."""
    if not HAS_VMDPY or np.max(np.abs(signal_1d)) < 1e-5: return np.zeros((len(signal_1d), K))
    try:
        u, _, _ = VMD(signal_1d, 2000, 0, K, 0, 1, 1e-7)  # alpha=2000, tau=0, DC=0, init=1, tol=1e-7
        return u.T
    except Exception:
        return np.zeros((len(signal_1d), K))


# =========================================================================== #
# Dataset
# =========================================================================== #
def fit_power_scaler(df, idx_range):
    """df 의 idx_range 행 Power (MW) 중 주간값(> 0.0001)으로 StandardScaler fit (주간값이 없으면 전체 사용)"""
    scaler = StandardScaler()
    train_power = df.loc[idx_range, 'Power (MW)'].values
    train_power_day = train_power[train_power > 0.0001].reshape(-1, 1)
    if len(train_power_day) == 0: train_power_day = train_power.reshape(-1, 1)
    scaler.fit(train_power_day)
    return scaler


class PVDataset(Dataset):
    """
    Base 모델용 Dataset. scaler 는 외부에서 fit 해서 주입.
      - npy_Y, npy_X[:, :, 0](Power) : scaler.transform
      - npy_X[:, :, 1:] (ESVD 성분)   : scaler.scale_ 로만 나눔
    item: pv / weather / temporal / target / idx
    """
    def __init__(self, df, npy_X, npy_Y, npy_Y_time, cfg, scaler):
        self.cfg, self.df, self.scaler = cfg, df.copy(), scaler

        npy_Y_scaled = self.scaler.transform(npy_Y.reshape(-1, 1)).flatten()
        npy_X_scaled = npy_X.copy()
        B, L, C = npy_X.shape
        npy_X_scaled[:, :, 0] = self.scaler.transform(npy_X[:, :, 0].reshape(-1, 1)).reshape(B, L)
        for c in range(1, C): npy_X_scaled[:, :, c] /= self.scaler.scale_[0]

        self.npy_X  = torch.tensor(npy_X_scaled, dtype=torch.float32)
        self.npy_Y  = torch.tensor(npy_Y_scaled, dtype=torch.float32)
        self.Y_time = npy_Y_time
        self.weather_data  = torch.tensor(self.df[cfg.weather_vars].values, dtype=torch.float32)
        self.temporal_data = torch.tensor(self.df[cfg.temporal_vars].values, dtype=torch.float32)

    def __len__(self): return len(self.npy_X)

    def __getitem__(self, idx):
        t_curr, t_end = idx + self.cfg.lookback, idx + self.cfg.lookback + self.cfg.horizon
        return {
            'pv':       self.npy_X[idx],
            'weather':  self.weather_data[t_curr - 1:t_curr],   # 현재 시점 기상
            'temporal': self.temporal_data[t_curr:t_end],       # 예측 시점 시간 피처
            'target':   self.npy_Y[idx].view(-1),
            'idx':      torch.tensor([idx], dtype=torch.long),
        }

    def collate(self, indices):
        """단일/소수 인덱스를 배치 dict 로 묶음 (sequential test 용)"""
        items = [self[i.item() if torch.is_tensor(i) else i] for i in indices]
        return {k: torch.stack([item[k] for item in items]) for k in items[0].keys()}


class ResidualDataset(PVDataset):
    """
    잔차 모델용 Dataset. PVDataset item 에 base_N_oof 의 OOF 값을 추가.
      - oof_pred      : OOF Base 예측 (scaled)
      - oof_res       : OOF 잔차 = target − oof_pred (scaled)
      - residual_hist : [oof_res[idx−R:idx], esvd_windows[idx]] → (R, 1+K). idx < R 이면 0
    """
    def __init__(self, df, npy_X, npy_Y, npy_Y_time, saved_oof_preds, saved_residuals, saved_esvd_windows, cfg, scaler):
        super().__init__(df, npy_X, npy_Y, npy_Y_time, cfg, scaler)
        self.saved_oof_preds    = saved_oof_preds.cpu()
        self.saved_residuals    = saved_residuals.cpu()
        self.saved_esvd_windows = saved_esvd_windows.cpu()
        self.R, self.C = cfg.residual_lookback, cfg.n_res_channels

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        batch['oof_pred'] = self.saved_oof_preds[idx].view(-1)
        batch['oof_res']  = self.saved_residuals[idx].view(-1)
        if idx >= self.R:
            batch['residual_hist'] = torch.cat([self.saved_residuals[idx - self.R:idx], self.saved_esvd_windows[idx]], dim=-1)
        else:
            batch['residual_hist'] = torch.zeros(self.R, self.C)
        return batch


def decode_time(t):
    """npy_Y_time 원소(bytes 문자열 가능) → 'YYYY-MM-DD HH:MM:SS' 문자열"""
    return str(t).replace("b'", "").replace("'", "")


def to_device(batch):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def point_forecast(y):
    """분위수 출력 (B, H, Q) 이면 중앙 분위수, 아니면 그대로 (현재 설정은 항상 (B, H))"""
    return y[..., y.shape[-1] // 2] if y.dim() == 3 else y


# =========================================================================== #
# 학습
# =========================================================================== #
def train_base_model(model, dataset, train_idx, val_idx, epochs=100, patience=15, lr=1e-3,
                     desc='Base', persistent_workers=False):
    """
    BaseModel 학습 (Adam, MSE) + early stopping.
      - 검증 손실: target > 0.0001 (주간) 샘플만
      - best epoch 가중치를 메모리에 보관했다가 종료 시 복원
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, counter, best_state, best_epoch = float('inf'), 0, None, 0
    loader_kw = dict(batch_size=512, num_workers=4, pin_memory=True, persistent_workers=persistent_workers)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,  **loader_kw)
    val_loader   = DataLoader(Subset(dataset, val_idx),   shuffle=False, **loader_kw)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad()
            y_hat = point_forecast(model(batch))
            F.mse_loss(y_hat, batch['target']).backward(); optimizer.step()

        model.eval(); val_loss, val_b = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch)
                y_hat = point_forecast(model(batch))
                mask = batch['target'] > 0.0001
                if mask.sum() > 0: val_loss += F.mse_loss(y_hat[mask], batch['target'][mask]).item(); val_b += 1
        val_loss /= max(1, val_b)
        _print_epoch(desc, epoch, val_loss, counter, patience)

        if val_loss < best_loss:
            best_loss, counter, best_epoch = val_loss, 0, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience: break
    print()
    _print_train_summary(desc, best_loss, best_epoch, epoch + 1)
    model.load_state_dict(best_state)
    return model


def train_residual_model(frozen_base, dataset, train_idx, cfg, save_path,
                         epochs=100, patience=15, desc='Residual', **loader_kwargs):
    """
    ResidualModel 학습 (Adam lr=1e-3, MSE) + early stopping.
      - 입력 ŷ_base / 정답 잔차: 저장된 OOF 값 (oof_pred / oof_res) → 누수 방지
      - train_idx 앞 90% 학습 / 뒤 10% 검증 (검증 손실은 target > 0.0001 샘플만)
      - best 가중치는 save_path 에 저장 후 종료 시 다시 로드
      - loader_kwargs: DataLoader 추가 옵션 (num_workers 등)
    """
    split = int(len(train_idx) * 0.9)
    tr, val = train_idx[:split], train_idx[split:]
    loader_tr  = DataLoader(Subset(dataset, tr),  batch_size=512, shuffle=True,  **loader_kwargs)
    loader_val = DataLoader(Subset(dataset, val), batch_size=512, shuffle=False, **loader_kwargs)

    frozen_base.eval()
    res_model = ResidualModel(cfg, frozen_base).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, res_model.parameters()), lr=1e-3)
    best_loss, counter, best_epoch = float('inf'), 0, 0

    for epoch in range(epochs):
        res_model.train()
        for batch in loader_tr:
            batch = to_device(batch)
            opt.zero_grad()
            e_hat = point_forecast(res_model(batch, batch['oof_pred']))
            F.mse_loss(e_hat, batch['oof_res']).backward(); opt.step()

        res_model.eval(); val_loss, val_b = 0.0, 0
        with torch.no_grad():
            for batch in loader_val:
                batch = to_device(batch)
                e_hat = point_forecast(res_model(batch, batch['oof_pred']))
                mask = batch['target'] > 0.0001
                if mask.sum() > 0: val_loss += F.mse_loss(e_hat[mask], batch['oof_res'][mask]).item(); val_b += 1
        val_loss /= max(1, val_b)
        _print_epoch(desc, epoch, val_loss, counter, patience)

        if val_loss < best_loss:
            best_loss, counter, best_epoch = val_loss, 0, epoch + 1
            torch.save(res_model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience: break
    print()
    _print_train_summary(desc, best_loss, best_epoch, epoch + 1)
    if os.path.exists(save_path): res_model.load_state_dict(torch.load(save_path))
    return res_model


# =========================================================================== #
# Sequential Test 유틸 (create_rf / crc_create_N_results)
# =========================================================================== #
def push_residual(window, err):
    """잔차 윈도우 (R, 1) 를 한 칸 밀고 마지막에 err 추가"""
    window = torch.roll(window, shifts=-1, dims=0)
    window[-1, 0] = err.squeeze()
    return window


@torch.no_grad()
def warmup_residual_window(rolling_base_model, dataset, test_start, R):
    """test 직전 R 스텝을 rolling_base 로 재추론해 잔차 윈도우 (R, 1) 초기화"""
    current_res_raw = torch.zeros(R, 1, device=device)
    for idx in range(test_start - R, test_start):
        batch = to_device(dataset.collate([idx]))
        y_hat_b = point_forecast(rolling_base_model(batch))
        current_res_raw = push_residual(current_res_raw, batch['target'] - y_hat_b)
    return current_res_raw


@torch.no_grad()
def predict_residual(residual_model, batch, current_res_raw, y_hat_b, cfg):
    """현재 잔차 윈도우로 실시간 VMD → residual_hist 구성 → ê 추론"""
    curr_esvd_np = extract_vmd_features(current_res_raw.cpu().numpy().flatten(), K=cfg.k_imfs)
    curr_esvd = torch.tensor(curr_esvd_np, dtype=torch.float32, device=device)
    batch['residual_hist'] = torch.cat([current_res_raw, curr_esvd], dim=-1).unsqueeze(0)
    return point_forecast(residual_model(batch, y_hat_b))


# =========================================================================== #
# 채점
# =========================================================================== #
def filter_eval_window(df, site_idx):
    """채점 구간: Site 7 이상 구간 제외 + 사이트별 가동 시간만. Parsed_Time / Time_HHMM 컬럼 추가된 사본 반환."""
    df = df.copy()
    df['Parsed_Time'] = pd.to_datetime(df['Time'])
    if site_idx == 7:
        exc_start, exc_end = pd.to_datetime(SITE7_EXCLUDE[0]), pd.to_datetime(SITE7_EXCLUDE[1])
        df = df[~((df['Parsed_Time'] >= exc_start) & (df['Parsed_Time'] <= exc_end))].copy()
    st, et = OPERATION_HOURS[site_idx]
    df['Time_HHMM'] = df['Parsed_Time'].dt.strftime('%H:%M')
    return df[(df['Time_HHMM'] >= st) & (df['Time_HHMM'] <= et)].copy()


def score(y_true, y_pred, capacity):
    """(NRMSE %, NMAE %, R²) — 분모 capacity"""
    nrmse = (np.sqrt(mean_squared_error(y_true, y_pred)) / capacity) * 100
    nmae  = (mean_absolute_error(y_true, y_pred) / capacity) * 100
    r2    = r2_score(y_true, y_pred)
    return nrmse, nmae, r2
