"""
plot_config.py
==============
Shared visual settings for analysis notebooks and scripts.
Updated for A4 paper column widths and custom font sizes.

References
----------
- Nature figure guide (fonts, sizes, formats, accessibility)
- Nature graphs guide (axes, colours, text, export)
- Wong, B. Points of view: Colour blindness. Nature Methods 8, 441 (2011).
"""

import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════════════════
# Paths / seeds
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT = "/Users/ohdonggeun/workspace/R-ECO/dataset"
SAVE_ROOT = "/Users/ohdonggeun/workspace/R-ECO/figures/fig"

SAVE_DIR_MAIN = os.path.join(SAVE_ROOT)
os.makedirs(SAVE_DIR_MAIN, exist_ok=True)

SAVE_DIR = os.path.join(SAVE_ROOT)
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [40, 41, 42, 43, 44]
sites = [1,2,4,5,6,7,8]
# sites = [4,6]

# ══════════════════════════════════════════════════════════════════════════════
# 1. Discrete colour list  — colorblind-safe (Wong 2011 + Nature house style)
#    Order maximises pairwise contrast for up to 8 series.
#    Avoids red/green adjacency (protanopia / deuteranopia confusion).
# ══════════════════════════════════════════════════════════════════════════════
clist: list[str] = [
    "#0073C2",  # deep blue      – series 1 / primary
    "#4DBBD5",  # sky blue       – series 3
    "#CD534C",  # muted red      – series 2
    "#EFC000",  # gold / amber   – series 4
    "#7E6148",  # brown          – series 5
    "#20854E",  # green          – series 6
    "#EE4C97",  # pink/magenta   – series 7
    "#747678",  # neutral grey   – series 8
]

# ══════════════════════════════════════════════════════════════════════════════
# 2. Continuous / gradient colour maps
#    Access via  get_cmap(name)  or  CMAPS[name]  directly.
# ══════════════════════════════════════════════════════════════════════════════
CMAPS: dict[str, mcolors.Colormap] = {}

# ── custom maps ───────────────────────────────────────────────────────────────
_custom: dict[str, list[str]] = {
    "nature_blue": ["#FFFFFF", "#0073C2"],
    "nature_red": ["#FFFFFF", "#CD534C"],
    "nature_green": ["#FFFFFF", "#20854E"],
    "nature_gold": ["#FFFFFF", "#EFC000"],
    "gradient_br": ["#0073C2", "#CD534C"],
    "blue_gold": ["#0073C2", "#EFC000"],
    "clist_gradient": clist,
    # 중요도 랭크용 발산형: 1등(가장 중요) = 빨강 → 꼴찌 = 파랑.
    # clist 의 muted red(#CD534C) / deep blue(#0073C2) 를 축으로,
    # 양 끝을 어둡게 확장해 랭크 폭이 넓어도 대비가 유지되도록 함.
    "rank_rb": ["#67201C", "#CD534C", "#F2F2F2", "#0073C2", "#03365C"],
}
for _name, _colors in _custom.items():
    CMAPS[_name] = LinearSegmentedColormap.from_list(_name, _colors, N=256)

# legacy alias kept for backward-compatibility
cmap = CMAPS["gradient_br"]

# ── matplotlib built-ins ──────────────────────────────────────────────────────
for _name in [
    "viridis", "plasma", "inferno", "magma", "cividis",  # perceptually uniform
    "RdBu_r", "coolwarm", "PiYG", "PRGn",  # diverging
    "Blues", "Reds", "YlOrRd", "GnBu",  # single-hue sequential
]:
    CMAPS[_name] = plt.get_cmap(_name)


def get_cmap(name: str = "viridis") -> mcolors.Colormap:
    """Return a colormap by name (custom or matplotlib built-in).

    주의: CMAPS.get(name, plt.get_cmap(name)) 로 쓰면 커스텀 맵이 등록돼 있어도
    기본값 인자가 먼저 평가돼 matplotlib 에서 에러가 납니다. 반드시 분기할 것.
    """
    if name in CMAPS:
        return CMAPS[name]
    return plt.get_cmap(name)


def get_colors(n: int, cmap_name: str = "clist") -> list[str]:
    """
    Return *n* colours as hex strings.
      cmap_name="clist"  → cycle through the discrete clist (default)
      anything else      → sample evenly from the named continuous cmap
    """
    if cmap_name == "clist":
        return [clist[i % len(clist)] for i in range(n)]
    cm = get_cmap(cmap_name)
    return [mcolors.to_hex(cm(i / max(n - 1, 1))) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# 3. rcParams  — A4 Custom Format Compliance
# ══════════════════════════════════════════════════════════════════════════════

# ── 글자 크기 ─────────────────────────────────────────────────────────────────
# 그림은 FULL_W(전체 폭) 또는 HALF_W(나란히 두 장) 로 저장하고, 원고에서는 각각
# width=\linewidth / 0.5\linewidth 로 넣습니다. 어느 쪽이든 배율 1.0 이라 여기
# 적은 pt 가 그대로 인쇄됩니다. 그림별 글자 보정 로직은 두지 않습니다.
PT_LABEL = 9.0     # 축 라벨 / 패널 라벨
PT_TICK = 7.0      # 눈금 숫자
PT_LEGEND = 8.0    # 범례
PT_ANNOT = 7.0     # 그림 안 주석(별표·p값 등) 기본값

# 스크립트가 10 pt 기준으로 적어 둔 값을 위 체계로 환산하는 비율
_PT_SCALE = PT_LABEL / 10.0


def fs(pt: float) -> float:
    """10 pt 기준으로 적힌 글자 크기를 현재 체계(PT_LABEL 기준)로 환산.

    ⚠️ 새 코드에서 쓰지 마세요. 배율이 0.9 라 fs(9)=8.1, fs(8)=7.2 가 되어
    위의 PT_LABEL(9) / PT_LEGEND(8) 과 **미묘하게 어긋납니다**. 실제로 5_e·5_f
    가 fs(9)/fs(8) 을 쓰는 바람에 같은 계열인 5_d(rcParams 기본 9/8)보다 축
    라벨·범례가 10% 작게 나왔습니다.

    축 라벨 → PT_LABEL, 눈금 → PT_TICK, 범례 → PT_LEGEND, 그림 안 주석 →
    PT_ANNOT 를 직접 쓰세요. fs() 는 옛 스크립트 호환용으로만 남깁니다.
    """
    return pt * _PT_SCALE


def figure_config() -> None:
    """Apply custom rcParams globally. Safe to call multiple times."""
    mpl.rcParams.update({
        # ── Typography ────────────────────────────────────────────────────────
        "font.family": "Helvetica",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",

        # Updated text-size hierarchy
        # Body text / Labels / Legends: 8~9 pt
        # Ticks: 6~7 pt
        # Elsevier 기준(90/140/190 mm 실크기 저장)에서 읽히도록 한 단계 키웠습니다.
        # 8/7/9 → 10/9/11. bbox="tight" 를 끄면서 확대 배율이 사라졌기 때문에
        # 여기 적은 pt 가 곧 인쇄되는 pt 입니다.
        "font.size": PT_LABEL,       # base / fallback
        "axes.labelsize": PT_LABEL,  # axis labels (must include units in parens)
        "axes.titlesize": PT_LABEL,  # subplot titles
        "xtick.labelsize": PT_TICK,  # tick numbers (x)
        "ytick.labelsize": PT_TICK,  # tick numbers (y)
        "legend.fontsize": PT_LEGEND, # legend text
        "figure.titlesize": PT_LABEL, # overall figure title

        # ── Axes / ticks ──────────────────────────────────────────────────────
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # ── Grid / clutter ────────────────────────────────────────────────────
        "axes.grid": False,

        # ── Legend ────────────────────────────────────────────────────────────
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,

        # ── Colours / backgrounds ─────────────────────────────────────────────
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        # ── Editable vector text ──────────────────────────────────────────────
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # ── Export defaults ───────────────────────────────────────────────────
        "savefig.dpi": 300,
        # bbox="tight" 는 저장 시 여백을 잘라 **실제 저장 폭 ≠ figsize** 가 됩니다.
        # 그러면 LaTeX 에서 목표 폭에 맞출 때 그림마다 확대 배율이 달라지고
        # 지정한 pt 와 무관하게 글자 크기가 제각각이 됩니다(7 은 8pt→10.9pt 였음).
        # None 으로 두어 저장 폭 = figsize 를 보장하고, 여백은 tight_layout 이 잡습니다.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.01,
    })


# Apply immediately on import
figure_config()

# ══════════════════════════════════════════════════════════════════════════════
# 원고 폭 기준 — 여기 하나만 맞으면 모든 그림의 인쇄 글자 크기가 같아집니다
# ══════════════════════════════════════════════════════════════════════════════
# 이 논문은 **1단(single column) elsarticle preprint** 이고 본문 폭은
#   \linewidth = 390 pt = 5.42 in = 137.6 mm
# 입니다(컴파일된 PDF 를 100% 로 재도 5.44 in 로 일치).
#
# 인쇄되는 글자 크기 = 지정 pt × (삽입 폭 / 저장 폭) 입니다. 따라서
# **저장 폭을 원고 폭과 같게** 두는 것이 전부입니다. 배율 1.0 이면 아래
# PT_LABEL/PT_TICK/PT_LEGEND 가 그대로 인쇄되고, 그림마다 크기가 어긋날 수 없습니다.
#
# ⚠️ 과거 사고: 표준 폭을 Elsevier 최종 2단 규격(183 mm = 7.2 in)으로 잡아 두고
#    원고에는 width=\linewidth 로 넣었습니다. 배율이 5.42/7.20 = 0.75 로 걸려
#    7 pt tick 이 지면에서 5.3 pt 로 인쇄됐고, 그림이 클수록 더 꾸겨져 보였습니다.
#    반대로 옛 빌드가 남아 있던 5_e(4.4 in)는 1.23 배로 확대되어 8.6 pt 였고,
#    그래서 5_d 와 5_e 의 글자가 1.6 배 차이 났습니다.
MS_W = 390 / 72          # 5.42 in — 원고 \linewidth (390 pt). 모든 폭의 기준.

# ── 전체 폭 그림 ──────────────────────────────────────────────────────────────
# 원고에 \includegraphics[width=\linewidth] 로 넣습니다 → 배율 1.0.
FULL_W = MS_W            # 5.42 in

# ── 가로로 두 개 나란히 넣는 그림 (3_a|3_b, 6_a|6_b, 8_c) ─────────────────────
# 원고에 width=0.5\linewidth 로 넣습니다 → 배율 1.0. 전체 폭 그림과 인쇄 pt 동일.
HALF_W = MS_W / 2        # 2.71 in

# ── 기존 고정 높이를 옮길 때 쓰는 비율 ────────────────────────────────────────
# 폭이 7.2 → 5.42 in 로 좁아졌으니, 인치로 박아 둔 **데이터 영역** 높이(패널 한
# 칸, 히트맵 셀 한 칸)에 이 비율을 곱해야 지면에서 차지하는 자리가 그대로입니다.
# 글자가 차지하는 여백(축 라벨·회전 눈금·컬러바 줄)은 절대 크기라 곱하지 않습니다.
ELS_FULL_W = 183 / 25.4  # 7.20 in — Elsevier 최종 2단 조판 규격(참고용)
H_SCALE = MS_W / ELS_FULL_W   # 0.752

SINGLE_COL = 90 / 25.4   # 3.54 in — 인쇄 1단
ONEHALF_COL = 140 / 25.4  # 5.51 in — 1.5단 (원고 폭과 거의 일치)
DOUBLE_COL = 190 / 25.4  # 7.48 in — 인쇄 2단 (최대 높이 240 mm)
MAX_FIG_H = 240 / 25.4   # 9.45 in


# ══════════════════════════════════════════════════════════════════════════════
# 4. Panel-label and axes utilities
# ══════════════════════════════════════════════════════════════════════════════

def label_panels(
        axes,
        labels: "str | list[str] | None" = None,
        fontsize: "float | None" = None,
        uppercase: bool = False,
) -> None:
    axes = list(axes)

    # [수정된 부분] labels가 없으면 'Site 1', 'Site 2'... 형식으로 생성
    if labels is None:
        labels = [f"Site {sites[i]}" for i in range(len(axes))]
    elif isinstance(labels, str):
        labels = list(labels)

    if fontsize is None:
        fontsize = PT_LABEL
    for ax, lbl in zip(axes, labels):
        # Site 번호를 그래프 내부가 아닌, 기존 패널 라벨처럼
        # 왼쪽 상단 밖(Nature 스타일)에 배치하고 싶다면 add_panel_label을 그대로 사용
        add_panel_label(ax, lbl, fontsize=fontsize, uppercase=uppercase)


def add_panel_label(ax, label: str, fontsize: "float | None" = None,
                    uppercase: bool = False) -> None:
    """패널 라벨(A/B/C, Site 1 …)을 축 박스 왼쪽 위에 붙입니다.

    예전에는 x=-0.15 (축 왼쪽 바깥)이라 savefig 의 tight 크롭에 기대고 있었습니다.
    크롭을 끄면(= 규격 폭 저장) 캔버스 밖으로 나가 잘리므로 축 왼쪽 끝(x=0)에
    맞춰 왼쪽 정렬합니다.
    """
    if uppercase:
        label = label.upper()
    if fontsize is None:
        fontsize = PT_LABEL
    ax.text(
        0.0, 1.02,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily="Helvetica",
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def legend_below_axes(fig, axes, handles=None, labels=None, y: float = 0.0,
                      ncol: int = 3, **kw):
    """서브플롯(축) 영역의 가로 중앙 아래에 공통 범례를 배치.

    fig.legend(loc="lower center") 는 *figure* 기준이라 y라벨 폭만큼 왼쪽으로
    치우쳐 보입니다. 이 함수는 실제 축 박스의 중앙에 맞춥니다.
    레이아웃(tight_layout 등)을 적용한 *뒤에* 호출하세요.
    """
    axes = list(axes) if hasattr(axes, "__iter__") else [axes]
    boxes = [ax.get_position() for ax in axes]
    x_center = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2.0
    if handles is None or labels is None:
        handles, labels = axes[0].get_legend_handles_labels()
    return fig.legend(handles, labels, loc="lower center",
                      bbox_to_anchor=(x_center, y), ncol=ncol, frameon=False, **kw)


def site_panel_axes(n: int, panel_h_ratio: float = 0.55, max_fig_h: float = 7.5,
                    width: "float | None" = None):
    """사이트 n개를 1열 세로 스택으로 배치하고 (fig, axes, bottom_axes) 를 돌려줍니다.

    패널 하나의 기본 높이는 SINGLE_COL * panel_h_ratio 인데, 사이트가 많아
    전체 높이가 max_fig_h(A4 한 페이지에 들어가는 한계)를 넘으면 패널 높이를
    균등하게 줄여 맞춥니다. 사이트 7개면 패널당 약 1.07 in 이 됩니다.

    bottom_axes 는 x 눈금/라벨을 붙일 축(맨 아래 하나) 목록입니다.
    """
    # 폭은 원고 표준(FULL_W = 원고 \linewidth). 패널 높이는 내용 기준으로 잡되,
    # 폭이 7.2 → 5.42 in 로 좁아진 만큼 H_SCALE 을 곱해 지면에서 차지하는 자리를
    # 그대로 유지합니다(글자는 배율 1.0 이 되어 제 크기로 인쇄됩니다).
    if width is None:
        width = FULL_W
    panel_h = min(SINGLE_COL * panel_h_ratio, max_fig_h / n) * H_SCALE
    fig, axes = plt.subplots(n, 1, figsize=(width, panel_h * n), sharex=True)
    axes = [axes] if n == 1 else list(axes)
    return fig, axes, [axes[-1]]


def thin_spines(ax, lw: "float | None" = None) -> None:
    """Set all four spine linewidths."""
    if lw is None:
        lw = 0.8
    for sp in ax.spines.values():
        sp.set_linewidth(lw)


def set_nature_ticks(ax, which: str = "both") -> None:
    """
    Apply outward tick marks.
    """
    kw = dict(direction="out", width=0.8, length=3.0, pad=2.0)
    if which in ("both", "x"):
        ax.xaxis.set_tick_params(**kw)
    if which in ("both", "y"):
        ax.yaxis.set_tick_params(**kw)
    thin_spines(ax)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Dataset layout / site-level settings  (ExoTFT-CRC, updated dataset)
# ══════════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass, field

RESULTS_DIR = os.path.join(DATA_ROOT, "results")
ORIGIN_DIR = os.path.join(DATA_ROOT, "origin")
SHAP_DIR = os.path.join(DATA_ROOT, "shap")

ORIGIN_TIME_COL = "Time(year-month-day h:m:s)"
POWER_COL = "Power (MW)"

# 가조시간 필터 전역 스위치.
#   False (기본) → 그림은 24시간 전 구간을 사용
#   True         → 아래 OPERATION_HOURS 로 잘라서 사용
# loader.load_site(apply_operation_hours=...) 로 그림마다 개별 지정도 가능합니다.
# (loader.metric_table 은 채점용이라 이 스위치와 무관하게 기본 True 입니다.)
APPLY_OPERATION_HOURS = False

# 채점(평가) 대상 가조시간. threshold(> 0.0001) 필터는 사용하지 않음.
OPERATION_HOURS = {
    1: ("06:00", "21:30"),
    2: ("00:00", "23:59"),
    4: ("00:00", "23:59"),
    5: ("00:00", "23:59"),
    6: ("06:00", "21:00"),
    7: ("06:00", "21:00"),
    8: ("06:00", "19:00"),
}

# Site 7 데이터 이상 구간 (반드시 제외)
SITE7_EXCLUDE = ("2020-12-14 08:00:00", "2020-12-31 23:45:00")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Model registry  ── 비교 대상 5개 모델
#
#    새 모델 결과가 준비되면 아래 MODELS 에 ModelSpec 한 줄만 추가/수정하면
#    loader 와 모든 figure 스크립트가 자동으로 반영합니다.
#
#      file      : dataset/results/ 아래 파일명 패턴 ('{site}' 치환)
#      pred_col  : 그 파일 안의 예측값 컬럼명
#      extra_cols: 함께 실어올 부가 컬럼 (없으면 [])
#      color     : clist 인덱스
# ══════════════════════════════════════════════════════════════════════════════
# ── 논문 공통 모델 표기 ───────────────────────────────────────────────────────
#   ExoTFT-FB       22개월 full-batch 학습 base
#   ExoTFT-RF       12개월 학습 + 10개월 rolling fine-tuning base
#   ExoTFT-CRC      RF + calibrated residual correction (최종 제안 모델)
#
#   figure 텍스트와 콘솔/CSV 출력 모두 하이픈 표기(ExoTFT-FB / -RF / -CRC)로 통일했습니다.
LABEL_FB  = "ExoTFT-FB"
LABEL_RF  = "ExoTFT-RF"
LABEL_CRC = "ExoTFT-CRC"

# 모델명은 더 이상 mathtext(아래첨자)가 아니라 하이픈 표기 일반 텍스트입니다.
# 수식과 섞어 쓸 때는 $..$ 안에 넣지 말고 문자열로 이어 붙이세요.
#   f"{config.LABEL_CRC} \u2212 {config.LABEL_FB}"
MATH_FB, MATH_RF, MATH_CRC = LABEL_FB, LABEL_RF, LABEL_CRC


@dataclass
class ModelSpec:
    key: str                      # 코드에서 쓰는 짧은 이름 (컬럼 접두사)
    label: str                    # 그래프 범례에 찍히는 이름 (mathtext 가능)
    short: str = ""               # 콘솔/표 출력용 ASCII 이름 (없으면 label 사용)
    pred_col: str = ""            # 결과 CSV 안의 예측값 컬럼명
    file: "str | None" = None     # 시계열 결과 CSV 파일명 패턴 (None = 집계값만 있는 모델)
    agg_name: "str | None" = None # model_total_results.csv 의 model 컬럼 값
    agg_scale: float = 1.0        # 집계표의 nmae/nrmse 를 [%] 로 맞추는 배수
    color: str = clist[0]         # 선/막대 색
    extra_cols: list = field(default_factory=list)
    linestyle: str = "-"
    zorder: int = 2

    def path(self, site: int) -> "str | None":
        if self.file is None:
            return None
        return os.path.join(RESULTS_DIR, self.file.format(site=site))

    def exists(self, site: int) -> bool:
        p = self.path(site)
        return p is not None and os.path.exists(p)

    # 표기 보조: math 는 $ 를 벗긴 mathtext 본문(다른 수식과 합칠 때),
    #            plain 은 콘솔/CSV 용 ASCII 이름.
    @property
    def math(self) -> str:  return self.label.strip("$")  # 하이픈 표기라 label 과 동일
    @property
    def plain(self) -> str: return self.short or self.label

    # 병합 후 생성되는 파생 컬럼 이름
    @property
    def pred(self) -> str:      return f"pred_{self.key}"
    @property
    def resid(self) -> str:     return f"resid_{self.key}"
    @property
    def abs_resid(self) -> str: return f"abs_resid_{self.key}"


# 사이트 × 모델 집계 성능표 (fig4)
TOTAL_RESULTS_FILE = os.path.join(RESULTS_DIR, "model_total_results.csv")

# 단위 보정용 예외 (현재 CSV 는 nmae/nrmse 가 전부 [%] 로 통일되어 있어 비어 있음).
# 특정 (site, model) 칸만 단위가 다를 때 { ("Site 1", "chen"): 100.0 } 식으로 넣습니다.
AGG_CELL_OVERRIDES = {}


MODELS: "dict[str, ModelSpec]" = {
    # ── 비교 모델 (집계 성능표에만 존재) ──────────────────────────────────────
    "dynamic": ModelSpec(
        key="dynamic",
        label="Dynamic",
        agg_name="dynamic",
        agg_scale=1.0,            # 이미 [%]
        color=clist[7],           # neutral grey
        linestyle=(0, (3, 1, 1, 1)),
    ),
    "chen": ModelSpec(
        key="chen",
        label="Chen et al.",
        agg_name="chen",
        agg_scale=1.0,
        color=clist[4],           # brown
        linestyle="-.",
    ),
    "kong": ModelSpec(
        key="kong",
        label="Kong et al.",
        agg_name="kong",
        agg_scale=1.0,
        color=clist[5],           # green
        linestyle=":",
    ),

    # ── Full-batch ExoTFT-FB 모델 ──────────────────────────────────────────
    "base": ModelSpec(
        key="base",
        label=LABEL_FB,
        short="ExoTFT-FB",
        file="site{site}_base_fullbatch_test_results.csv",
        pred_col="Base_Pred",
        agg_name="base",
        agg_scale=1.0,            # 이미 [%]
        color=clist[0],           # deep blue
        linestyle="--",
    ),

    # ── ExoTFT-RF (보정 전) ─────────────────────────────────────────
    #   잔차 보정 직전 단계. ExoTFT-CRC 결과 CSV 안의 Rolling_Base_Pred 컬럼입니다.
    #   집계표(model_total_results.csv)에는 항목이 없어 agg_name 은 None →
    #   fig4 는 agg_name 필터로 자동 제외됩니다.
    #   잔차 보정만의 순기여를 보려면 이 모델이 기준선이어야 합니다
    #   (ExoTFT-FB 와 비교하면 롤링 재학습 효과가 섞임).
    "rolling": ModelSpec(
        key="rolling",
        label=LABEL_RF,
        short="ExoTFT-RF",
        file="site{site}_residual_adaptive_results.csv",
        pred_col="Rolling_Base_Pred",
        color=clist[1],           # sky blue
        linestyle="--",
    ),

    # ── 제안 모델: ExoTFT-RF + Adaptive-α 잔차 보정 ────────────────────────
    "reco": ModelSpec(
        key="reco",
        label=LABEL_CRC,
        short="ExoTFT-CRC",
        file="site{site}_residual_adaptive_results.csv",
        pred_col="Final_Ensemble_Pred",
        agg_name="\uc794\ucc28\ubcf4\uc815",
        agg_scale=1.0,
        color=clist[2],           # muted red
        extra_cols=["Rolling_Base_Pred", "Alpha_Used"],
        linestyle="-",
        zorder=5,
    ),
}

# 집계 성능표(fig4) 용 순서 — 항상 ExoTFT-FB 기준. rolling 은 agg_name 이 없어
# fig4 의 agg_name 필터에서 자동 제외되므로 여기 넣지 않습니다.
MODEL_ORDER = ["dynamic", "chen", "kong", "base", "reco"]

RECO_KEY = "reco"

# ══════════════════════════════════════════════════════════════════════════════
# 기준선(baseline) 전역 스위치
#
#   True  → 시계열 figure(5_d~8_x)의 "Base" 자리에 **ExoTFT-RF**(보정 전)를 씁니다.
#           ExoTFT-CRC 와의 차이가 잔차 보정 α·ê 의 순기여만 담습니다.
#   False → ExoTFT-FB. 차이에 롤링 재학습 효과가 섞여 들어갑니다.
#
# 사이트별 일평균 NMAE 분해(통학습→롤링→ExoTFT-CRC)를 보면 개선폭의 대부분이
# 롤링 재학습에서 나옵니다(예: site 7 은 +2.081 vs +0.025). 잔차 모델의 기여를
# 주장하려면 True 가 맞는 기준선입니다.
#
# fig4 는 model_total_results.csv 집계표를 쓰고 그 표에 rolling 항목이 없어
# 이 스위치의 영향을 받지 않습니다.
# ══════════════════════════════════════════════════════════════════════════════
USE_ROLLING_AS_BASE = True

BASE_KEY = "rolling" if USE_ROLLING_AS_BASE else "base"

# ExoTFT-FB 는 True_Target 기준 파일이자 비교용으로 항상 로드해 둡니다.
FULLBATCH_BASE_KEY = "base"


def ts_model_order(order=None):
    """시계열 figure 용 기본 모델 순서.

    order=None (기본) 일 때만 스위치를 적용해 base → rolling 로 치환합니다.
    order 를 명시해서 넘기면 **그대로 존중**합니다 — 스위치를 적용받지 않아야 하는
    그림(6_a, 6_b, 7 처럼 ExoTFT-FB 와의 비교가 목적인 그림)이
    order=config.MODEL_ORDER 를 넘겨 opt-out 할 수 있게 하기 위함입니다.
    """
    if order is not None:
        return list(order)
    if not USE_ROLLING_AS_BASE:
        return list(MODEL_ORDER)
    return [BASE_KEY if k == FULLBATCH_BASE_KEY else k for k in MODEL_ORDER]


def model_specs(order=None, site=None, available_only=True):
    """시계열 figure 용 모델 순서대로 ModelSpec 리스트 반환.
    기본 순서는 ts_model_order() 이므로 USE_ROLLING_AS_BASE 스위치를 따릅니다.
    available_only=True 이고 site 가 주어지면 시계열 결과 파일이 실제로 있는 것만
    (= 집계값만 있는 비교 모델은 제외)."""
    order = ts_model_order(order)
    specs = [MODELS[k] for k in order if k in MODELS]
    if available_only and site is not None:
        specs = [s for s in specs if s.exists(site)]
    return specs


def load_total_results(normalize: bool = True):
    """model_total_results.csv 를 사이트 × 모델 표로 로드 (nmae/nrmse 를 [%] 로 통일)."""
    import pandas as pd

    df = pd.read_csv(TOTAL_RESULTS_FILE, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["site_num"] = df["site"].astype(str).str.extract(r"(\d+)").astype(int)

    agg2key = {m.agg_name: k for k, m in MODELS.items() if m.agg_name}
    df["key"] = df["model"].map(agg2key)
    unknown = sorted(set(df.loc[df["key"].isna(), "model"]))
    if unknown:
        print(f"[warn] config.MODELS 에 없는 모델: {unknown}")
    df = df.dropna(subset=["key"])

    if normalize:
        scale = df["key"].map({k: m.agg_scale for k, m in MODELS.items()})
        for site_label, model_name in AGG_CELL_OVERRIDES:
            mask = (df["site"] == site_label) & (df["model"] == model_name)
            scale = scale.where(~mask, AGG_CELL_OVERRIDES[(site_label, model_name)])
        df["nmae"] = df["nmae"] * scale
        df["nrmse"] = df["nrmse"] * scale

    df["label"] = df["key"].map({k: m.label for k, m in MODELS.items()})
    return df


def model_colors(specs):
    return [s.color for s in specs]
