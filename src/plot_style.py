"""모든 모듈과 노트북이 함께 쓰는 Matplotlib 설정과 저장 함수.

계산에는 쓰지 않고, 그림의 글꼴(한글)과 저장 위치(outputs/)만 담당한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, ticker

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

BLUE, RED, GREEN, ORANGE, GRAY = "#378ADD", "#E24B4A", "#1D9E75", "#BA7517", "#888780"
PURPLE, PINK = "#7F77DD", "#D4537E"


def setup_style():
    """설치된 한글 글꼴을 찾아 Matplotlib 기본 글꼴로 지정한다.

    macOS(AppleGothic), Linux(NanumGothic), Windows(Malgun Gothic) 순서로 찾고,
    없으면 기본 글꼴(DejaVu Sans)을 쓴다. 이 경우 한글이 네모로 보일 수 있다.
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next(
        (f for f in ("AppleGothic", "NanumGothic", "Malgun Gothic") if f in installed),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, name):
    """그림을 outputs/<name>.png 로 저장하고 저장 경로를 돌려준다.

    Args:
        fig: 저장할 Matplotlib Figure.
        name: 확장자를 뺀 파일 이름.

    Returns:
        저장된 파일의 Path.
    """
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[저장] {path.relative_to(ROOT)}")
    return path


def plain_log_ticks(ax, axis="both"):
    """로그 축 눈금을 10^-5 대신 1e-5 형태로 표시한다.

    한글 글꼴에는 수식용 마이너스 기호(U+2212)가 없어 깨지는 것을 막기 위함이다.

    Args:
        ax: Matplotlib Axes.
        axis: "x", "y", "both" 중 하나.
    """
    fmt = ticker.FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else "0")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def save_animation(anim, name, fps=15):
    """Matplotlib 애니메이션을 outputs/<name>.gif 로 저장한다 (Pillow 사용).

    GitHub README에서 GIF는 별도 설정 없이 바로 재생된다.

    Args:
        anim: matplotlib.animation.FuncAnimation 객체.
        name: 확장자를 뺀 파일 이름.
        fps: 초당 프레임 수.

    Returns:
        저장된 파일의 Path.
    """
    from matplotlib.animation import PillowWriter

    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{name}.gif"
    anim.save(path, writer=PillowWriter(fps=fps), dpi=72)
    size_kb = path.stat().st_size / 1024
    print(f"[저장] {path.relative_to(ROOT)} ({size_kb:.0f} KB)")
    return path
