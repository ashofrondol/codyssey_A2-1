"""README 개념 설명용 그림을 생성하는 스크립트.

과제 결과물(outputs/)과는 별개다. 개념을 보여 주기 위해
과제에서 요구하는 실험과는 **다른 함수/값**을 일부러 사용한다.

실행:
    uv run python docs/figures/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

np.random.seed(42)

OUT = Path(__file__).resolve().parent
_installed = {f.name for f in font_manager.fontManager.ttflist}
KOREAN_FONT = next(
    (f for f in ("AppleGothic", "NanumGothic", "Malgun Gothic") if f in _installed),
    "DejaVu Sans",
)
plt.rcParams.update(
    {
        "font.family": KOREAN_FONT,
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BLUE, RED, GREEN, ORANGE, GRAY = "#378ADD", "#E24B4A", "#1D9E75", "#BA7517", "#888780"


def save(fig, name):
    """그림을 docs/figures/<name>.png 로 저장하고 닫는다."""
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def f_sq(x):
    """f(x) = x²"""
    return x**2


def fig_terrain():
    """2변수 함수를 '지형'으로 보는 법: 옆에서 본 3D 모습과 위에서 본 등고선 지도."""
    g = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(g, g)
    Z = X**2 + 3 * Y**2
    px, py = 2.0, 1.0
    pz = px**2 + 3 * py**2
    fig = plt.figure(figsize=(11, 4.4))
    a1 = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    a1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.55, linewidth=0, zorder=1)
    a1.contour(X, Y, Z, levels=[1, 3, 6, 10, 15], zdir="z", offset=0, colors=GRAY, linewidths=0.8)
    a1.plot([px, px], [py, py], [0, pz], color=RED, lw=1.5, ls="--", zorder=3)
    a1.scatter([px], [py], [pz], color=RED, s=45, depthshade=False, zorder=4)
    a1.scatter([0], [0], [0], color="black", s=30, depthshade=False, zorder=4)
    a1.text2D(0.02, 0.95, f"빨간 점: 위치 ({px:.0f}, {py:.0f})의 높이 = {pz:.0f}\n검은 점: 가장 낮은 바닥 (0, 0)",
              transform=a1.transAxes, fontsize=9, color=RED, va="top")
    a1.set(xlabel="x (동서)", ylabel="y (남북)", zlabel="높이 f(x, y)",
           title="옆에서 본 모습: 그릇 모양의 지형")
    a1.view_init(elev=28, azim=-60)
    a2 = fig.add_subplot(1, 2, 2)
    cs = a2.contour(X, Y, Z, levels=[1, 3, 6, 10, 15], colors=GRAY, linewidths=1)
    a2.clabel(cs, fontsize=8, fmt="높이 %d")
    a2.plot(px, py, "o", color=RED)
    a2.annotate(f"위치 ({px:.0f}, {py:.0f})\n높이 {pz:.0f}", (px, py), xytext=(8, -26),
                textcoords="offset points", color=RED, fontsize=9)
    a2.plot(0, 0, "o", color="black")
    a2.annotate("바닥", (0, 0), xytext=(6, -12), textcoords="offset points", fontsize=9)
    a2.set_aspect("equal")
    a2.set(xlabel="x (동서)", ylabel="y (남북)",
           title="위에서 내려다본 지도: 등고선 = 높이가 같은 점을 이은 선")
    fig.suptitle("$f(x, y) = x^2 + 3y^2$ : (x, y)를 넣으면 높이 하나가 나오는 함수 = 지형", y=1.0)
    save(fig, "fig00a_terrain")


def _arrow(ax, x0, y0, dx, dy, color, label=None, lw=2.2):
    """원점 (x0, y0)에서 (dx, dy)만큼 가는 화살표를 그린다."""
    ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14))
    if label:
        ax.text(x0 + dx * 0.55, y0 + dy * 0.55, label, color=color, fontsize=10,
                ha="right", va="bottom")


def _grid_axes(ax, lim):
    """모눈과 x, y축이 있는 정사각형 좌표평면을 만든다."""
    ax.set(xlim=lim, ylim=lim, xticks=range(int(lim[0]), int(lim[1]) + 1),
           yticks=range(int(lim[0]), int(lim[1]) + 1))
    ax.grid(color="#dddddd", lw=0.6)
    ax.axhline(0, color=GRAY, lw=1)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_aspect("equal")


def fig_vector_basics():
    """벡터 = 화살표, 길이(norm), 단위벡터와 정규화, 벡터 덧셈."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    a = axes[0]
    _grid_axes(a, (-0.5, 5))
    a.plot([0, 3], [0, 0], color=BLUE, ls="--", lw=1.4)
    a.plot([3, 3], [0, 4], color=GREEN, ls="--", lw=1.4)
    a.text(1.5, -0.35, "가로 성분 3", color=BLUE, ha="center", fontsize=9)
    a.text(3.15, 2, "세로 성분 4", color=GREEN, fontsize=9)
    _arrow(a, 0, 0, 3, 4, RED)
    a.text(1.0, 2.2, r"길이 = $\sqrt{3^2 + 4^2}$ = 5", color=RED, fontsize=10, ha="center", rotation=53)
    a.set_title("벡터 (3, 4) = 오른쪽 3, 위로 4 가는 화살표")

    a = axes[1]
    _grid_axes(a, (-1.5, 4.5))
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(t), np.sin(t), color=GRAY, ls=":", lw=1.2)
    a.text(-1.35, -1.35, "길이 1인 원", color=GRAY, fontsize=9)
    _arrow(a, 0, 0, 3, 4, RED, lw=1.4)
    a.text(3.05, 4.05, "(3, 4)  길이 5", color=RED, fontsize=9)
    _arrow(a, 0, 0, 0.6, 0.8, ORANGE, lw=2.6)
    a.text(0.75, 0.55, "(0.6, 0.8)\n길이 1", color=ORANGE, fontsize=9)
    _arrow(a, 0, 0, 1, 0, BLUE, lw=2)
    a.text(1.05, -0.4, r"$\hat{\imath}$ = (1, 0)", color=BLUE, fontsize=9)
    _arrow(a, 0, 0, 0, 1, GREEN, lw=2)
    a.text(-1.35, 1.1, r"$\hat{\jmath}$ = (0, 1)", color=GREEN, fontsize=9)
    a.set_title("단위벡터 = 길이가 1인 화살표\n정규화: (3, 4) ÷ 5 = (0.6, 0.8), 방향은 그대로")

    a = axes[2]
    _grid_axes(a, (-0.5, 5))
    _arrow(a, 0, 0, 1, 2, BLUE, "(1, 2)  ")
    _arrow(a, 1, 2, 3, 1, GREEN)
    a.text(2.6, 2.75, "(3, 1)", color=GREEN, fontsize=10)
    _arrow(a, 0, 0, 4, 3, RED)
    a.text(2.6, 1.3, "(4, 3)", color=RED, fontsize=10)
    a.set_title("벡터 덧셈 = 화살표 이어 붙이기\n(1, 2) + (3, 1) = (4, 3)")
    save(fig, "fig00b_vector_basics")


def fig_dot_product():
    """내적 = (a의 길이) × (b의 그림자 길이). 각도에 따라 양수, 0, 음수가 된다."""
    a_vec = np.array([3.0, 0.0])
    cases = [(30, "같은 쪽 → 양수"), (90, "직각 → 0"), (150, "반대쪽 → 음수")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (deg, desc) in zip(axes, cases):
        th = np.deg2rad(deg)
        b_vec = 2 * np.array([np.cos(th), np.sin(th)])
        dot = a_vec @ b_vec
        shadow = b_vec[0]  # a가 x축 방향이므로 그림자는 b의 x 성분
        ax.set(xlim=(-2.2, 3.6), ylim=(-0.9, 2.4), xticks=[], yticks=[])
        ax.set_aspect("equal")
        ax.axhline(0, color="#dddddd", lw=0.8)
        ax.plot([b_vec[0], b_vec[0]], [0, b_vec[1]], color=GRAY, ls=":", lw=1.2)
        ax.plot([0, shadow], [-0.25, -0.25], color=ORANGE, lw=5, solid_capstyle="butt")
        ax.text(shadow / 2, -0.6, f"그림자 {shadow:.2f}", color=ORANGE, ha="center", fontsize=9)
        _arrow(ax, 0, 0, *a_vec, BLUE)
        ax.text(3.0, 0.12, "a = (3, 0)", color=BLUE, ha="right", fontsize=9)
        _arrow(ax, 0, 0, *b_vec, RED)
        ax.text(b_vec[0], b_vec[1] + 0.12, f"b (길이 2, {deg}도)", color=RED, ha="left" if deg < 90 else ("center" if deg == 90 else "left"), fontsize=9)
        ax.set_title(f"{desc}\na · b = 3 × {shadow:.2f} = {dot:.2f}", fontsize=10)
    fig.suptitle("내적 = a의 길이 × (b가 a 방향에 드리운 그림자 길이)", fontsize=11)
    save(fig, "fig00d_dot_product")


def fig_trig():
    """삼각함수: 반지름 1인 원 위의 점 (cos θ, sin θ), 기울기 tan θ, 각도에 따른 그래프."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    th = np.deg2rad(30)
    c, s = np.cos(th), np.sin(th)
    t = np.linspace(0, 2 * np.pi, 300)
    a1.plot(np.cos(t), np.sin(t), color=GRAY, ls=":", lw=1.2)
    a1.axhline(0, color=GRAY, lw=0.8)
    a1.axvline(0, color=GRAY, lw=0.8)
    a1.plot([0, c], [0, s], color="black", lw=2)
    a1.plot([0, c], [0, 0], color=BLUE, lw=5, solid_capstyle="butt")
    a1.plot([c, c], [0, s], color=ORANGE, lw=5, solid_capstyle="butt")
    a1.plot(c, s, "o", color="black", ms=7)
    a1.plot([0, 1.3], [0, 1.3 * np.tan(th)], color=GREEN, ls="--", lw=1.2)
    ts = np.linspace(0, th, 30)
    a1.plot(0.25 * np.cos(ts), 0.25 * np.sin(ts), color="black", lw=1)
    a1.text(0.28, 0.06, "θ = 30도", fontsize=9)
    a1.text(c / 2, -0.13, f"cos θ = {c:.3f}", color=BLUE, ha="center", fontsize=10)
    a1.text(c + 0.05, s / 2, f"sin θ = {s:.1f}", color=ORANGE, fontsize=10)
    a1.text(c + 0.03, s + 0.07, f"({c:.3f}, {s:.1f})", fontsize=9)
    a1.text(0.55, 0.82, f"tan θ = sin ÷ cos\n= {np.tan(th):.3f} (기울기)", color=GREEN, fontsize=9)
    a1.text(0.62, -0.62, "반지름 1", color=GRAY, fontsize=9)
    a1.set(xlim=(-1.25, 1.45), ylim=(-1.25, 1.25), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           title="반지름 1인 원 위의 점 = (cos θ, sin θ)")
    a1.set_aspect("equal")

    deg = np.linspace(0, 360, 361)
    a2.plot(deg, np.cos(np.deg2rad(deg)), color=BLUE, lw=2, label="cos θ (가로 위치)")
    a2.plot(deg, np.sin(np.deg2rad(deg)), color=ORANGE, lw=2, label="sin θ (세로 위치)")
    a2.axhline(0, color=GRAY, lw=0.8)
    for d in (30, 90, 180, 270):
        a2.axvline(d, color=GRAY, lw=0.6, ls=":")
    a2.plot(30, c, "o", color=BLUE)
    a2.plot(30, s, "o", color=ORANGE)
    a2.set(xticks=[0, 30, 90, 180, 270, 360], xlabel="각도 θ (도)", ylim=(-1.2, 1.2),
           title="각도를 바꾸면: 항상 -1 ~ 1 사이를 오르내린다")
    a2.legend(loc="lower left", fontsize=9)
    save(fig, "fig00g_trig")


def fig_e_number():
    """e가 나오는 곳: ① 복리 (1 + 1/n)^n의 극한 ② e^x는 높이와 기울기가 같다."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    n = np.unique(np.logspace(0, 4, 60).astype(int))
    a1.plot(n, (1 + 1 / n) ** n, "o-", color=BLUE, ms=3, lw=1.5)
    a1.axhline(np.e, color=RED, ls="--", lw=1.2)
    a1.text(1.2, np.e + 0.03, "e = 2.71828...", color=RED, fontsize=10)
    for k in (1, 2, 12, 365):
        a1.annotate(f"n={k}: {(1 + 1 / k) ** k:.3f}", xy=(k, (1 + 1 / k) ** k),
                    xytext=(k * 2.2, (1 + 1 / k) ** k - 0.15), fontsize=8, color=GRAY,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
    a1.set_xscale("log")
    a1.set(xlabel="1년을 몇 번으로 나눠 이자를 붙이나 (n)", ylabel="1년 뒤 금액 (원금 1)",
           ylim=(1.8, 2.85), title="연 100% 이자를 잘게 나눠 붙이면 e에 다가간다")

    x = np.linspace(-2, 2, 200)
    a2.plot(x, np.exp(x), color=BLUE, lw=2.2, label="$e^x$")
    for x0, c in [(0, GREEN), (1, ORANGE)]:
        y0 = np.exp(x0)
        xs = np.linspace(x0 - 1, x0 + 0.8, 10)
        a2.plot(xs, y0 + y0 * (xs - x0), color=c, ls="--", lw=1.4)
        a2.plot(x0, y0, "o", color=c)
        a2.text(x0 - 0.15, y0 + 0.35, f"x={x0}: 높이 {y0:.2f}, 기울기 {y0:.2f}", color=c,
                fontsize=9, ha="right")
    a2.set(xlabel="x", ylabel="y", ylim=(-0.5, 7.4),
           title="$e^x$는 어느 점에서나 높이 = 기울기\n→ 미분해도 자기 자신 ($(e^x)' = e^x$)")
    a2.legend(loc="upper left")
    save(fig, "fig00e_e_number")


def fig_mse():
    """MSE = 오차를 한 변으로 하는 정사각형 넓이의 평균."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 4.0, 9.0])
    y_hat = 2 * x + 1
    err = y - y_hat
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.6))
    xs = np.linspace(0.5, 3.8, 50)
    for ax in (a1, a2):
        ax.plot(xs, 2 * xs + 1, color=BLUE, lw=2, label=r"예측 $\hat{y}$ = 2x + 1")
        ax.plot(x, y, "o", color=RED, ms=7, label="정답 y", zorder=5)
        ax.set(xlim=(0.4, 5.0), ylim=(1.5, 10), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
    for xi, yi, yh, e in zip(x, y, y_hat, err):
        a1.plot([xi, xi], [yh, yi], color=ORANGE, lw=2)
        a1.text(xi + 0.1, (yi + yh) / 2, f"오차 {e:+.0f}", color=ORANGE, fontsize=9, va="center")
        a2.add_patch(plt.Rectangle((xi, min(yi, yh)), abs(e), abs(e), color=ORANGE, alpha=0.35))
        a2.text(xi + abs(e) + 0.08, (yi + yh) / 2, f"넓이 {e**2:.0f}", color=ORANGE, fontsize=9,
                va="center")
    a1.set_title(f"오차 = 정답 - 예측: {err[0]:+.0f}, {err[1]:+.0f}, {err[2]:+.0f}\n"
                 "그냥 더하면 +1과 -1이 지워진다")
    a2.set_title(f"제곱 = 오차를 한 변으로 하는 정사각형 넓이\n"
                 f"MSE = (1 + 1 + 4) ÷ 3 = {np.mean(err**2):.0f}")
    a1.legend(loc="lower right", fontsize=9)
    save(fig, "fig00f_mse")


def fig_secant_tangent():
    """할선이 h → 0 일 때 접선에 다가가는 모습."""
    x = np.linspace(-0.3, 4.8, 300)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, f_sq(x), color="black", lw=2, label="곡선 $y=x^2$")
    xs = np.linspace(1.5, 4.8, 2)
    ax.plot(xs, 9 + 6 * (xs - 3), color=BLUE, lw=2, label="접선 (기울기 6)")
    for h, a in [(1.0, 1.0), (0.5, 0.6), (0.1, 0.35)]:
        s = (f_sq(3 + h) - 9) / h
        ax.plot(xs, 9 + s * (xs - 3), color=RED, lw=1.5, alpha=a,
                label=f"할선 h={h} (기울기 {s:.1f})")
        ax.plot(3 + h, f_sq(3 + h), "o", color=RED, alpha=a)
    ax.plot(3, 9, "o", color="black")
    ax.annotate("(3, 9)", (3, 9), xytext=(-42, 8), textcoords="offset points")
    ax.set(xlim=(-0.3, 4.8), ylim=(-1, 24), xlabel="x", ylabel="y",
           title="h를 줄이면 할선(빨강)이 접선(파랑)에 다가간다")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig01_secant_tangent")


def fig_zoom_tangent():
    """곡선을 한 점 주변으로 확대할수록 직선(접선)처럼 보인다."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, w in zip(axes, [2.0, 0.5, 0.05]):
        x = np.linspace(3 - w, 3 + w, 300)
        ax.plot(x, f_sq(x), color="black", lw=2.5, label="곡선 $y=x^2$")
        ax.plot(x, 9 + 6 * (x - 3), color=BLUE, lw=1.5, ls="--", label="기울기 6인 직선")
        ax.plot(3, 9, "o", color=RED, zorder=5)
        ax.set(xlim=(3 - w, 3 + w), xlabel="x", ylabel="y",
               title=f"x = {3 - w:g} ~ {3 + w:g} 구간을 확대")
        ax.legend(fontsize=9, loc="upper left")
    axes[2].annotate("곡선과 직선이\n거의 구별되지 않는다", (3.02, 9.12), xytext=(3.005, 8.75),
                     fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    fig.suptitle("점 (3, 9) 주변을 확대할수록 곡선이 기울기 6인 직선처럼 보인다 "
                 "= 한 점에서의 기울기", y=1.03)
    save(fig, "fig01b_zoom_tangent")


def fig_three_differences():
    """전방·후방·중심차분 할선을 접선과 비교."""
    x = np.linspace(1.2, 4.6, 300)
    xs = np.array([1.4, 4.6])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, f_sq(x), color="black", lw=2, label="곡선 $y=x^2$")
    ax.plot(xs, 9 + 6 * (xs - 3), color=BLUE, lw=3, alpha=0.5, label="접선: 기울기 6")
    ax.plot([3, 4], [9, 16], "-o", color=RED, lw=2, label="전방차분 (3→4): 기울기 7")
    ax.plot([2, 3], [4, 9], "-o", color=GREEN, lw=2, label="후방차분 (2→3): 기울기 5")
    ax.plot([2, 4], [4, 16], "--o", color=ORANGE, lw=2, label="중심차분 (2→4): 기울기 6")
    ax.set(xlabel="x", ylabel="y", title="x=3, h=1 일 때 세 가지 차분")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig02_three_differences")


def fig_h_error():
    """h 크기에 따른 오차: 절단 오차 vs 반올림 오차 (f = sin, x = 1)."""
    hs = np.logspace(-16, 0, 200)
    x0, true = 1.0, np.cos(1.0)
    fwd = np.abs((np.sin(x0 + hs) - np.sin(x0)) / hs - true)
    cen = np.abs((np.sin(x0 + hs) - np.sin(x0 - hs)) / (2 * hs) - true)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(hs, fwd + 1e-18, color=RED, label="전방차분 오차")
    ax.loglog(hs, cen + 1e-18, color=ORANGE, label="중심차분 오차")
    ax.set(xlabel="h (로그 스케일)", ylabel="|근사값 - 정답| (로그 스케일)",
           title="h가 너무 커도, 너무 작아도 오차가 커진다  ($f(x)=\\sin x$, x=1)")
    ax.text(1e-15, 1e-11, "반올림 오차 구간\n(너무 작은 h)", fontsize=9, color=GRAY)
    ax.text(2e-3, 1e-10, "절단 오차 구간\n(너무 큰 h)", fontsize=9, color=GRAY)
    ax.set_ylim(1e-13, 10)
    exp_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}")
    ax.xaxis.set_major_formatter(exp_fmt)
    ax.yaxis.set_major_formatter(exp_fmt)
    ax.legend(fontsize=9, loc="upper center")
    save(fig, "fig03_h_error")


def fig_gradient_contour():
    """f(x,y) = x² + 3y² 의 등고선과 gradient 화살표 (등고선에 수직)."""
    g = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(g, g)
    Z = X**2 + 3 * Y**2
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    cs = ax.contour(X, Y, Z, levels=[1, 3, 6, 10, 15], colors=GRAY, linewidths=1)
    ax.clabel(cs, fontsize=8)
    pts = []
    for c, ts in [(3, [0.3, 1.9, 3.6, 5.2]), (6, [1.0, 2.7, 4.4, 5.9])]:
        for t in ts:
            pts.append([np.sqrt(c) * np.cos(t), np.sqrt(c / 3) * np.sin(t)])
    pts = np.array(pts)
    gx, gy = 2 * pts[:, 0], 6 * pts[:, 1]
    norm = np.hypot(gx, gy)
    ax.quiver(pts[:, 0], pts[:, 1], gx / norm, gy / norm, color=RED, scale=8, width=0.008)
    ax.plot(pts[:, 0], pts[:, 1], "o", color=RED, ms=4)
    ax.set_aspect("equal")
    ax.set(xlabel="x", ylabel="y",
           title="$f(x,y)=x^2+3y^2$ 의 gradient(빨강)\n등고선에 수직 = 가장 가파르게 오르는 방향")
    save(fig, "fig04_gradient_contour")


def fig_linear_transform():
    """격자와 단위 정사각형이 행렬에 의해 어떻게 변하는지."""
    th = np.deg2rad(30)
    mats = [
        ("변환 없음 (I)", np.eye(2)),
        ("회전 30°", np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])),
        ("스케일 (1.5, 0.5)", np.array([[1.5, 0], [0, 0.5]])),
        ("전단 k=1", np.array([[1, 1], [0, 1]])),
    ]
    sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (name, A) in zip(axes, mats):
        for t in np.linspace(-2, 2, 9):
            for line in (np.array([[t, t], [-2, 2]]), np.array([[-2, 2], [t, t]])):
                l2 = A @ line
                ax.plot(l2[0], l2[1], color=GRAY, lw=0.6, alpha=0.6)
        s2 = A @ sq
        ax.fill(s2[0], s2[1], color=BLUE, alpha=0.35)
        ax.plot(s2[0], s2[1], color=BLUE, lw=1.5)
        ax.annotate("", A[:, 0], (0, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.annotate("", A[:, 1], (0, 0), arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
        ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), title=f"{name}\ndet = {np.linalg.det(A):.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("행렬은 공간 전체를 변형한다 — 파란 정사각형(넓이 1)의 새 넓이 = |det|", y=1.02)
    save(fig, "fig05_linear_transform")


def fig_eigenvectors():
    """A=[[2,1],[1,2]] 에서 대부분의 벡터는 방향이 바뀌지만 고유벡터는 그대로."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    angs = np.deg2rad(np.arange(0, 180, 22.5))
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for a in angs:
        v = np.array([np.cos(a), np.sin(a)])
        Av = A @ v
        eig = np.isclose(abs(np.cos(a - np.pi / 4)), 1) or np.isclose(abs(np.cos(a + np.pi / 4)), 1)
        c = RED if eig else GRAY
        ax.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.6))
        ax.annotate("", Av, (0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2 if eig else 1))
    for d, lam in [((1, 1), 3), ((-1, 1), 1)]:
        d = np.array(d) / np.sqrt(2)
        ax.plot([-3.2 * d[0], 3.2 * d[0]], [-3.2 * d[1], 3.2 * d[1]], ":", color=RED, lw=1)
        ax.text(*(d * 2.25 * (1 if lam == 3 else 1.6)), f"λ={lam}", color=RED)
    ax.set(xlim=(-3.3, 3.3), ylim=(-0.5, 3.3),
           title="$A=[[2,1],[1,2]]$: 얇은 화살표 v → 굵은 화살표 Av\n빨강(고유벡터)만 같은 직선 위에 남는다")
    ax.set_aspect("equal")
    save(fig, "fig06_eigenvectors")


def fig_eigen_uses():
    """고유값이 쓰이는 곳: 반복의 운명(λ^n), 장기 비율(마르코프 체인), 데이터의 주된 방향(PCA)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    a = axes[0]
    n = np.arange(0, 21)
    for lam, c in [(1.2, RED), (1.0, GRAY), (0.8, BLUE)]:
        a.plot(n, lam**n, "-o", ms=3, color=c, label=f"λ = {lam}: " + ("폭발" if lam > 1 else "유지" if lam == 1 else "소멸"))
    a.set(xlabel="곱한 횟수 n", ylabel="길이 배율 $\\lambda^n$", ylim=(0, 10),
          title="같은 행렬을 n번 곱하면\n고유벡터 방향은 $\\lambda^n$ 배가 된다")
    a.legend(fontsize=9)

    a = axes[1]
    P = np.array([[0.9, 0.5], [0.1, 0.5]])
    for v0, c, lab in [((1.0, 0.0), ORANGE, "첫날 맑음"), ((0.0, 1.0), BLUE, "첫날 비")]:
        v = np.array(v0)
        ys = [v[0]]
        for _ in range(12):
            v = P @ v
            ys.append(v[0])
        a.plot(range(13), ys, "-o", ms=3, color=c, label=lab)
    a.axhline(5 / 6, color=RED, ls="--", lw=1.2, label="고유값 1의 고유벡터: 5/6")
    a.set(xlabel="날짜", ylabel="맑을 확률", ylim=(0, 1.05),
          title="날씨 모델: 출발이 달라도\n결국 같은 비율(고유벡터)에 도착한다")
    a.legend(fontsize=9, loc="lower right")

    a = axes[2]
    h = np.random.normal(170, 7, 200)
    w = 0.9 * (h - 170) + 65 + np.random.normal(0, 4, 200)
    X = np.c_[h, w]
    m = X.mean(axis=0)
    lam, U = np.linalg.eigh(np.cov(X.T))
    a.scatter(X[:, 0], X[:, 1], s=8, color=GRAY, alpha=0.6)
    for i, c in [(1, RED), (0, GREEN)]:
        d = U[:, i] * 2 * np.sqrt(lam[i])
        a.annotate("", m + d, m, arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
    ratio = lam[1] / lam.sum()
    a.text(0.03, 0.97, f"빨강: 가장 큰 고유값 방향 (퍼짐의 {ratio:.0%})\n초록: 두 번째 방향 (퍼짐의 {1 - ratio:.0%})",
           transform=a.transAxes, va="top", fontsize=9)
    a.set(xlabel="키 (cm)", ylabel="몸무게 (kg)",
          title="PCA: 데이터가 가장 많이 퍼진 방향\n= 가장 큰 고유값의 고유벡터")
    a.set_aspect("equal")
    save(fig, "fig07_eigen_uses")


def fig_svd_rank():
    """합성 이미지의 rank-k 근사와 특이값 크기."""
    n = 48
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = 0.5 * xx + 0.3 * np.sin(6 * yy)
    img += ((xx - 0.6) ** 2 + (yy - 0.4) ** 2 < 0.04) * 0.6
    img += np.random.rand(n, n) * 0.05
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
    for ax, k in zip(axes[:4], [1, 3, 8, n]):
        rec = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
        ax.imshow(rec, cmap="gray")
        ax.set_title(f"k = {k}" + (" (원본)" if k == n else ""))
        ax.axis("off")
    axes[4].bar(np.arange(1, 16), S[:15], color=BLUE)
    axes[4].set(title="특이값 $\\sigma$ (앞의 15개)", xlabel="순서")
    fig.suptitle("SVD: 큰 특이값 몇 개만 남겨도 이미지의 대부분이 복원된다 (48×48 합성 이미지)", y=1.04)
    save(fig, "fig08_svd_rank")


def fig_lr_1d():
    """1차원 f(x)=x² 에서 학습률에 따른 경사하강 경로."""
    x = np.linspace(-5, 5, 300)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, lr in zip(axes, [0.05, 0.3, 0.9, 1.05]):
        ax.plot(x, x**2, color="black", lw=1.5)
        p = [4.0]
        for _ in range(8):
            p.append(p[-1] - lr * 2 * p[-1])
        p = np.array(p)
        ax.plot(p, p**2, "-o", color=RED, ms=4, lw=1)
        state = {0.05: "느리게 수렴", 0.3: "빠르게 수렴", 0.9: "진동하며 수렴", 1.05: "발산"}[lr]
        ax.set(title=f"lr = {lr}: {state}", xlim=(-5.5, 5.5), ylim=(-1, 30), xlabel="x")
    axes[0].set_ylabel("f(x) = x²")
    fig.suptitle("학습률(lr)에 따라 같은 출발점(x=4)에서도 결과가 완전히 달라진다 (8번 이동)", y=1.04)
    save(fig, "fig09_lr_1d")


def fig_network():
    """2-2-1 신경망 구조도."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    L = {"x1": (0, 1.6), "x2": (0, 0.4), "h1": (2.4, 1.6), "h2": (2.4, 0.4), "y": (4.8, 1.0)}
    for a in ("x1", "x2"):
        for b in ("h1", "h2"):
            ax.plot(*zip(L[a], L[b]), color=BLUE, lw=1.2)
    for b in ("h1", "h2"):
        ax.plot(*zip(L[b], L["y"]), color=ORANGE, lw=1.2)
    labels = {"x1": "$x_1$", "x2": "$x_2$", "h1": "$z^{(1)}_1 \\to a_1$",
              "h2": "$z^{(1)}_2 \\to a_2$", "y": "$z_2 \\to \\hat{y}$"}
    for k, (px, py) in L.items():
        ax.add_patch(plt.Circle((px, py), 0.28, color="white", ec="black", zorder=3))
        ax.text(px, py - 0.5, labels[k], ha="center", fontsize=10)
    ax.text(1.2, 2.05, "$W_1$ (2×2), $b_1$", ha="center", color=BLUE)
    ax.text(3.6, 1.75, "$W_2$ (2,), $b_2$", ha="center", color=ORANGE)
    for px, t in [(0, "입력층 x\nshape (2,)"), (2.4, "은닉층 (sigmoid)\nshape (2,)"),
                  (4.8, "출력층 (sigmoid)\nscalar")]:
        ax.text(px, -0.65, t, ha="center", fontsize=9, color=GRAY)
    ax.set(xlim=(-0.8, 5.6), ylim=(-1.1, 2.4))
    ax.axis("off")
    ax.set_title("2층 신경망: 입력 2개 → 은닉 2개 → 출력 1개")
    save(fig, "fig11_network")


def fig_sigmoid_bce():
    """sigmoid와 그 도함수, BCE 손실 곡선."""
    z = np.linspace(-6, 6, 300)
    s = 1 / (1 + np.exp(-z))
    p = np.linspace(0.01, 0.99, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(z, s, color=BLUE, lw=2, label="$\\sigma(z)$")
    a1.plot(z, s * (1 - s), color=ORANGE, lw=2, label="$\\sigma'(z)=\\sigma(1-\\sigma)$")
    a1.set(xlabel="z", title="sigmoid: 어떤 수든 0~1 사이로 눌러 준다")
    a1.legend()
    a2.plot(p, -np.log(p), color=GREEN, lw=2, label="정답 y=1 일 때: $-\\log\\hat{y}$")
    a2.plot(p, -np.log(1 - p), color=RED, lw=2, label="정답 y=0 일 때: $-\\log(1-\\hat{y})$")
    a2.set(xlabel="예측 확률 $\\hat{y}$", ylabel="손실 L", ylim=(0, 4.7),
           title="BCE: 틀린 쪽으로 확신할수록 손실이 폭발한다")
    a2.legend()
    save(fig, "fig12_sigmoid_bce")


def fig_normal_mse():
    """정규분포 PDF와, 음의 로그우도가 오차의 제곱(포물선)이 되는 모습."""
    e = np.linspace(-4, 4, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for sig, c in [(1, BLUE), (2, ORANGE)]:
        pdf = np.exp(-(e**2) / (2 * sig**2)) / np.sqrt(2 * np.pi * sig**2)
        a1.plot(e, pdf, color=c, lw=2, label=f"$\\sigma$ = {sig}")
        a2.plot(e, -np.log(pdf), color=c, lw=2, label=f"$\\sigma$ = {sig}")
    a1.set(xlabel="오차 $\\varepsilon = y - f(x)$", ylabel="확률밀도 $p(\\varepsilon)$",
           title="오차가 정규분포를 따른다고 가정하면")
    a2.set(xlabel="오차 $\\varepsilon = y - f(x)$", ylabel="$-\\log p(\\varepsilon)$",
           title="$-\\log$ 확률은 $\\varepsilon^2$ 모양(포물선) → MSE")
    a1.legend()
    a2.legend()
    save(fig, "fig13_normal_mse")


def fig_softmax():
    """점수(logit)가 softmax를 거쳐 확률로 바뀌는 모습."""
    z = np.array([2.0, 1.0, 0.1])
    p = np.exp(z - z.max())
    p /= p.sum()
    names = ["고양이", "개", "새"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.bar(names, z, color=GRAY)
    a1.set(title="모델의 원점수 z (아무 실수)")
    a2.bar(names, p, color=BLUE)
    for i, v in enumerate(p):
        a2.text(i, v + 0.02, f"{v:.2f}", ha="center")
    a2.set(title=f"softmax(z): 모두 양수, 합 = {p.sum():.1f}", ylim=(0, 1))
    save(fig, "fig14_softmax")



def fig_landscape_problems():
    """경사하강법이 어려워하는 지형: 지역 최솟값, 안장점, 그 근처의 평탄 구간."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    a = axes[0]
    f = lambda x: x**4 - 3 * x**2 + x
    g = lambda x: 4 * x**3 - 6 * x + 1
    xs = np.linspace(-2.1, 2.1, 400)
    a.plot(xs, f(xs), color="black", lw=2)
    for x0, c in [(2.0, RED), (-2.0, BLUE)]:
        x, path = x0, [x0]
        for _ in range(60):
            x -= 0.02 * g(x)
            path.append(x)
        path = np.array(path)
        a.plot(path, f(path), "o-", color=c, ms=3, lw=1, label=f"시작 x = {x0:+.0f} → x = {x:.2f}")
    a.annotate("지역 최솟값\n(local minimum)", (1.13, f(1.13)), xytext=(0.3, 2.5), fontsize=9,
               arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    a.annotate("전역 최솟값\n(global minimum)", (-1.30, f(-1.30)), xytext=(-1.0, 1.5), fontsize=9,
               arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE)
    a.set(xlabel="x", ylabel="f(x)", ylim=(-4.5, 6),
          title="지역 최솟값: 출발점에 따라 다른 골짜기에 갇힌다")
    a.legend(fontsize=8, loc="upper center")

    h = lambda p: p[0] ** 2 + p[1] ** 4 / 4 - p[1] ** 2 / 2
    dh = lambda p: np.array([2 * p[0], p[1] ** 3 - p[1]])
    rng = np.random.RandomState(0)
    runs = []
    for label, start, noise, c in [("GD, 시작 y = 0", (2.0, 0.0), 0.0, GRAY),
                                   ("GD, 시작 y = 0.0001", (2.0, 1e-4), 0.0, BLUE),
                                   ("잡음 있는 GD (SGD 흉내)", (2.0, 0.0), 0.05, GREEN)]:
        p = np.array(start)
        path = [p.copy()]
        for _ in range(120):
            p = p - 0.1 * (dh(p) + noise * rng.randn(2))
            path.append(p.copy())
        runs.append((label, np.array(path), c))
    a = axes[1]
    gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 300), np.linspace(-1.6, 1.6, 300))
    a.contour(gx, gy, h(np.array([gx, gy])), levels=15, colors=GRAY, linewidths=0.6, alpha=0.7)
    for label, path, c in runs:
        a.plot(path[:, 0], path[:, 1], "-", color=c, lw=1.6, label=label)
    a.plot(0, 0, "X", color=RED, ms=10)
    a.annotate("안장점 (saddle point)\n기울기 0이지만 최솟값이 아님", (0, 0), xytext=(-2.1, -1.45),
               fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    a.plot([0, 0], [-1, 1], "*", color="black", ms=10)
    a.set(xlabel="x", ylabel="y", title=r"$f = x^2 + y^4/4 - y^2/2$: 안장점 (0, 0), 최솟값 (0, ±1)")
    a.set_aspect("equal")
    a.legend(fontsize=8, loc="upper left")

    a = axes[2]
    for label, path, c in runs:
        a.plot([h(p) for p in path], color=c, lw=1.6, label=label)
    a.axhline(0, color=RED, ls=":", lw=1)
    a.text(60, 0.08, "안장점의 높이 0에서 오래 머무는 구간\n= 평탄 구간 (plateau)", color=RED, fontsize=9)
    a.axhline(-0.25, color="black", ls=":", lw=1)
    a.text(2, -0.23, "최솟값 -0.25", fontsize=9)
    a.set(xlabel="걸음 수", ylabel="f", ylim=(-0.3, 1.2),
          title="손실 그래프: 멈춘 것처럼 보이다가 갑자기 떨어진다")
    a.legend(fontsize=8)
    save(fig, "fig10_landscape_problems")


def fig_xor():
    """XOR: 직선 하나로는 못 나누지만, 은닉층이 공간을 바꾸면 직선 하나로 나눌 수 있다."""
    sig = lambda z: 1 / (1 + np.exp(-z))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    W1 = np.array([[20.0, 20.0], [-20.0, -20.0]])  # h1 ≈ OR, h2 ≈ NAND
    b1 = np.array([-10.0, 30.0])
    W2 = np.array([20.0, 20.0])                     # 출력 ≈ AND(h1, h2)
    b2 = -30.0
    H = sig(X @ W1.T + b1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def points(ax, P, title, xl, yl):
        for (px, py), t in zip(P, y):
            ax.scatter(px, py, s=160, marker="o" if t else "X", color=RED if t else BLUE,
                       edgecolor="black", zorder=3)
        ax.set(xlim=(-0.4, 1.4), ylim=(-0.4, 1.4), xlabel=xl, ylabel=yl, title=title)
        ax.set_aspect("equal")

    a = axes[0]
    points(a, X, "입력 공간: 직선 하나로는 못 나눈다", "$x_1$", "$x_2$")
    xs = np.linspace(-0.4, 1.4, 2)
    for k, (m, c) in enumerate([(-1, 0.5), (-1, 1.5), (1, 0.5)]):
        a.plot(xs, m * xs + c, "--", color=GRAY, lw=1, label="어떤 직선을 그어도 실패" if k == 0 else None)
    for (px, py), t in zip(X, y):
        a.annotate(f"({px:.0f},{py:.0f}) → {t}", (px, py), xytext=(8, -14), textcoords="offset points", fontsize=8)
    a.legend(fontsize=8, loc="upper center")

    a = axes[1]
    points(a, H, "은닉층이 바꾼 공간: 직선 하나로 나뉜다", "$h_1$ (OR처럼 동작)", "$h_2$ (NAND처럼 동작)")
    a.plot(xs, 1.5 - xs, "-", color=GREEN, lw=2, label="$h_1 + h_2 = 1.5$")
    a.annotate("(0,1)과 (1,0)이\n같은 점 (1, 1)로 모인다", (1, 1), xytext=(-0.35, 0.45), fontsize=8,
               arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    a.legend(fontsize=8, loc="lower left")

    a = axes[2]
    g = np.linspace(-0.4, 1.4, 300)
    gx, gy = np.meshgrid(g, g)
    G = np.stack([gx.ravel(), gy.ravel()], axis=1)
    out = sig(sig(G @ W1.T + b1) @ W2 + b2).reshape(gx.shape)
    a.contourf(gx, gy, out, levels=[0, 0.5, 1], colors=["#E6F1FB", "#FCEBEB"])
    a.contour(gx, gy, out, levels=[0.5], colors=[GREEN], linewidths=2)
    points(a, X, "2층 신경망 (2 → 2 → 1)의 결정 영역", "$x_1$", "$x_2$")
    fig.suptitle("XOR 문제: 1969년 단층 퍼셉트론의 한계로 지적되었고, 은닉층과 역전파로 풀렸다 "
                 "(빨간 원 = 1, 파란 X = 0)", y=1.03)
    save(fig, "fig00c_xor")


def fig_activations():
    """활성화 함수의 변천: 계단 → sigmoid → tanh → ReLU. 실선은 함수, 점선은 그 기울기(미분)."""
    z = np.linspace(-5, 5, 1001)
    sig = 1 / (1 + np.exp(-z))
    funcs = [
        ("계단 함수 (1958, 퍼셉트론)", (z >= 0).astype(float), np.zeros_like(z),
         "기울기가 모든 곳에서 0\n(0에서는 정의 안 됨)\n→ 경사하강법 불가"),
        ("sigmoid (1980년대)", sig, sig * (1 - sig), "매끄러워서 미분 가능\n기울기 최대 0.25\n→ 층이 깊으면 기울기 소실"),
        ("tanh", np.tanh(z), 1 - np.tanh(z) ** 2, "sigmoid를 -1~1로 늘린 모양\n기울기 최대 1\n양 끝은 여전히 0"),
        ("ReLU (2010년대)", np.maximum(z, 0), (z > 0).astype(float), "양수에서 기울기가 항상 1\n→ 깊은 신경망도 학습\n계산도 가장 간단"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (title, f, d, note) in zip(axes, funcs):
        if "계단" in title:
            ax.plot(z[z < 0], f[z < 0], color=BLUE, lw=2.5, label="함수")
            ax.plot(z[z >= 0], f[z >= 0], color=BLUE, lw=2.5)
            ax.plot([0], [1], "o", color=BLUE)
            ax.plot([0], [0], "o", color=BLUE, mfc="white")
        else:
            ax.plot(z, f, color=BLUE, lw=2.5, label="함수")
        ax.plot(z, d, "--", color=RED, lw=2, label="기울기 (미분)")
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set(xlabel="z (가중합)", title=title, ylim=(-1.2, 2.2))
        ax.text(-4.8, 1.25, note, fontsize=8.5, va="bottom")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("은닉층 활성화 함수의 변천: 실선 = 함수, 빨간 점선 = 기울기 (역전파에 곱해지는 값)", y=1.03)
    save(fig, "fig11b_activations")

if __name__ == "__main__":
    fig_terrain()
    fig_vector_basics()
    fig_trig()
    fig_dot_product()
    fig_e_number()
    fig_mse()
    fig_xor()
    fig_secant_tangent()
    fig_zoom_tangent()
    fig_three_differences()
    fig_h_error()
    fig_gradient_contour()
    fig_linear_transform()
    fig_eigenvectors()
    fig_eigen_uses()
    fig_svd_rank()
    fig_lr_1d()
    fig_landscape_problems()
    fig_network()
    fig_activations()
    fig_sigmoid_bce()
    fig_normal_mse()
    fig_softmax()
