"""미적분 모듈: 중심차분 수치 미분과 Gradient 시각화.

수식 요약
    중심차분:   f'(x) ≈ (f(x + h) - f(x - h)) / (2h)        오차 O(h²)
    편미분:     ∂f/∂xᵢ ≈ (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h)   (eᵢ: i번째 단위벡터)
    Gradient:   ∇f = (∂f/∂x₁, ∂f/∂x₂, ...)  → 가장 가파르게 오르는 방향

실행:
    uv run python src/calculus.py
"""

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_style import BLUE, GRAY, RED, plain_log_ticks, save_figure, setup_style

np.random.seed(42)

DEFAULT_H = 1e-5  # 중심차분에서 절단 오차와 반올림 오차의 합이 가장 작아지는 크기 (약 ε^(1/3))


def numerical_derivative(f, x, h=DEFAULT_H):
    """1변수 함수 f의 x에서의 미분을 중심차분으로 근사한다.

    f'(x) ≈ (f(x + h) - f(x - h)) / (2h)

    Args:
        f: 실수 하나를 받아 실수 하나를 돌려주는 함수.
        x: 미분할 위치.
        h: 양옆으로 움직일 거리. 너무 크면 절단 오차, 너무 작으면 반올림 오차가 커진다.

    Returns:
        x에서의 기울기 근사값 (float).
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, point, h=DEFAULT_H):
    """다변수 함수 f의 point에서의 gradient를 중심차분으로 근사한다.

    변수마다 그 변수만 ±h 움직이고 나머지는 고정해서 편미분을 구한다.
    ∂f/∂xᵢ ≈ (f(p + h·eᵢ) - f(p - h·eᵢ)) / (2h)

    Args:
        f: 길이 n인 배열을 받아 실수 하나를 돌려주는 함수.
        point: gradient를 구할 위치. shape (n,).
        h: 각 변수를 움직일 거리.

    Returns:
        gradient 벡터. shape (n,).
    """
    point = np.asarray(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(point.size):
        step = np.zeros_like(point)
        step[i] = h
        grad[i] = (f(point + step) - f(point - step)) / (2 * h)
    return grad


def bowl(p):
    """과제의 2변수 함수 f(x, y) = x² + y²."""
    return p[0] ** 2 + p[1] ** 2


def check_derivative_at_3(tol=1e-4):
    """f(x) = x² 의 x = 3에서 수치 미분을 해석적 미분값 6과 비교한다.

    Args:
        tol: 허용 오차 (과제 기준 1e-4).

    Returns:
        (수치 미분값, 절대 오차, 통과 여부) 튜플.
    """
    value = numerical_derivative(lambda x: x**2, 3.0)
    error = abs(value - 6.0)
    passed = error <= tol
    print("[수치 미분] f(x) = x², x = 3")
    print(f"  중심차분(h={DEFAULT_H:g}) = {value:.10f}, 해석적 미분 = 6")
    print(f"  오차 = {error:.2e} → {'통과' if passed else '실패'} (기준 {tol:g})")
    return value, error, passed


def plot_h_error(save=True):
    """h 크기에 따른 전방차분과 중심차분의 오차를 로그 그래프로 비교한다.

    f(x) = x² 는 중심차분의 절단 오차가 0이라 비교가 어려우므로,
    f(x) = sin(x), x = 1 (정답 cos 1)을 함께 그린다.

    Args:
        save: True면 outputs/calculus_h_error.png 로 저장한다.

    Returns:
        Matplotlib Figure.
    """
    hs = np.logspace(-14, -1, 60)
    x0 = 1.0
    exact = np.cos(x0)
    forward = np.abs((np.sin(x0 + hs) - np.sin(x0)) / hs - exact)
    central = np.abs((np.sin(x0 + hs) - np.sin(x0 - hs)) / (2 * hs) - exact)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(hs, forward, "o-", ms=3, color=BLUE, label="전방차분 (오차 ∝ h)")
    ax.loglog(hs, central, "o-", ms=3, color=RED, label="중심차분 (오차 ∝ h²)")
    ax.axvline(DEFAULT_H, color=GRAY, ls="--", lw=1, label=f"사용한 h = {DEFAULT_H:g}")
    ax.set(xlabel="h", ylabel="|근사값 - 정답|",
           title="f(x) = sin x, x = 1: h가 너무 크거나 작으면 오차가 커진다")
    plain_log_ticks(ax)
    ax.legend(fontsize=9)
    if save:
        save_figure(fig, "calculus_h_error")
    return fig


def plot_gradient_field(f=bowl, lim=3.0, save=True):
    """f의 등고선과, 등고선 위 여러 점에서의 gradient 화살표를 그린다.

    화살표는 numerical_gradient로 구하고, 방향만 보이도록 길이 1로 정규화한다.
    각 점에서 gradient와 등고선 접선의 내적(cos)을 계산해 수직인지 확인한다.

    Args:
        f: 2변수 함수 (길이 2 배열 → 실수).
        lim: 그림의 x, y 범위 [-lim, lim].
        save: True면 outputs/calculus_gradient_field.png 로 저장한다.

    Returns:
        (Figure, 각 점에서 |cos(gradient, 접선)| 의 최댓값) 튜플.
    """
    g = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(g, g)
    Z = f(np.array([X, Y]))
    levels = [0.5, 1, 2, 4, 6]

    # 등고선 위의 점: 반지름 r = √c 인 원 위에 8개씩
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    points = np.array([[np.sqrt(c) * np.cos(t), np.sqrt(c) * np.sin(t)]
                       for c in (1, 4) for t in angles])
    grads = np.array([numerical_gradient(f, p) for p in points])
    unit = grads / np.linalg.norm(grads, axis=1, keepdims=True)

    # 원 위의 접선 방향은 (-y, x). 내적이 0이면 수직.
    tangents = np.c_[-points[:, 1], points[:, 0]]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    max_cos = np.max(np.abs(np.sum(unit * tangents, axis=1)))

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    cs = ax.contour(X, Y, Z, levels=levels, colors=GRAY, linewidths=1)
    ax.clabel(cs, fontsize=8)
    ax.quiver(points[:, 0], points[:, 1], unit[:, 0], unit[:, 1], color=RED,
              angles="xy", scale_units="xy", scale=2.2, width=0.007)
    ax.plot(points[:, 0], points[:, 1], "o", color=RED, ms=4)
    ax.set_aspect("equal")  # 비율이 다르면 수직인 화살표가 비스듬해 보인다
    ax.set(xlabel="x", ylabel="y",
           title=f"$f(x,y)=x^2+y^2$ 의 gradient (빨강)\n"
                 f"등고선에 수직: 최대 |cos| = {max_cos:.1e}")
    print(f"[Gradient] 등고선 접선과 gradient의 최대 |cos| = {max_cos:.2e} (0이면 수직)")
    if save:
        save_figure(fig, "calculus_gradient_field")
    return fig, max_cos


def main():
    """과제 요구 사항을 차례대로 실행하고 그림을 outputs/에 저장한다."""
    setup_style()
    check_derivative_at_3()
    plot_h_error()
    plot_gradient_field()


if __name__ == "__main__":
    main()
