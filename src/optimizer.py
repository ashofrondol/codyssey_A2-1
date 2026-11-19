"""최적화 모듈: VanillaGD, SGD, Momentum 옵티마이저와 수렴 경로 시각화.

수식 요약
    Vanilla GD:  θ_{t+1} = θ_t - η ∇L(θ_t)                  (전체 데이터 N개로 gradient 계산)
    SGD:         θ_{t+1} = θ_t - η ∇L_B(θ_t)                (무작위로 뽑은 B개(미니배치)로 gradient 추정)
    Momentum:    v_{t+1} = β v_t + ∇f(θ_t),   θ_{t+1} = θ_t - η v_{t+1}
    Adam:        m ← β₁m + (1-β₁)g,  s ← β₂s + (1-β₂)g²,  θ ← θ - η m̂ / (√ŝ + ε)   (보너스)
    Newton:      θ_{t+1} = θ_t - H(θ_t)⁻¹ ∇f(θ_t)            (H: Hessian, 2차 미분 행렬, 보너스)
    f = x² + y²   →  ∇f = (2x, 2y),   한 걸음마다 각 좌표에 (1 - 2η)를 곱한다.
    f = x² + 10y² →  ∇f = (2x, 20y),  x에는 (1 - 2η), y에는 (1 - 20η)를 곱한다.
    곱해지는 수의 절댓값이 1보다 크면 발산, 1이면 제자리 진동, 1보다 작으면 수렴한다.

실행:
    uv run python src/optimizer.py
"""

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

from plot_style import (BLUE, GRAY, GREEN, ORANGE, PINK, PURPLE, RED, plain_log_ticks,
                        save_animation, save_figure, setup_style)

np.random.seed(42)

START = np.array([5.0, 5.0])


# ---------------------------------------------------------------------------
# 테스트 함수
# ---------------------------------------------------------------------------

def bowl(p):
    """2차 함수 f(x, y) = x² + y²."""
    return p[0] ** 2 + p[1] ** 2


def bowl_grad(p):
    """f(x, y) = x² + y² 의 gradient (2x, 2y)."""
    return np.array([2 * p[0], 2 * p[1]])


def ellipse(p):
    """타원형 함수 f(x, y) = x² + 10y² (y 방향이 10배 가파른 길쭉한 골짜기)."""
    return p[0] ** 2 + 10 * p[1] ** 2


def ellipse_grad(p):
    """f(x, y) = x² + 10y² 의 gradient (2x, 20y)."""
    return np.array([2 * p[0], 20 * p[1]])


def ellipse_hess(p):
    """f(x, y) = x² + 10y² 의 Hessian [[2, 0], [0, 20]] (모든 위치에서 같다)."""
    return np.array([[2.0, 0.0], [0.0, 20.0]])


def rosenbrock(p):
    """로젠브록 함수 f(x, y) = (1 - x)² + 100 (y - x²)². 최솟값은 (1, 1)에서 0.

    휘어진 바나나 모양의 좁은 골짜기라서 최적화 알고리즘 비교에 자주 쓰인다.
    """
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def rosenbrock_grad(p):
    """로젠브록 함수의 gradient (-2(1-x) - 400x(y-x²), 200(y-x²))."""
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def rosenbrock_hess(p):
    """로젠브록 함수의 Hessian [[2 - 400(y - 3x²), -400x], [-400x, 200]]."""
    x, y = p
    return np.array([[2 - 400 * (y - 3 * x**2), -400 * x], [-400 * x, 200.0]])


def saddle(p):
    """안장점이 있는 함수 f(x, y) = x² + y⁴/4 - y²/2. 안장점 (0, 0), 최솟값 (0, ±1)에서 -0.25."""
    return p[0] ** 2 + p[1] ** 4 / 4 - p[1] ** 2 / 2


def saddle_grad(p):
    """안장점 함수의 gradient (2x, y³ - y)."""
    return np.array([2 * p[0], p[1] ** 3 - p[1]])


def saddle_hess(p):
    """안장점 함수의 Hessian [[2, 0], [0, 3y² - 1]]. |y| < 1/√3 에서는 음수 고유값이 있다."""
    return np.array([[2.0, 0.0], [0.0, 3 * p[1] ** 2 - 1]])


# ---------------------------------------------------------------------------
# 옵티마이저
# ---------------------------------------------------------------------------

class VanillaGD:
    """기본 경사하강법. gradient의 반대 방향으로 학습률만큼 이동한다.

    θ_{t+1} = θ_t - η ∇f(θ_t)

    Attributes:
        lr: 학습률 η.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.1):
        """학습률을 정한다.

        Args:
            lr: 학습률 η (한 걸음의 크기).
        """
        self.lr = lr
        self.name = f"Vanilla GD (lr={lr})"

    def reset(self):
        """저장한 상태가 없으므로 아무것도 하지 않는다 (Momentum과 같은 사용법을 위해 존재)."""

    def step(self, params, grad):
        """한 걸음 이동한 새 위치를 돌려준다.

        Args:
            params: 현재 위치 θ. shape (n,).
            grad: 현재 위치의 gradient ∇f(θ). shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        return params - self.lr * grad


class Momentum:
    """관성을 가진 경사하강법. 지난 gradient를 속도 v로 누적해서 이동한다.

    v_{t+1} = β v_t + ∇f(θ_t)
    θ_{t+1} = θ_t - η v_{t+1}

    같은 방향의 gradient가 계속되면 속도가 쌓여 빨라지고(최대 약 1/(1-β)배),
    부호가 번갈아 바뀌는 방향(지그재그)은 서로 상쇄되어 진동이 줄어든다.

    Attributes:
        lr: 학습률 η.
        beta: 관성 계수 β (이전 속도를 얼마나 유지할지).
        velocity: 현재 속도 v. 처음에는 None이고 첫 step에서 0으로 만든다.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.1, beta=0.9):
        """학습률과 관성 계수를 정한다.

        Args:
            lr: 학습률 η.
            beta: 관성 계수 β (보통 0.9).
        """
        self.lr = lr
        self.beta = beta
        self.velocity = None
        self.name = f"Momentum (lr={lr}, β={beta})"

    def reset(self):
        """속도를 지워 새 최적화를 처음부터 시작할 수 있게 한다."""
        self.velocity = None

    def step(self, params, grad):
        """속도를 갱신하고 한 걸음 이동한 새 위치를 돌려준다.

        Args:
            params: 현재 위치 θ. shape (n,).
            grad: 현재 위치의 gradient ∇f(θ). shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.beta * self.velocity + grad
        return params - self.lr * self.velocity


class SGD:
    """확률적 경사하강법 (Stochastic Gradient Descent, 미니배치 SGD).

    업데이트 식은 Vanilla GD와 같다:  θ_{t+1} = θ_t - η g_t
    차이는 g_t 를 **전체 데이터가 아니라 무작위로 뽑은 batch_size개(미니배치)** 로 계산한다는 점이다.

        g_t = (1/B) Σ_{i ∈ 미니배치} ∇ℓ_i(θ_t)     (ℓ_i: i번째 데이터 하나의 손실)

    한 걸음의 계산량이 B/N 로 줄어드는 대신, g_t 는 진짜 gradient의 '추정값'이라 잡음이 섞인다.
    그래서 경로가 흔들리고, 학습률이 고정이면 최솟값 근처에서 계속 조금씩 흔들린다.

    Attributes:
        lr: 학습률 η.
        batch_size: 한 걸음에 사용할 데이터 개수 B (1이면 순수 SGD).
        rng: 미니배치를 뽑는 난수 생성기 (seed 고정으로 재현 가능).
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.05, batch_size=16, seed=42):
        """학습률, 미니배치 크기, 난수 시드를 정한다.

        Args:
            lr: 학습률 η.
            batch_size: 미니배치 크기 B.
            seed: 미니배치 샘플링용 시드.
        """
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.name = f"SGD (lr={lr}, batch={batch_size})"

    def reset(self):
        """난수 생성기를 처음 상태로 되돌려 같은 미니배치 순서를 재현한다."""
        self.rng = np.random.RandomState(self.seed)

    def sample_batch(self, n_data):
        """전체 n_data개 중 batch_size개의 인덱스를 중복 없이 무작위로 뽑는다.

        Args:
            n_data: 전체 데이터 개수 N.

        Returns:
            인덱스 배열. shape (batch_size,).
        """
        return self.rng.choice(n_data, size=self.batch_size, replace=False)

    def step(self, params, grad):
        """미니배치로 구한 gradient로 한 걸음 이동한 새 위치를 돌려준다.

        Args:
            params: 현재 위치 θ. shape (n,).
            grad: 미니배치로 계산한 gradient 추정값. shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        return params - self.lr * grad


class Nesterov:
    """Nesterov 가속 경사법 (Nesterov, 1983): Momentum의 "한 걸음 앞을 보고" 고치는 버전.

    원래 식은 '관성으로 갈 위치 θ - ηβv 에서의 gradient' 를 쓴다.
        v_{t+1} = β v_t + ∇f(θ_t - η β v_t),   θ_{t+1} = θ_t - η v_{t+1}
    여기서는 현재 위치의 gradient만으로 같은 효과를 내는, 딥러닝 라이브러리(PyTorch 등)에서
    쓰는 동등한 형태로 구현한다.
        v_{t+1} = β v_t + g_t,   θ_{t+1} = θ_t - η (g_t + β v_{t+1})

    Attributes:
        lr: 학습률 η.
        beta: 관성 계수 β.
        velocity: 속도 v.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.01, beta=0.9):
        """학습률과 관성 계수를 정한다.

        Args:
            lr: 학습률 η.
            beta: 관성 계수 β.
        """
        self.lr, self.beta = lr, beta
        self.velocity = None
        self.name = f"Nesterov (lr={lr}, β={beta})"

    def reset(self):
        """속도를 지운다."""
        self.velocity = None

    def step(self, params, grad):
        """속도를 갱신하고, 속도 방향으로 한 번 더 내다본 만큼 이동한다.

        Args:
            params: 현재 위치. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.beta * self.velocity + grad
        return params - self.lr * (grad + self.beta * self.velocity)


class AdaGrad:
    """AdaGrad (Duchi, Hazan, Singer, 2011): 방향마다 지금까지의 gradient 제곱을 모두 더해 나눈다.

    G_t = G_{t-1} + g_t²              (처음부터 지금까지의 합, 성분별)
    θ_{t+1} = θ_t - η g_t / (√G_t + ε)

    gradient가 컸던 방향은 보폭이 줄고, 작았던 방향은 상대적으로 커진다.
    단점: G가 계속 커지기만 해서, 오래 학습하면 보폭이 0에 가까워져 멈춘다.

    Attributes:
        lr: 학습률 η.
        eps: 0으로 나누기를 막는 작은 수.
        G: gradient 제곱의 누적합.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.5, eps=1e-8):
        """학습률을 정한다.

        Args:
            lr: 학습률 η.
            eps: 분모 안정화용 작은 수.
        """
        self.lr, self.eps = lr, eps
        self.G = None
        self.name = f"AdaGrad (lr={lr})"

    def reset(self):
        """누적합을 지운다."""
        self.G = None

    def step(self, params, grad):
        """누적합을 갱신하고 방향별로 나눈 보폭만큼 이동한다.

        Args:
            params: 현재 위치. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G = self.G + grad**2
        return params - self.lr * grad / (np.sqrt(self.G) + self.eps)


class RMSProp:
    """RMSProp (Tieleman & Hinton, 2012, 강의 자료로 발표): AdaGrad의 누적합을 이동 평균으로 바꾼다.

    s_t = ρ s_{t-1} + (1 - ρ) g_t²     (최근 gradient 제곱의 평균, 성분별)
    θ_{t+1} = θ_t - η g_t / (√s_t + ε)

    오래된 gradient는 점점 잊으므로, AdaGrad처럼 보폭이 0으로 줄어들지 않는다.
    Adam은 여기에 Momentum(1차 모멘트)과 치우침 보정을 더한 것이다.

    Attributes:
        lr: 학습률 η.
        rho: 이동 평균 계수 ρ (보통 0.9).
        eps: 0으로 나누기를 막는 작은 수.
        s: gradient 제곱의 이동 평균.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        """하이퍼파라미터를 정한다.

        Args:
            lr: 학습률 η.
            rho: 이동 평균 계수 ρ.
            eps: 분모 안정화용 작은 수.
        """
        self.lr, self.rho, self.eps = lr, rho, eps
        self.s = None
        self.name = f"RMSProp (lr={lr})"

    def reset(self):
        """이동 평균을 지운다."""
        self.s = None

    def step(self, params, grad):
        """이동 평균을 갱신하고 방향별로 나눈 보폭만큼 이동한다.

        Args:
            params: 현재 위치. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        if self.s is None:
            self.s = np.zeros_like(params)
        self.s = self.rho * self.s + (1 - self.rho) * grad**2
        return params - self.lr * grad / (np.sqrt(self.s) + self.eps)


class Adam:
    """Adam 옵티마이저 (보너스). Momentum과 RMSProp을 결합한다.

    m_t = β₁ m_{t-1} + (1 - β₁) g_t           (1차 모멘트: gradient의 이동 평균 = 관성)
    s_t = β₂ s_{t-1} + (1 - β₂) g_t²          (2차 모멘트: gradient 제곱의 이동 평균 = 방향별 크기)
    m̂_t = m_t / (1 - β₁ᵗ),  ŝ_t = s_t / (1 - β₂ᵗ)   (처음에 0에서 시작한 치우침 보정)
    θ_{t+1} = θ_t - η m̂_t / (√ŝ_t + ε)

    방향마다 gradient 크기 √ŝ 로 나누므로, 가파른 방향은 보폭이 줄고 완만한 방향은 보폭이
    커진다(적응적 학습률). 그래서 길쭉한 골짜기에서도 방향별 속도가 비슷해진다.

    Attributes:
        lr: 학습률 η.
        beta1, beta2: 1차, 2차 모멘트의 관성 계수.
        eps: 0으로 나누기를 막는 작은 수.
        m, s, t: 1차 모멘트, 2차 모멘트, 걸음 수.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        """하이퍼파라미터를 정한다.

        Args:
            lr: 학습률 η.
            beta1: 1차 모멘트 계수 β₁ (보통 0.9).
            beta2: 2차 모멘트 계수 β₂ (보통 0.999).
            eps: 분모 안정화용 작은 수 ε.
        """
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.s = None
        self.t = 0
        self.name = f"Adam (lr={lr})"

    def reset(self):
        """모멘트와 걸음 수를 지워 처음부터 시작한다."""
        self.m = self.s = None
        self.t = 0

    def step(self, params, grad):
        """모멘트를 갱신하고 방향별로 조절된 보폭으로 한 걸음 이동한다.

        Args:
            params: 현재 위치 θ. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.s = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        s_hat = self.s / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


class Newton:
    """Newton Method (보너스). 2차 미분(Hessian)으로 휘어진 정도까지 고려해 이동한다.

    θ_{t+1} = θ_t - H(θ_t)⁻¹ ∇f(θ_t)

    현재 위치에서 함수를 2차 함수(포물면)로 근사하고, 그 포물면의 바닥으로 바로 이동한다.
    그래서 2차 함수에서는 한 걸음에 최솟값에 도착하고, 최솟값 근처에서는 매우 빠르게 수렴한다.
    단점: 파라미터가 n개면 n×n Hessian을 구하고 연립방정식을 풀어야 해서(약 n³ 계산)
    수백만 개 파라미터를 가진 신경망에는 쓰기 어렵다. 또 Hessian이 양의 정부호가 아니면
    (안장점, 꼭대기 근처) 오르막이나 안장점 쪽으로 갈 수 있다.

    Attributes:
        damping: H에 더하는 작은 값 (H + damping·I). 0이면 순수 Newton.
        needs_hessian: optimize()가 Hessian을 넘겨주도록 알리는 표시.
        name: 그래프 범례에 쓸 이름.
    """

    needs_hessian = True

    def __init__(self, damping=0.0):
        """감쇠 값을 정한다.

        Args:
            damping: Hessian 대각선에 더할 값. 역행렬 계산을 안정적으로 만든다.
        """
        self.damping = damping
        self.name = "Newton"

    def reset(self):
        """저장한 상태가 없으므로 아무것도 하지 않는다."""

    def step(self, params, grad, hess):
        """H d = ∇f 를 풀어 d만큼 이동한다 (역행렬을 직접 구하지 않고 연립방정식을 푼다).

        Args:
            params: 현재 위치 θ. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).
            hess: 현재 위치의 Hessian. shape (n, n).

        Returns:
            새 위치. shape (n,).
        """
        H = hess + self.damping * np.eye(len(params))
        return params - np.linalg.solve(H, grad)


class Lookahead:
    """Lookahead (Zhang et al., 2019): 다른 옵티마이저를 감싸서 "k걸음 앞으로, 1걸음 뒤로" 움직인다.

    빠른 가중치(fast): 안쪽 옵티마이저(inner)로 매 걸음 이동한다.
    느린 가중치(slow): k걸음마다 빠른 가중치 쪽으로 α만큼만 따라간다.
        slow ← slow + α (fast - slow),  그리고 fast ← slow 로 되돌린다.
    안쪽 옵티마이저가 흔들려도 느린 가중치는 그 평균 쪽으로 움직여 경로가 안정된다.

    Attributes:
        inner: 감쌀 옵티마이저 (VanillaGD, Momentum, Adam 등).
        k: 느린 가중치를 갱신하는 주기.
        alpha: 느린 가중치가 따라가는 비율 (0~1).
        slow: 느린 가중치. 첫 step에서 시작 위치로 정한다.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, inner, k=5, alpha=0.5):
        """감쌀 옵티마이저와 k, α를 정한다.

        Args:
            inner: 안쪽 옵티마이저.
            k: 느린 가중치 갱신 주기.
            alpha: 따라가는 비율.
        """
        self.inner, self.k, self.alpha = inner, k, alpha
        self.slow = None
        self.count = 0
        self.name = f"Lookahead({inner.name}, k={k}, α={alpha})"

    def reset(self):
        """안쪽 옵티마이저와 느린 가중치를 처음 상태로 되돌린다."""
        self.inner.reset()
        self.slow = None
        self.count = 0

    def step(self, params, grad):
        """안쪽 옵티마이저로 한 걸음 가고, k걸음마다 느린 가중치로 되돌아온다.

        Args:
            params: 현재 (빠른) 위치. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).

        Returns:
            새 위치. shape (n,).
        """
        if self.slow is None:
            self.slow = np.array(params, dtype=float)
        fast = self.inner.step(params, grad)
        self.count += 1
        if self.count % self.k == 0:
            self.slow = self.slow + self.alpha * (fast - self.slow)
            fast = self.slow.copy()
        return fast


class SwitchOptimizer:
    """학습 도중에 옵티마이저를 바꾼다 (SWATS, Adam→L-BFGS 같은 '순서대로 쓰기'의 단순한 형태).

    처음 switch_at걸음은 first로, 그 뒤로는 second로 이동한다.

    Attributes:
        first, second: 앞 단계, 뒤 단계 옵티마이저.
        switch_at: 바꾸는 걸음 번호.
        t: 지금까지의 걸음 수.
        name: 그래프 범례에 쓸 이름.
    """

    def __init__(self, first, second, switch_at):
        """두 옵티마이저와 바꿀 시점을 정한다.

        Args:
            first: 처음에 쓸 옵티마이저.
            second: switch_at걸음 이후에 쓸 옵티마이저.
            switch_at: 바꾸는 걸음 번호.
        """
        self.first, self.second, self.switch_at = first, second, switch_at
        self.t = 0
        self.name = f"{first.name.split(' (')[0]} {switch_at}걸음 → {second.name.split(' (')[0]}"

    @property
    def needs_hessian(self):
        """지금 단계의 옵티마이저가 Hessian을 필요로 하는지 (optimize()가 확인한다)."""
        current = self.first if self.t < self.switch_at else self.second
        return getattr(current, "needs_hessian", False)

    def reset(self):
        """두 옵티마이저와 걸음 수를 처음 상태로 되돌린다."""
        self.first.reset()
        self.second.reset()
        self.t = 0

    def step(self, params, grad, hess=None):
        """지금 단계의 옵티마이저로 한 걸음 이동한다.

        Args:
            params: 현재 위치. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).
            hess: Hessian (뒤 단계가 Newton 계열일 때만 필요).

        Returns:
            새 위치. shape (n,).
        """
        current = self.first if self.t < self.switch_at else self.second
        self.t += 1
        if getattr(current, "needs_hessian", False):
            return current.step(params, grad, hess)
        return current.step(params, grad)


class LevenbergMarquardt:
    """Levenberg–Marquardt 방식의 감쇠 Newton: GD와 Newton 사이를 자동으로 오간다.

    d = (H + λI)⁻¹ ∇f
        λ가 작으면 d ≈ H⁻¹∇f        → Newton처럼 크게 이동
        λ가 크면  d ≈ ∇f / λ         → 학습률 1/λ 인 GD처럼 조심스럽게 이동
    이동한 뒤 손실이 줄면 그 걸음을 받아들이고 λ를 줄인다(Newton 쪽으로).
    손실이 늘면 그 걸음을 버리고 제자리에서 λ를 키운다(GD 쪽으로).

    원래 Levenberg–Marquardt는 최소제곱 문제에서 Gauss–Newton과 GD를 섞는 방법이다.
    여기서는 같은 아이디어를 일반 Hessian에 적용한다.

    Attributes:
        f: 손실 함수 (걸음을 받아들일지 판단하는 데 쓴다).
        lam: 현재 감쇠값 λ.
        factor: λ를 줄이거나 키우는 배율.
        lam_history: 매 걸음의 λ 기록 (그래프용).
        needs_hessian: optimize()가 Hessian을 넘겨주도록 알리는 표시.
        name: 그래프 범례에 쓸 이름.
    """

    needs_hessian = True

    def __init__(self, f, lam=10.0, factor=10.0):
        """손실 함수와 감쇠 설정을 정한다.

        Args:
            f: 손실 함수.
            lam: 처음 감쇠값 λ (크게 시작하면 처음에는 GD처럼 조심스럽다).
            factor: λ 조절 배율.
        """
        self.f, self.lam0, self.factor = f, lam, factor
        self.lam = lam
        self.lam_history = []
        self.name = "Levenberg–Marquardt 방식 Newton"

    def reset(self):
        """λ와 기록을 처음 상태로 되돌린다."""
        self.lam = self.lam0
        self.lam_history = []

    def step(self, params, grad, hess):
        """감쇠 Newton 걸음을 시도하고, 손실이 줄면 받아들인다.

        Args:
            params: 현재 위치. shape (n,).
            grad: 현재 위치의 gradient. shape (n,).
            hess: 현재 위치의 Hessian. shape (n, n).

        Returns:
            새 위치 (거절되면 현재 위치 그대로). shape (n,).
        """
        self.lam_history.append(self.lam)
        d = np.linalg.solve(hess + self.lam * np.eye(len(params)), grad)
        candidate = params - d
        if self.f(candidate) < self.f(params):
            self.lam = max(self.lam / self.factor, 1e-12)
            return candidate
        self.lam = min(self.lam * self.factor, 1e12)
        return params.copy()


def optimize(grad_fn, start, optimizer, n_iters=100, hess_fn=None):
    """옵티마이저로 n_iters번 이동하며 지나간 모든 위치를 기록한다.

    Args:
        grad_fn: 위치를 받아 gradient를 돌려주는 함수.
        start: 시작 위치. shape (n,).
        optimizer: step(params, grad)와 reset()을 가진 객체.
        n_iters: 반복 횟수.
        hess_fn: 위치를 받아 Hessian을 돌려주는 함수. Newton에만 필요하다.

    Returns:
        경로 배열. shape (n_iters + 1, n). 첫 행이 시작점.
    """
    optimizer.reset()
    p = np.asarray(start, dtype=float).copy()
    path = [p.copy()]
    for _ in range(n_iters):
        if getattr(optimizer, "needs_hessian", False):
            p = optimizer.step(p, grad_fn(p), hess_fn(p))
        else:
            p = optimizer.step(p, grad_fn(p))
        path.append(p.copy())
    return np.array(path)


def iterations_to_reach(path, tol=1e-3, target=(0.0, 0.0)):
    """경로가 원점에서 반경 tol 안에 처음 들어간 반복 번호를 찾는다.

    Args:
        path: optimize가 돌려준 경로.
        tol: 반경.
        target: 최솟값의 위치 (기본: 원점).

    Returns:
        처음 도달한 반복 번호. 끝까지 도달하지 못하면 None.
    """
    dist = np.linalg.norm(path - np.asarray(target), axis=1)
    hit = np.nonzero(dist < tol)[0]
    return int(hit[0]) if hit.size else None


def _fmt_iters(n):
    """도달 반복 수를 출력용 문자열로 바꾼다 (None이면 '도달 못 함')."""
    return "도달 못 함" if n is None else f"{n}회"


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_paths(f, paths, title, ax=None, lim=6.0):
    """등고선 위에 여러 최적화 경로를 점선으로 겹쳐 그린다.

    Args:
        f: 2변수 함수 (등고선용).
        paths: {범례 이름: (경로 배열, 색)} 딕셔너리.
        title: 그림 제목.
        ax: 그릴 Axes. None이면 새로 만든다.
        lim: x, y 범위 [-lim, lim].

    Returns:
        그림을 그린 Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.6))
    g = np.linspace(-lim, lim, 300)
    X, Y = np.meshgrid(g, g)
    Z = f(np.array([X, Y]))
    ax.contour(X, Y, Z, levels=np.geomspace(0.05, Z.max(), 14), colors=GRAY,
               linewidths=0.7, alpha=0.7)
    for label, (path, color) in paths.items():
        ax.plot(path[:, 0], path[:, 1], "--", color=color, lw=1.6, label=label)
        ax.plot(path[:, 0], path[:, 1], "o", color=color, ms=2.5)
    ax.plot(*START, "ks", ms=6)
    ax.plot(0, 0, "k*", ms=12)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="x", ylabel="y", title=title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def plot_distance(paths, title, ax):
    """반복 횟수에 따른 원점까지의 거리를 로그 스케일로 그린다.

    Args:
        paths: {범례 이름: (경로 배열, 색)} 딕셔너리.
        title: 그림 제목.
        ax: 그릴 Axes.
    """
    for label, (path, color) in paths.items():
        dist = np.maximum(np.linalg.norm(path, axis=1), 1e-16)
        ax.semilogy(dist, color=color, label=label)
    plain_log_ticks(ax, "y")
    ax.set(xlabel="반복 횟수", ylabel="원점까지 거리 (로그)", title=title)
    ax.legend(fontsize=8)


def plot_path_panels(paths, draw_background, axes, n_show=None):
    """방법마다 칸을 나눠, 같은 배경 위에 경로를 하나씩 그린다 (small multiples).

    경로가 같은 곳을 지나면 한 그림에 겹쳐 그릴 때 서로 가려지므로,
    같은 축 범위의 칸을 나란히 두어 위치를 비교한다.

    Args:
        paths: {이름: (경로 배열, 색)} 딕셔너리.
        draw_background: Axes를 받아 등고선 등 배경을 그리는 함수.
        axes: 경로 개수만큼의 Axes 목록.
        n_show: 앞에서부터 그릴 걸음 수. None이면 전부.
    """
    for ax, (label, (path, color)) in zip(axes, paths.items()):
        draw_background(ax)
        shown = path if n_show is None else path[: n_show + 1]
        ax.plot(shown[:, 0], shown[:, 1], "-", color=color, lw=1.4)
        ax.plot(shown[:, 0], shown[:, 1], "o", color=color, ms=2)
        ax.plot(*shown[-1], "o", color=color, ms=7, mec="white", mew=1)
        ax.set_title(label, fontsize=10, color=color)


def contour_background(f, xlim, ylim, levels, start=None, target=(0.0, 0.0), aspect=True,
                       xlabel="x", ylabel="y"):
    """등고선, 시작점(■), 최솟값(★)을 그리는 배경 함수를 만들어 돌려준다.

    Args:
        f: 2변수 함수.
        xlim, ylim: 축 범위.
        levels: 등고선 높이 목록.
        start: 시작점 (없으면 None).
        target: 최솟값 위치.
        aspect: True면 x, y 눈금 비율을 같게 한다.
        xlabel, ylabel: 축 이름.

    Returns:
        Axes를 받아 배경을 그리는 함수.
    """
    gx, gy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    Z = f(np.array([gx, gy]))

    def draw(ax):
        ax.contour(gx, gy, Z, levels=levels, colors=GRAY, linewidths=0.6, alpha=0.7)
        if start is not None:
            ax.plot(*start, "ks", ms=5)
        ax.plot(*target, "k*", ms=11)
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        if aspect:
            ax.set_aspect("equal")

    return draw


# ---------------------------------------------------------------------------
# 실험
# ---------------------------------------------------------------------------

def experiment_vanilla_bowl(radius=1.0, spec_radius=0.1, lr=0.1, n_iters=100):
    """x² + y² 에 Vanilla GD (lr=0.1, 100회)를 적용해 원점 근처에 도달하는지 확인한다.

    판정 반경은 인자로 바꿀 수 있다. 평가 기준은 반경 1, 과제 명세 PDF는 반경 0.1이라
    두 기준을 모두 확인해 출력한다.

    Args:
        radius: 평가 기준 반경 (기본 1.0).
        spec_radius: 과제 명세의 반경 (기본 0.1). None이면 확인하지 않는다.
        lr: 학습률 η.
        n_iters: 반복 횟수.

    Returns:
        경로. shape (n_iters + 1, 2).
    """
    path = optimize(bowl_grad, START, VanillaGD(lr), n_iters)
    final = path[-1]
    dist = np.linalg.norm(final)
    print(f"[Vanilla GD] f = x² + y², 시작 (5, 5), lr = {lr}, {n_iters}회")
    print(f"  최종 위치 = ({final[0]:.2e}, {final[1]:.2e}), 원점까지 거리 = {dist:.2e}")
    checks = [("평가 기준", radius)] + ([("과제 명세", spec_radius)] if spec_radius is not None else [])
    for name, r in checks:
        print(f"  반경 {r} 이내 ({name}): {'통과' if dist < r else '실패'}")
    fig, ax = plt.subplots(figsize=(6, 5.6))
    plot_paths(bowl, {"Vanilla GD (lr=0.1)": (path, BLUE)},
               f"$x^2+y^2$, lr = 0.1, 100회: 최종 거리 {dist:.1e}", ax=ax)
    save_figure(fig, "optimizer_vanilla_bowl")
    return path


def experiment_learning_rates():
    """x² + y² 에서 학습률을 바꿔 수렴, 진동, 발산을 비교한다.

    한 걸음마다 좌표에 (1 - 2·lr)이 곱해지므로
    lr = 0.5 → 0 (한 번에 원점), lr = 1.0 → -1 (제자리 진동), lr > 1 → 발산.
    """
    lrs = [(0.1, BLUE), (0.5, GREEN), (0.9, ORANGE), (1.0, GRAY), (1.05, RED)]
    print("[학습률 비교] f = x² + y², 시작 (5, 5), 20회")
    paths = {}
    for lr, c in lrs:
        path = optimize(bowl_grad, START, VanillaGD(lr), 20)
        paths[f"lr = {lr} (곱해지는 수 {1 - 2 * lr:+.1f})"] = (path, c)
        print(f"  lr = {lr:<4}: 곱해지는 수 1 - 2·lr = {1 - 2 * lr:+.2f}, "
              f"20회 후 거리 = {np.linalg.norm(path[-1]):.3e}")
    # x² + y² 는 둥근 그릇이라 모든 경로가 시작점과 원점을 잇는 직선 위에 있다.
    # 그래서 학습률마다 칸을 나눠, 원점을 사이에 두고 어느 쪽에 착지하는지 보이게 한다.
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, len(lrs), height_ratios=[1, 0.85])
    for j, (label, (path, c)) in enumerate(paths.items()):
        ax = fig.add_subplot(gs[0, j])
        lim = max(6.0, np.abs(path).max() * 1.1)
        plot_paths(bowl, {label: (path, c)}, label.replace(" (", "\n("), ax=ax, lim=lim)
        ax.get_legend().remove()
    ax = fig.add_subplot(gs[1, :])
    plot_distance(paths, "원점까지 거리: |곱해지는 수| < 1 이면 수렴, = 1 이면 제자리, > 1 이면 발산", ax)
    ax.set_ylim(1e-17, 1e3)
    ax.text(1.2, 3e-16, "lr = 0.5: 한 걸음 만에 원점 도착 (거리 0을 1e-16으로 표시)",
            color=GREEN, fontsize=9)
    ax.text(14, 0.6, "lr = 0.1과 0.9는 |곱해지는 수|가 0.8로 같아 거리 선이 겹친다\n(0.9는 원점을 넘나들며 줄어든다)",
            color=ORANGE, fontsize=9)
    fig.suptitle("$f(x,y)=x^2+y^2$, 시작 (5, 5), 20회: 학습률에 따른 경로", y=1.0)
    save_figure(fig, "optimizer_learning_rates")
    return paths


def experiment_momentum_bowl():
    """x² + y² 에서 Vanilla GD와 Momentum(β=0.9)의 경로를 한 Figure에 겹쳐 비교한다."""
    pv = optimize(bowl_grad, START, VanillaGD(0.1), 100)
    pm = optimize(bowl_grad, START, Momentum(0.1, 0.9), 100)
    print("[Vanilla vs Momentum] f = x² + y², 시작 (5, 5), lr = 0.1, 100회")
    for name, p in [("Vanilla GD", pv), ("Momentum", pm)]:
        print(f"  {name:<10}: 최종 ({p[-1][0]:+.2e}, {p[-1][1]:+.2e}), "
              f"거리 {np.linalg.norm(p[-1]):.2e}, 반경 1e-3 도달: {_fmt_iters(iterations_to_reach(p))}")
    paths = {"Vanilla GD (lr=0.1)": (pv, BLUE), "Momentum (lr=0.1, β=0.9)": (pm, RED)}
    # 둥근 그릇에서는 두 경로가 모두 (5,5)-(0,0) 대각선 위에 있어 한 그림에서 겹친다.
    # 그래서 경로를 칸으로 나누고, 걸음에 따른 x 좌표와 거리를 따로 그린다.
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    bg = contour_background(bowl, (-6, 6), (-6, 6), np.geomspace(0.05, 72, 14), start=START)
    plot_path_panels(paths, bg, axes[:2])
    ax = axes[2]
    for label, (path, c) in paths.items():
        ax.plot(path[:, 0], "-o", ms=2, color=c, label=label)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set(xlabel="걸음 수", ylabel="x 좌표 (y도 같다)",
           title="x 좌표: Momentum은 0을 지나쳤다 돌아온다")
    ax.legend(fontsize=8)
    plot_distance(paths, "원점까지 거리: 둥근 그릇에서는 Vanilla가 빠르다", axes[3])
    fig.suptitle("$x^2+y^2$, 시작 (5, 5), lr = 0.1, 100회: Vanilla GD vs Momentum", y=1.03)
    save_figure(fig, "optimizer_momentum_bowl")
    return pv, pm


def experiment_ellipse():
    """x² + 10y² 에서 여러 학습률로 Vanilla GD와 Momentum을 비교한다.

    - lr = 0.01: 작은 학습률. Vanilla는 완만한 x 방향이 매우 느리다 (x에 0.98씩 곱함).
    - lr = 0.05: 과제 예시 값. y 방향에 곱해지는 수가 1 - 20·0.05 = 0 이라 y가 한 번에 0이 된다.
    - lr = 0.09: y 방향에 -0.8이 곱해져 Vanilla가 지그재그로 움직인다.
    """
    print("[타원형 함수] f = x² + 10y², 시작 (5, 5), β = 0.9, 반경 1e-3 도달 반복 수")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for j, lr in enumerate([0.01, 0.05, 0.09]):
        pv = optimize(ellipse_grad, START, VanillaGD(lr), 2000)
        pm = optimize(ellipse_grad, START, Momentum(lr, 0.9), 2000)
        nv, nm = iterations_to_reach(pv), iterations_to_reach(pm)
        print(f"  lr = {lr}: x 곱 {1 - 2 * lr:+.2f}, y 곱 {1 - 20 * lr:+.2f} | "
              f"Vanilla {nv}회, Momentum {nm}회 → "
              f"{'Momentum이 빠름' if nm < nv else 'Vanilla가 빠름'}")
        label_v, label_m = f"Vanilla GD ({nv}회)", f"Momentum ({nm}회)"
        plot_paths(ellipse, {label_v: (pv[:101], BLUE), label_m: (pm[:101], RED)},
                   f"lr = {lr}: 처음 100걸음", ax=axes[0, j])
        plot_distance({label_v: (pv, BLUE), label_m: (pm, RED)},
                      f"lr = {lr}: 원점까지 거리", axes[1, j])
        axes[1, j].set_xlim(0, max(nv or 2000, nm or 2000) * 1.1)
    fig.suptitle("$f(x,y)=x^2+10y^2$: 학습률이 작을수록 Momentum의 이점이 커진다", y=1.0)
    save_figure(fig, "optimizer_ellipse")


# ---------------------------------------------------------------------------
# GD vs SGD vs Momentum: 데이터가 있는 문제 (선형 회귀)
# ---------------------------------------------------------------------------

def make_regression_data(n=200, seed=42):
    """y = 2x + 1 + 잡음(σ=0.5) 데이터를 만든다. x는 0~4 사이에서 고르게 뽑는다.

    x가 0을 중심으로 하지 않아서 기울기 w와 절편 b가 서로 얽히고,
    손실 지형이 길쭉한 골짜기가 된다 (Momentum의 효과를 보기 좋다).

    Args:
        n: 데이터 개수 N.
        seed: 난수 시드.

    Returns:
        (X, y) 튜플. 각각 shape (n,).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 4, n)
    y = 2 * X + 1 + rng.normal(0, 0.5, n)
    return X, y


def mse_loss(theta, X, y):
    """선형 모델 ŷ = w·x + b 의 MSE 손실 L(w, b) = (1/N) Σ (w·xᵢ + b - yᵢ)².

    Args:
        theta: 파라미터 (w, b). shape (2,).
        X, y: 데이터. shape (N,).

    Returns:
        손실 (float).
    """
    w, b = theta
    return np.mean((w * X + b - y) ** 2)


def mse_grad(theta, X, y):
    """MSE 손실의 gradient (∂L/∂w, ∂L/∂b) = (2/N) Σ (w·xᵢ + b - yᵢ)·(xᵢ, 1).

    X, y에 미니배치만 넣으면 그 미니배치로 추정한 gradient가 된다.

    Args:
        theta: 파라미터 (w, b). shape (2,).
        X, y: 데이터 (전체 또는 미니배치). shape (B,).

    Returns:
        gradient. shape (2,).
    """
    w, b = theta
    err = w * X + b - y
    return np.array([2 * np.mean(err * X), 2 * np.mean(err)])


def train(optimizer, X, y, start, n_steps=300):
    """옵티마이저로 선형 회귀를 학습하며 경로와 사용한 데이터 개수를 기록한다.

    SGD는 매 걸음 미니배치를 뽑아 gradient를 추정하고, 나머지는 전체 데이터를 쓴다.

    Args:
        optimizer: VanillaGD, SGD, Momentum, Adam, Newton 중 하나.
        X, y: 전체 데이터.
        start: 시작 파라미터 (w, b).
        n_steps: 걸음 수.

    Returns:
        (경로 shape (n_steps+1, 2), 누적 사용 데이터 개수 shape (n_steps+1,)) 튜플.
    """
    optimizer.reset()
    theta = np.asarray(start, dtype=float).copy()
    path, used = [theta.copy()], [0]
    for _ in range(n_steps):
        if isinstance(optimizer, SGD):
            idx = optimizer.sample_batch(len(X))
            g = mse_grad(theta, X[idx], y[idx])
            used.append(used[-1] + len(idx))
        else:
            g = mse_grad(theta, X, y)
            used.append(used[-1] + len(X))
        if getattr(optimizer, "needs_hessian", False):
            # MSE의 Hessian (2/N) Σ [[x², x], [x, 1]] 은 θ와 무관한 상수 행렬이다.
            H = 2 * np.array([[np.mean(X**2), np.mean(X)], [np.mean(X), 1.0]])
            theta = optimizer.step(theta, g, H)
        else:
            theta = optimizer.step(theta, g)
        path.append(theta.copy())
    return np.array(path), np.array(used)


def experiment_gd_sgd_momentum(n_steps=300):
    """선형 회귀에서 GD, SGD(미니배치 16, 1), Momentum, Adam, Newton을 비교한다.

    - 왼쪽: 손실 지형 (w, b) 위의 경로
    - 가운데: 걸음 수에 따른 초과 손실 L(θ) - L(θ*) (θ*: 최소제곱 해)
    - 오른쪽: 사용한 데이터 개수(계산량)에 따른 초과 손실

    Returns:
        {이름: (경로, 사용 데이터, 초과 손실)} 딕셔너리.
    """
    X, y = make_regression_data()
    A = np.c_[X, np.ones_like(X)]
    best = np.linalg.solve(A.T @ A, A.T @ y)  # 정답 확인용 최소제곱 해 (정규방정식)
    best_loss = mse_loss(best, X, y)
    start = np.array([0.0, 4.0])
    runs = [
        (VanillaGD(0.05), BLUE),
        (SGD(0.05, batch_size=16), GREEN),
        (SGD(0.02, batch_size=1), ORANGE),
        (Momentum(0.05, 0.9), RED),
        (Adam(0.1), PURPLE),
        (Newton(), GRAY),
    ]
    print(f"[GD vs SGD vs Momentum] 선형 회귀 y = 2x + 1 + 잡음, N = {len(X)}, 시작 (w, b) = (0, 4)")
    print(f"  최소제곱 해 (w*, b*) = ({best[0]:.4f}, {best[1]:.4f}), 최소 손실 = {best_loss:.4f}")
    results = {}
    for opt, color in runs:
        path, used = train(opt, X, y, start, n_steps)
        excess = np.array([mse_loss(p, X, y) for p in path]) - best_loss
        hit = np.nonzero(excess < 1e-2)[0]
        steps = int(hit[0]) if hit.size else None
        print(f"  {opt.name:<28}: 초과 손실 < 0.01 도달 {_fmt_iters(steps):>8}"
              f" (사용 데이터 {used[steps] if steps is not None else '-':>6}개), "
              f"{n_steps}걸음 후 초과 손실 {max(excess[-1], 0):.1e}, (w, b) = ({path[-1][0]:.3f}, {path[-1][1]:.3f})")
        results[opt.name] = (path, used, excess, color)

    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15])
    W, B = np.meshgrid(np.linspace(-0.5, 3.5, 300), np.linspace(-1.5, 4.5, 300))
    L = np.mean((W[..., None] * X + B[..., None] - y) ** 2, axis=-1)
    levels = np.geomspace(best_loss * 1.05, L.max(), 16)

    def regression_bg(ax):
        ax.contour(W, B, L, levels=levels, colors=GRAY, linewidths=0.6, alpha=0.7)
        ax.plot(*start, "ks", ms=5)
        ax.plot(*best, "k*", ms=11)
        ax.set(xlabel="기울기 w", ylabel="절편 b")

    plot_path_panels({name: (path, c) for name, (path, _, _, c) in results.items()},
                     regression_bg, [fig.add_subplot(gs[0, k]) for k in range(6)])
    axes = [None, fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])]
    for ax, key, xlabel, title in [
        (axes[1], "step", "걸음 수", "걸음 수 기준: Newton(1걸음), Adam, Momentum 순"),
        (axes[2], "used", "사용한 데이터 개수 (계산량, 로그)", "계산량 기준: SGD가 가장 적은 계산"),
    ]:
        for name, (path, used, excess, c) in results.items():
            xs = np.arange(len(excess)) if key == "step" else np.maximum(used, 1)
            ax.semilogy(xs, np.maximum(excess, 1e-16), color=c, lw=1.2, label=name)
        if key == "step":
            ax.set_xscale("symlog", linthresh=10)
        if key == "used":
            ax.set_xscale("log")
            plain_log_ticks(ax, "x")
        plain_log_ticks(ax, "y")
        ax.axhline(1e-2, color=GRAY, ls=":", lw=1)
        ax.set(xlabel=xlabel, ylabel=r"초과 손실 $L(\theta) - L(\theta^*)$", title=title)
        ax.legend(fontsize=8)
    fig.suptitle("선형 회귀에서 GD, SGD, Momentum, Adam, Newton 비교 (300걸음, 위: 방법별 경로)", y=1.01)
    save_figure(fig, "optimizer_gd_sgd_momentum")
    return results


def experiment_bonus_adam_newton():
    """(보너스) Vanilla GD, Momentum, Adam, Newton의 수렴 속도를 두 함수에서 비교한다.

    - x² + 10y² (시작 (5, 5)): 길쭉한 2차 함수. Newton은 1걸음에 도착해야 한다.
    - 로젠브록 함수 (시작 (-1.5, 2)): 휘어진 좁은 골짜기. 최솟값 (1, 1).
    각 학습률은 발산하지 않는 범위에서 고른 값이다.
    """
    cases = [
        ("$x^2+10y^2$", ellipse, ellipse_grad, ellipse_hess, START, (0.0, 0.0), 2000, 6.0,
         [VanillaGD(0.05), Momentum(0.01, 0.9), Adam(0.5), Newton()]),
        ("로젠브록 $(1-x)^2+100(y-x^2)^2$", rosenbrock, rosenbrock_grad, rosenbrock_hess,
         np.array([-1.5, 2.0]), (1.0, 1.0), 5000, None,
         [VanillaGD(0.001), Momentum(0.0005, 0.9), Adam(0.1), Newton()]),
    ]
    colors = [BLUE, RED, GREEN, ORANGE]
    fig = plt.figure(figsize=(22, 9.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.5])
    print("[보너스: GD vs Momentum vs Adam vs Newton] 최솟값 반경 1e-3 도달 반복 수")
    for row, (title, f, gf, hf, start, target, n, lim, opts) in enumerate(cases):
        paths = {}
        for opt, c in zip(opts, colors):
            path = optimize(gf, start, opt, n, hess_fn=hf)
            k = iterations_to_reach(path, 1e-3, target)
            paths[f"{opt.name}: {_fmt_iters(k)}"] = (path, c)
            print(f"  {title.split(' ')[0].strip('$'):<10} {opt.name:<26}: {_fmt_iters(k):>8}, "
                  f"최종 ({path[-1][0]:.4f}, {path[-1][1]:.4f})")
        if lim is not None:
            bg = contour_background(f, (-lim, lim), (-lim, lim), np.geomspace(0.05, 400, 14),
                                    start=start, target=target)
        else:
            bg = contour_background(f, (-2, 2), (-1, 3), np.geomspace(0.1, 2500, 18),
                                    start=start, target=target, aspect=False)
        plot_path_panels(paths, bg, [fig.add_subplot(gs[row, k]) for k in range(4)])
        ax = fig.add_subplot(gs[row, 4])
        for label, (path, c) in paths.items():
            d = np.maximum(np.linalg.norm(path - np.asarray(target), axis=1), 1e-16)
            ax.semilogy(d, color=c, label=label)
        ax.set_xscale("symlog", linthresh=10)
        plain_log_ticks(ax, "y")
        ax.set(xlabel="반복 횟수 (10 이후 로그)", ylabel="최솟값까지 거리 (로그)",
               title=f"{title}: 최솟값까지 거리")
        ax.legend(fontsize=8)
    fig.suptitle("보너스: Vanilla GD, Momentum, Adam, Newton 수렴 속도 비교", y=1.0)
    save_figure(fig, "optimizer_bonus_adam_newton")


def experiment_lr_sweep():
    """학습률을 로그 간격으로 바꿔 가며 알고리즘별로 수렴에 필요한 걸음 수를 잰다.

    - x² + 10y² (시작 (5, 5)): 원점 반경 1e-3 도달까지, 최대 3000걸음
    - 선형 회귀 (시작 (0, 4)): 초과 손실 0.01 도달까지, 최대 2000걸음 (SGD 포함)

    2차 함수에서는 Hessian의 고유값 λ로 이론적인 학습률 한계를 구할 수 있다.
        GD 안정 조건:        η < 2 / λ_max
        GD 최적 학습률:      η* = 2 / (λ_min + λ_max)
        Momentum 안정 조건:  η < 2(1 + β) / λ_max
    실험 곡선에 이 값들을 세로선으로 함께 표시한다.

    Returns:
        {문제 이름: {알고리즘 이름: (학습률 배열, 걸음 수 리스트)}} 딕셔너리.
    """
    X, y = make_regression_data()
    A = np.c_[X, np.ones_like(X)]
    best = np.linalg.solve(A.T @ A, A.T @ y)
    best_loss = mse_loss(best, X, y)
    H_reg = 2 * A.T @ A / len(X)

    def run_function(make, lr):
        with np.errstate(all="ignore"):
            path = optimize(ellipse_grad, START, make(lr), 3000)
        return iterations_to_reach(path, 1e-3)

    def run_regression(make, lr):
        with np.errstate(all="ignore"):
            path, _ = train(make(lr), X, y, np.array([0.0, 4.0]), 2000)
            excess = np.array([mse_loss(p, X, y) for p in path]) - best_loss
        if not np.isfinite(excess[-1]):
            return None
        hit = np.nonzero(excess < 1e-2)[0]
        return int(hit[0]) if hit.size else None

    makers = [
        ("Vanilla GD", lambda lr: VanillaGD(lr), BLUE),
        ("SGD (미니배치 16)", lambda lr: SGD(lr, 16), GREEN),
        ("Momentum (β=0.9)", lambda lr: Momentum(lr, 0.9), RED),
        ("Adam", lambda lr: Adam(lr), PURPLE),
    ]
    problems = [
        ("$x^2+10y^2$ (반경 1e-3까지)", np.diag([2.0, 20.0]), run_function, np.geomspace(1e-4, 3, 50)),
        ("선형 회귀 (초과 손실 0.01까지)", H_reg, run_regression, np.geomspace(1e-3, 3, 40)),
    ]
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    print("[학습률 탐색] 수렴까지 걸음 수 (None = 발산 또는 시간 초과)")
    for ax, (title, H, runner, lrs) in zip(axes, problems):
        lam = np.linalg.eigvalsh(H)
        lam_min, lam_max = lam[0], lam[-1]
        gd_limit, gd_best, mom_limit = 2 / lam_max, 2 / (lam_min + lam_max), 2 * 1.9 / lam_max
        print(f"  {title}: Hessian 고유값 = ({lam_min:.3f}, {lam_max:.3f}), 조건수 = {lam_max / lam_min:.1f}")
        print(f"    이론: GD 한계 η < {gd_limit:.4f}, GD 최적 η* = {gd_best:.4f}, Momentum 한계 η < {mom_limit:.4f}")
        results[title] = {}
        for name, make, c in makers:
            if "SGD" in name and runner is run_function:
                continue  # 데이터가 없는 함수에는 SGD를 쓸 수 없다
            steps = [runner(make, lr) for lr in lrs]
            results[title][name] = (lrs, steps)
            ok = [(lr, k) for lr, k in zip(lrs, steps) if k is not None]
            if ok:
                lr_best, k_best = min(ok, key=lambda t: t[1])
                print(f"    {name:<18}: 수렴 범위 {ok[0][0]:.4f} ~ {ok[-1][0]:.4f}, "
                      f"가장 빠른 lr = {lr_best:.4f} ({k_best}걸음)")
            ys = np.array([np.nan if k is None else k for k in steps], dtype=float)
            ax.plot(lrs, ys, "o-", ms=3, color=c, label=name)
        ax.axvline(gd_limit, color=BLUE, ls="--", lw=1)
        ax.axvline(gd_best, color=BLUE, ls=":", lw=1)
        ax.axvline(mom_limit, color=RED, ls="--", lw=1)
        ax.text(gd_limit * 1.05, 0.97, "GD 한계\n" + r"$2/\lambda_{max}$", transform=ax.get_xaxis_transform(),
                color=BLUE, fontsize=8, va="top")
        ax.text(gd_best * 0.95, 0.75, "GD 최적\n" + r"$2/(\lambda_{min}+\lambda_{max})$", transform=ax.get_xaxis_transform(),
                color=BLUE, fontsize=8, va="top", ha="right")
        ax.text(mom_limit * 1.05, 0.6, "Momentum 한계\n" + r"$2(1+\beta)/\lambda_{max}$", transform=ax.get_xaxis_transform(),
                color=RED, fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_ticks(ax)
        ax.set(xlabel=r"학습률 $\eta$ (로그)", ylabel="수렴까지 걸음 수 (로그, 선이 끊기면 발산/시간 초과)",
               title=title)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("학습률 탐색: 너무 작으면 느리고, 한계를 넘으면 발산한다 (Adam은 범위가 넓다)", y=1.02)
    save_figure(fig, "optimizer_lr_sweep")
    return results


def experiment_history():
    """옵티마이저의 계보를 역사 순서대로, "직전 방법의 문제를 다음 방법이 어떻게 고쳤나"로 비교한다.

    1행 (관성 계열, x² + 10y², lr = 0.02): GD → Momentum(1964) → Nesterov(1983)
    2행 (적응적 학습률 계열, 로젠브록, lr = 0.03): AdaGrad(2011) → RMSProp(2012) → Adam(2014)
    같은 행에서는 같은 학습률을 써서 알고리즘 차이만 보이게 한다.
    """
    fig = plt.figure(figsize=(22, 10.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1.25, 1.25])
    print("[옵티마이저 계보] 목표 반경 1e-3 도달 걸음 수")

    def run_row(r, title, gf, start, target, n, runs, bg):
        paths = {}
        for opt, c in runs:
            path = optimize(gf, start, opt, n)
            k = iterations_to_reach(path, 1e-3, target)
            paths[f"{opt.name}: {_fmt_iters(k)}"] = (path, c)
            print(f"  [{title}] {opt.name:<26}: {_fmt_iters(k):>8}, 최종 ({path[-1][0]:+.3f}, {path[-1][1]:+.3f})")
        plot_path_panels(paths, bg, [fig.add_subplot(gs[r, k]) for k in range(3)])
        ax = fig.add_subplot(gs[r, 3])
        for label, (path, c) in paths.items():
            ax.semilogy(np.maximum(np.linalg.norm(path - np.asarray(target), axis=1), 1e-16),
                        color=c, label=label)
        plain_log_ticks(ax, "y")
        ax.set(xlabel="걸음 수", ylabel="목표까지 거리 (로그)", title=title)
        ax.legend(fontsize=7)
        return ax

    ax = run_row(0, "관성 계열 ($x^2+10y^2$, lr=0.02)", ellipse_grad, START, (0.0, 0.0), 300,
                 [(VanillaGD(0.02), BLUE), (Momentum(0.02, 0.9), RED), (Nesterov(0.02, 0.9), ORANGE)],
                 contour_background(ellipse, (-6, 6), (-6, 6), np.geomspace(0.05, 400, 14), start=START))
    fig.add_subplot(gs[0, 4]).axis("off")

    start = np.array([-1.5, 2.0])
    run_row(1, "적응적 학습률 계열 (로젠브록, lr=0.03)", rosenbrock_grad, start, (1.0, 1.0), 5000,
            [(AdaGrad(0.03), BLUE), (RMSProp(0.03), GREEN), (Adam(0.03), PURPLE)],
            contour_background(rosenbrock, (-2, 2), (-1, 3), np.geomspace(0.1, 2500, 18),
                               start=start, target=(1.0, 1.0), aspect=False))
    # AdaGrad와 RMSProp의 "실제 보폭" (lr / √누적값)의 평균을 걸음마다 기록
    ax = fig.add_subplot(gs[1, 4])
    for opt, c in [(AdaGrad(0.03), BLUE), (RMSProp(0.03), GREEN)]:
        opt.reset()
        p = start.copy()
        eff = []
        for _ in range(5000):
            p = opt.step(p, rosenbrock_grad(p))
            acc = opt.G if isinstance(opt, AdaGrad) else opt.s
            eff.append(np.mean(opt.lr / (np.sqrt(acc) + opt.eps)))
        ax.semilogy(eff, color=c, label=opt.name)
    plain_log_ticks(ax, "y")
    ax.set(xlabel="걸음 수", ylabel="실제 학습률 η/√(누적값) (로그)",
           title="AdaGrad는 실제 보폭이 계속 줄어든다")
    ax.legend(fontsize=8)
    fig.suptitle("옵티마이저의 계보: 위 = 관성 계열 (Momentum → Nesterov), "
                 "아래 = 적응적 학습률 계열 (AdaGrad → RMSProp → Adam)", y=1.0)
    save_figure(fig, "optimizer_history")


def experiment_combinations():
    """여러 옵티마이저를 함께 쓰는 방법 세 가지를 실험한다.

    A. 감싸기 — Lookahead: x² + 10y² 에서 Momentum, Adam을 Lookahead로 감쌌을 때와 비교
    B. 섞기 — Levenberg–Marquardt 방식 Newton: 안장점 함수에서 순수 Newton과 비교
    C. 바꾸기 — Adam으로 시작해 Newton으로 전환: 안장점 함수, 로젠브록 함수에서 비교
    """
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.4])
    print("[여러 옵티마이저 함께 쓰기]")

    def row(r, title, f, gf, hf, start, target, n, runs, bg, tol=1e-3):
        paths = {}
        for opt, c in runs:
            with np.errstate(all="ignore"):
                path = optimize(gf, start, opt, n, hess_fn=hf)
            k = iterations_to_reach(path, tol, target)
            label = f"{opt.name}: {_fmt_iters(k)}"
            paths[label] = (path, c)
            print(f"  [{title}] {opt.name:<40}: 목표 도달 {_fmt_iters(k):>8}, "
                  f"최종 ({path[-1][0]:+.4f}, {path[-1][1]:+.4f}), f = {f(path[-1]):.4f}")
        plot_path_panels(paths, bg, [fig.add_subplot(gs[r, k]) for k in range(len(paths))])
        ax = fig.add_subplot(gs[r, 4])
        for label, (path, c) in paths.items():
            ax.semilogy(np.maximum(np.linalg.norm(path - np.asarray(target), axis=1), 1e-16),
                        color=c, label=label)
        ax.set_xscale("symlog", linthresh=10)
        plain_log_ticks(ax, "y")
        ax.set(xlabel="걸음 수 (10 이후 로그)", ylabel="목표까지 거리 (로그)", title=title)
        ax.legend(fontsize=7)
        return paths

    row(0, "A. Lookahead로 감싸기 ($x^2+10y^2$)", ellipse, ellipse_grad, ellipse_hess, START,
        (0.0, 0.0), 2000,
        [(Momentum(0.05, 0.9), RED), (Lookahead(Momentum(0.05, 0.9)), ORANGE),
         (Adam(0.5), GREEN), (Lookahead(Adam(0.5)), PURPLE)],
        contour_background(ellipse, (-6, 6), (-6, 6), np.geomspace(0.05, 400, 14), start=START))

    start = np.array([2.0, 0.3])
    row(1, "B·C. 안장점 함수, 목표 최솟값 (0, 1)", saddle, saddle_grad, saddle_hess, start,
        (0.0, 1.0), 300,
        [(Newton(), GRAY), (LevenbergMarquardt(saddle), BLUE),
         (SwitchOptimizer(Adam(0.1), Newton(), 20), GREEN), (Adam(0.1), PURPLE)],
        contour_background(saddle, (-2.2, 2.2), (-1.6, 1.6), 15, start=start, target=(0.0, 1.0)))

    start = np.array([-1.5, 2.0])
    row(2, "B·C. 로젠브록 함수, 최솟값 (1, 1)", rosenbrock, rosenbrock_grad, rosenbrock_hess,
        start, (1.0, 1.0), 5000,
        [(Adam(0.1), PURPLE), (SwitchOptimizer(Adam(0.1), Newton(), 50), GREEN),
         (Newton(), GRAY), (LevenbergMarquardt(rosenbrock), BLUE)],
        contour_background(rosenbrock, (-2, 2), (-1, 3), np.geomspace(0.1, 2500, 18),
                           start=start, target=(1.0, 1.0), aspect=False))
    fig.suptitle("여러 옵티마이저 함께 쓰기: A. 감싸기 (Lookahead)  B. 섞기 (Levenberg–Marquardt)  "
                 "C. 바꾸기 (Adam → Newton)", y=1.0)
    save_figure(fig, "optimizer_combinations")


# ---------------------------------------------------------------------------
# 애니메이션 (GIF)
# ---------------------------------------------------------------------------

def animate_paths(paths, draw_background, title, name, n_frames=None, fps=15, hold=20,
                  steps=None, legend_loc="lower right", ncols=None):
    """여러 최적화 경로가 한 걸음씩 움직이는 모습을 GIF로 저장한다.

    매 프레임마다 각 경로의 지나온 길(선)과 현재 위치(큰 점)를 그리고,
    제목 옆에 현재 걸음 수를 표시한다. 마지막 프레임은 hold번 반복해 잠시 멈춘다.

    Args:
        paths: {범례 이름: (경로 배열 shape (T, 2), 색)} 딕셔너리.
        draw_background: Axes를 받아 등고선 등 배경을 그리는 함수.
            칸을 나눌 때는 칸마다 다른 배경 함수의 목록을 줄 수도 있다(칸마다 축 범위가 다를 때).
        title: 그림 제목.
        name: 저장할 파일 이름 (확장자 제외).
        n_frames: 보여 줄 걸음 수. None이면 가장 긴 경로 길이.
        fps: 초당 프레임 수.
        hold: 마지막 장면에서 멈춰 있을 프레임 수.
        steps: 프레임마다 보여 줄 걸음 번호 목록. 걸음 수가 수천 개일 때
            로그 간격으로 골라 GIF 크기를 줄인다. None이면 0, 1, 2, ... 를 모두 보여 준다.
        legend_loc: 범례 위치 (한 칸에 겹쳐 그릴 때).
        ncols: 지정하면 방법마다 칸을 나눠 ncols열 격자로 그린다(경로가 겹칠 때).
            None이면 모든 경로를 한 칸에 겹쳐 그린다.

    Returns:
        저장된 GIF 경로.
    """
    n_frames = n_frames or max(len(p) for p, _ in paths.values())
    steps = list(range(n_frames)) if steps is None else list(steps)
    artists = []
    if ncols is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        draw_background(ax)
        axes = [ax] * len(paths)
    else:
        nrows = int(np.ceil(len(paths) / ncols))
        fig, grid = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.2 * nrows + 0.5),
                                 squeeze=False)
        axes = list(grid.ravel())
        for ax in axes[len(paths):]:
            ax.axis("off")
        backgrounds = (draw_background if isinstance(draw_background, (list, tuple))
                       else [draw_background] * len(paths))
        for ax, bg, (label, (_, color)) in zip(axes, backgrounds, paths.items()):
            bg(ax)
            ax.set_title(label, fontsize=9, color=color)
    for ax, (label, (path, color)) in zip(axes, paths.items()):
        (line,) = ax.plot([], [], "-", color=color, lw=1.6, label=label)
        (dot,) = ax.plot([], [], "o", color=color, ms=7, mec="white", mew=1)
        artists.append((path, line, dot))
    if ncols is None:
        axes[0].legend(fontsize=8, loc=legend_loc)
        text = axes[0].set_title(title)
    else:
        text = fig.suptitle(title)

    def update(frame):
        k = steps[min(frame, len(steps) - 1)]
        for path, line, dot in artists:
            j = min(k, len(path) - 1)
            line.set_data(path[: j + 1, 0], path[: j + 1, 1])
            dot.set_data([path[j, 0]], [path[j, 1]])
        text.set_text(f"{title}  (걸음 {k})")
        return [a for _, line, dot in artists for a in (line, dot)] + [text]

    anim = FuncAnimation(fig, update, frames=len(steps) + hold, blit=False)
    fig.tight_layout()
    path = save_animation(anim, name, fps=fps)
    plt.close(fig)
    return path


def make_animations():
    """구현한 모든 경사하강법의 수렴 과정을 GIF 네 개로 저장한다.

    1. x² + y², 학습률 다섯 가지 (수렴, 한 번에 도착, 진동 수렴, 제자리 진동, 발산)
    2. x² + 10y², Vanilla GD, Momentum, Adam, Newton
    3. 선형 회귀 손실 지형, GD, SGD(미니배치 16, 1), Momentum, Adam, Newton
    4. 로젠브록 함수, Vanilla GD, Momentum, Adam, Newton (5000걸음을 로그 간격으로)
    """
    contour_bg = contour_background

    # 1. 학습률: 학습률마다 칸을 나누고, 모두 같은 출발점 (5, 5)에서 시작한다.
    #    발산하는 경우만 멀리 나가므로 칸마다 축 범위를 경로에 맞춘다.
    lrs = [(0.1, BLUE), (0.5, GREEN), (0.9, ORANGE), (1.0, GRAY), (1.05, RED)]
    paths, backgrounds = {}, []
    for lr, c in lrs:
        path = optimize(bowl_grad, START, VanillaGD(lr), 12)
        paths[f"lr = {lr}\n(곱해지는 수 {1 - 2 * lr:+.1f})"] = (path, c)
        lim = max(6.0, np.abs(path).max() * 1.1)
        backgrounds.append(contour_bg(bowl, (-lim, lim), (-lim, lim),
                                      np.geomspace(0.5, 2 * lim**2, 12), start=START))
    animate_paths(paths, backgrounds, "$x^2+y^2$, 시작 (5, 5): 학습률 비교",
                  "optimizer_learning_rates_anim", fps=3, hold=6, ncols=5)

    # 2. 길쭉한 골짜기
    n = 150
    paths = {
        "Vanilla GD (lr=0.01)": (optimize(ellipse_grad, START, VanillaGD(0.01), n), BLUE),
        "Momentum (lr=0.01, β=0.9)": (optimize(ellipse_grad, START, Momentum(0.01, 0.9), n), RED),
        "Adam (lr=0.5)": (optimize(ellipse_grad, START, Adam(0.5), n), GREEN),
        "Newton": (optimize(ellipse_grad, START, Newton(), n, hess_fn=ellipse_hess), ORANGE),
    }
    bg = contour_bg(ellipse, (-6, 6), (-6, 6), np.geomspace(0.05, 400, 14), start=START)
    animate_paths(paths, bg, "$x^2+10y^2$", "optimizer_ellipse_anim", n_frames=n + 1, ncols=4)

    # 3. 선형 회귀 (데이터가 있어야 SGD를 비교할 수 있다)
    X, y = make_regression_data()
    A = np.c_[X, np.ones_like(X)]
    best = np.linalg.solve(A.T @ A, A.T @ y)
    start = np.array([0.0, 4.0])

    def regression_bg(ax):
        W, B = np.meshgrid(np.linspace(-0.5, 3.5, 250), np.linspace(-1.5, 4.5, 250))
        L = np.mean((W[..., None] * X + B[..., None] - y) ** 2, axis=-1)
        ax.contour(W, B, L, levels=np.geomspace(mse_loss(best, X, y) * 1.05, L.max(), 16),
                   colors=GRAY, linewidths=0.7, alpha=0.7)
        ax.plot(*start, "ks", ms=6)
        ax.plot(*best, "k*", ms=12)
        ax.set(xlabel="기울기 w", ylabel="절편 b")

    n = 150
    paths = {
        "GD (lr=0.05, 200개 전부)": (train(VanillaGD(0.05), X, y, start, n)[0], BLUE),
        "SGD (lr=0.05, 미니배치 16)": (train(SGD(0.05, 16), X, y, start, n)[0], GREEN),
        "SGD (lr=0.02, 미니배치 1)": (train(SGD(0.02, 1), X, y, start, n)[0], ORANGE),
        "Momentum (lr=0.05, β=0.9)": (train(Momentum(0.05, 0.9), X, y, start, n)[0], RED),
        "Adam (lr=0.1)": (train(Adam(0.1), X, y, start, n)[0], PURPLE),
        "Newton": (train(Newton(), X, y, start, n)[0], GRAY),
    }
    animate_paths(paths, regression_bg, "선형 회귀 손실 지형", "optimizer_gd_sgd_momentum_anim",
                  n_frames=n + 1, ncols=3)

    # 4. 로젠브록: 5000걸음을 로그 간격 120장으로
    n = 5000
    start = np.array([-1.5, 2.0])
    paths = {
        "Vanilla GD (lr=0.001)": (optimize(rosenbrock_grad, start, VanillaGD(0.001), n), BLUE),
        "Momentum (lr=0.0005, β=0.9)": (optimize(rosenbrock_grad, start, Momentum(0.0005, 0.9), n), RED),
        "Adam (lr=0.1)": (optimize(rosenbrock_grad, start, Adam(0.1), n), GREEN),
        "Newton": (optimize(rosenbrock_grad, start, Newton(), n, hess_fn=rosenbrock_hess), ORANGE),
    }
    steps = np.unique(np.r_[0, np.round(np.geomspace(1, n, 120)).astype(int)])
    bg = contour_bg(rosenbrock, (-2, 2), (-1, 3), np.geomspace(0.1, 2500, 18), start=start,
                    target=(1.0, 1.0), aspect=False)
    animate_paths(paths, bg, "로젠브록 함수", "optimizer_rosenbrock_anim", steps=steps, fps=12, ncols=4)


LR_GRID = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])

ALL_METHODS = [
    ("Vanilla GD", lambda lr: VanillaGD(lr), BLUE),
    ("Momentum", lambda lr: Momentum(lr, 0.9), RED),
    ("Nesterov", lambda lr: Nesterov(lr, 0.9), ORANGE),
    ("AdaGrad", lambda lr: AdaGrad(lr), PINK),
    ("RMSProp", lambda lr: RMSProp(lr), GREEN),
    ("Adam", lambda lr: Adam(lr), PURPLE),
    ("Newton", None, GRAY),   # 학습률 없음 (Hessian으로 보폭을 정한다)
]


def iterations_to_settle(path, tol=1e-3, target=(0.0, 0.0)):
    """경로가 목표 반경 tol 안에 들어간 뒤 **끝까지 머문** 첫 걸음 번호를 찾는다.

    iterations_to_reach는 반경을 한 번 스쳐 지나가도 도달로 센다. 학습률이 크면 최솟값을
    지나쳐 다시 멀어지는 경우가 있어서, 여기서는 그 뒤로 계속 반경 안에 있어야 수렴으로 본다.

    Returns:
        수렴한 걸음 번호. 마지막 위치가 반경 밖이면 None.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        dist = np.linalg.norm(path - np.asarray(target), axis=1)
    outside = ~(dist < tol)            # nan, inf도 반경 밖으로 본다
    if outside[-1]:
        return None
    idx = np.nonzero(outside)[0]
    return 0 if idx.size == 0 else int(idx[-1] + 1)


def tune_lr(make_opt, grad_fn, start, target, n_iters, hess_fn=None, lrs=LR_GRID, tol=1e-3):
    """학습률 후보마다 실행해 목표 반경 tol 안으로 가장 빨리 수렴한(들어가서 머문) 학습률을 고른다.

    Args:
        make_opt: 학습률을 받아 옵티마이저를 만드는 함수. None이면 학습률 없는 방법(Newton)으로 1번만 실행.
        lrs: 학습률 후보.

    Returns:
        (가장 좋은 lr, 그때의 경로, 도달 걸음 수, {lr: 도달 걸음 수}).
        어느 학습률로도 도달하지 못하면 최종 거리가 가장 가까운 학습률을 고른다.
    """
    if make_opt is None:
        path = optimize(grad_fn, start, Newton(), n_iters, hess_fn=hess_fn)
        k = iterations_to_settle(path, tol, target)
        return None, path, k, {None: k}

    results = {}
    best = None   # (정렬 키, lr, path, k)
    for lr in lrs:
        with np.errstate(over="ignore", invalid="ignore"):
            path = optimize(grad_fn, start, make_opt(lr), n_iters)
            k = iterations_to_settle(path, tol, target)
            final = np.linalg.norm(path[-1] - np.asarray(target))
        results[lr] = k
        key = (0, k) if k is not None else (1, final if np.isfinite(final) else np.inf)
        if best is None or key < best[0]:
            best = (key, lr, path, k)
    return best[1], best[2], best[3], results


def experiment_all_optimizers():
    """모든 경사하강법을 x² + 10y²와 로젠브록 두 지형에서, 각자 가장 좋은 학습률로 비교한다.

    학습률이 같으면 어떤 방법은 발산하고 어떤 방법은 멈추므로, 방법마다 LR_GRID의 13개
    후보를 모두 시험해 목표(반경 1e-3) 안으로 가장 빨리 수렴한 학습률을 쓴다. 수렴은
    반경 안에 들어간 뒤 끝까지 머문 것을 말한다(iterations_to_settle). 이렇게 하면
    "각 방법이 최선을 다했을 때"의 비교가 된다. 도달한 학습률 후보 개수는 학습률을
    고르기 쉬운 정도(민감도)를 보여 준다.
    SGD는 데이터가 없는 함수라 비교에서 뺐다 (선형 회귀 실험에서 비교).
    """
    cases = [
        ("$x^2+10y^2$", "x² + 10y²", ellipse, ellipse_grad, ellipse_hess, START, (0.0, 0.0), 3000,
         contour_background(ellipse, (-6, 6), (-6, 6), np.geomspace(0.05, 400, 14), start=START)),
        ("로젠브록", "로젠브록", rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array([-1.5, 2.0]),
         (1.0, 1.0), 10000,
         contour_background(rosenbrock, (-2, 2), (-1, 3), np.geomspace(0.1, 2500, 18),
                            start=np.array([-1.5, 2.0]), target=(1.0, 1.0), aspect=False)),
    ]
    n_m = len(ALL_METHODS)
    fig = plt.figure(figsize=(3.3 * n_m, 13))
    gs = fig.add_gridspec(3, n_m, height_ratios=[1, 1, 0.9])
    fig_h, axes_h = plt.subplots(1, 2, figsize=(16, 4.6))
    print(f"[모든 경사하강법 비교] 방법마다 학습률 {len(LR_GRID)}개 중 가장 빠른 값 사용, "
          "목표 반경 1e-3 안에 들어가 끝까지 머문 걸음 수")
    summary = {}

    for row, (title, name, f, gf, hf, start, target, n, bg) in enumerate(cases):
        paths, grid = {}, []
        print(f"  [{name}] 시작 ({start[0]:g}, {start[1]:g}), 최대 {n}걸음")
        for m_name, make_opt, color in ALL_METHODS:
            lr, path, k, results = tune_lr(make_opt, gf, start, target, n, hess_fn=hf)
            n_ok = sum(v is not None for v in results.values())
            lr_txt = "없음" if lr is None else f"{lr:g}"
            summary[(name, m_name)] = (lr, k, n_ok)
            label = f"{m_name}\n{'학습률 없음' if lr is None else f'lr={lr_txt}'}: {_fmt_iters(k)}"
            paths[label] = (path, color)
            final = path[-1]
            print(f"    {m_name:<11} 최적 lr = {lr_txt:<7} 도달 {_fmt_iters(k):>9}, "
                  f"도달한 lr 후보 {n_ok if make_opt else '-'}/{len(LR_GRID) if make_opt else '-'}, "
                  f"최종 ({final[0]:+.4f}, {final[1]:+.4f})")
            if make_opt is not None:
                grid.append([np.nan if results[lr_] is None else results[lr_] for lr_ in LR_GRID])
        plot_path_panels(paths, bg, [fig.add_subplot(gs[row, k]) for k in range(n_m)])

        ax = fig.add_subplot(gs[2, row * (n_m // 2): (row + 1) * (n_m // 2) + row])
        for label, (path, c) in paths.items():
            with np.errstate(over="ignore", invalid="ignore"):
                d = np.linalg.norm(path - np.asarray(target), axis=1)
            ax.semilogy(np.maximum(np.nan_to_num(d, nan=1e3, posinf=1e3), 1e-16), color=c,
                        label=label.replace("\n", " "))
        ax.set_xscale("symlog", linthresh=10)
        plain_log_ticks(ax, "y")
        ax.set(xlabel="걸음 수 (10 이후 로그)", ylabel="최솟값까지 거리 (로그)", title=f"{title}: 최솟값까지 거리")
        ax.legend(fontsize=7)

        # 학습률 × 방법 도달 걸음 수 표 (회색 = 도달 못 함)
        ah = axes_h[row]
        grid = np.array(grid, dtype=float)
        im = ah.imshow(np.log10(grid), cmap="viridis_r", aspect="auto")
        ah.set_facecolor("#dddddd")
        names = [m for m, mk, _ in ALL_METHODS if mk is not None]
        ah.set(xticks=range(len(LR_GRID)), xticklabels=[f"{v:g}" for v in LR_GRID],
               yticks=range(len(names)), yticklabels=names, xlabel="학습률 lr",
               title=f"{title}: 수렴 걸음 수 (회색 = 수렴 못 함)")
        ah.tick_params(axis="x", labelrotation=45)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    v = int(grid[i, j])
                    ah.text(j, i, f"{v}", ha="center", va="center", fontsize=7,
                            color="white" if np.log10(v) > np.nanmean(np.log10(grid)) else "black")

    fig.suptitle("모든 경사하강법 비교: 위 = x² + 10y², 가운데 = 로젠브록 (각 방법의 최적 학습률)", y=1.0)
    save_figure(fig, "optimizer_all_methods")
    fig_h.suptitle("학습률별 수렴 걸음 수: 칸이 많이 채워진 방법일수록 학습률을 고르기 쉽다", y=1.02)
    save_figure(fig_h, "optimizer_all_methods_lr_grid")
    return summary


def main():
    """과제 요구 사항을 차례대로 실행하고 그림을 outputs/에 저장한다."""
    setup_style()
    experiment_vanilla_bowl()
    experiment_learning_rates()
    experiment_momentum_bowl()
    experiment_ellipse()
    experiment_gd_sgd_momentum()
    experiment_bonus_adam_newton()
    experiment_lr_sweep()
    experiment_history()
    experiment_combinations()
    experiment_all_optimizers()
    make_animations()


if __name__ == "__main__":
    main()
