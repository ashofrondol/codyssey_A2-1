"""확률·통계: 분포 시각화, softmax, 그리고 손실 함수(MSE, Cross-Entropy)와 MLE의 연결.

노트북 `notebooks/probability_loss.ipynb`가 이 모듈을 불러 쓴다. 단독 실행도 된다.

실행:
    uv run python src/probability.py

하는 일
    1. 정규분포 N(0, 1)과 N(2, 0.5)의 PDF를 한 Figure에 비교한다.        → plot_normal_pdfs()
    2. 베르누이 분포 B(0.3)과 B(0.7)의 PMF를 한 Figure에 비교한다.       → plot_bernoulli_pmfs()
    3. softmax를 구현하고 출력 합과 1의 차이가 1e-6 이내인지 확인한다.   → softmax(), check_softmax_sum()
    4. MSE 최소화 = 정규분포 가정 하의 MLE 를 유도하고 숫자로 확인한다.  → check_mse_mle()
    5. Cross-Entropy 최소화 = 베르누이/카테고리 분포 가정 하의 MLE.      → check_bce_mle(), check_softmax_ce_grad()
    6. (보너스) 엔트로피, KL-Divergence, CE = H + KL.                   → check_info_theory()

유도 1: MSE 최소화 = 정규분포 가정 하의 MLE
    가정: y_i = f(x_i) + ε_i, ε_i ~ N(0, σ²) 이고 서로 독립.
      p(y_i | x_i) = 1/√(2πσ²) · exp(-(y_i - f(x_i))² / (2σ²))
    우도는 곱, 로그를 씌우면 합이 된다.
      log L = Σ log p(y_i | x_i) = -n·log√(2πσ²) - (1/(2σ²)) Σ (y_i - f(x_i))²
    첫 항과 1/(2σ²)는 f와 무관한 상수다. 따라서
      argmax log L = argmin Σ (y_i - f(x_i))² = argmin (1/n) Σ (y_i - f(x_i))² = argmin MSE
    즉 NLL = n/(2σ²) × MSE + 상수 이고, 최솟값의 위치가 같다. (check_mse_mle에서 확인)

유도 2: BCE 최소화 = 베르누이 가정 하의 MLE
    가정: y_i ∈ {0, 1} 은 ŷ_i = f(x_i) 를 성공 확률로 하는 베르누이 분포를 따른다.
      p(y_i | x_i) = ŷ_i^{y_i} (1 - ŷ_i)^{1 - y_i}
      log L = Σ [ y_i log ŷ_i + (1 - y_i) log(1 - ŷ_i) ]
    부호를 바꿔 n으로 나누면 BCE 다.
      argmax log L = argmin -(1/n) Σ [ y_i log ŷ_i + (1 - y_i) log(1 - ŷ_i) ] = argmin BCE
    (check_bce_mle: 동전 10번 중 앞면 7번 → BCE 최소 = 우도 최대 = p 0.70)

유도 3: Cross-Entropy 최소화 = 카테고리 분포 가정 하의 MLE
    K개 클래스, 정답 one-hot y_i, 예측 확률 ŷ_i = softmax(z_i).
      p(y_i | x_i) = Π_k ŷ_{ik}^{y_{ik}}
      -log L = -Σ_i Σ_k y_{ik} log ŷ_{ik} = Σ_i CE(y_i, ŷ_i)
    softmax + CE 의 기울기는 ∂CE/∂z_k = ŷ_k - y_k 이다. (check_softmax_ce_grad에서 중심차분으로 확인)
"""

import sys
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculus import numerical_gradient  # noqa: E402
from plot_style import BLUE, GRAY, GREEN, ORANGE, RED, save_figure, setup_style  # noqa: E402

SEED = 42
SOFTMAX_TOL = 1e-6


# ---------------------------------------------------------------------------
# 1. 정규분포 PDF
# ---------------------------------------------------------------------------

def normal_pdf(x, mu, var):
    """정규분포 N(mu, var)의 확률밀도 p(x) = exp(-(x-mu)²/(2var)) / sqrt(2π var).

    Args:
        x: 값 (배열 가능).
        mu: 평균 μ.
        var: 분산 σ².

    Returns:
        x와 같은 shape의 확률밀도.
    """
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def plot_normal_pdfs():
    """N(0, 1)과 N(2, 0.5)의 PDF를 한 Figure에 그린다. N(2, 0.5)의 0.5는 분산으로 해석한다.

    Returns:
        Figure. outputs/prob_normal_pdf.png 로도 저장한다.
    """
    xs = np.linspace(-4, 5, 1000)
    curves = [
        (r"N(0, 1): $\mu=0,\ \sigma^2=1$", normal_pdf(xs, 0, 1), BLUE, "-"),
        (r"N(2, 0.5): $\mu=2,\ \sigma^2=0.5\ (\sigma \approx 0.707)$", normal_pdf(xs, 2, 0.5), RED, "-"),
        (r"참고: $\sigma=0.5\ (\sigma^2=0.25)$ 로 해석한 경우", normal_pdf(xs, 2, 0.25), RED, ":"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, ys, c, ls in curves:
        area = np.trapezoid(ys, xs)
        ax.plot(xs, ys, color=c, ls=ls, lw=2, label=f"{label}  (넓이 {area:.4f}, 최고점 {ys.max():.3f})")
        print(f"{label:<55} 곡선 아래 넓이 = {area:.6f}, 최고점 = {ys.max():.4f}")
    ax.fill_between(xs, normal_pdf(xs, 0, 1), color=BLUE, alpha=0.08)
    ax.fill_between(xs, normal_pdf(xs, 2, 0.5), color=RED, alpha=0.08)
    ax.set(xlabel="x", ylabel="확률밀도 p(x)",
           title="정규분포 PDF: 분산이 작을수록 좁고 높다 (넓이는 항상 1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 1.05)
    save_figure(fig, "prob_normal_pdf")
    return fig


# ---------------------------------------------------------------------------
# 2. 베르누이 PMF
# ---------------------------------------------------------------------------

def bernoulli_pmf(y, p):
    """베르누이 분포 B(p)의 확률 P(Y=y) = p^y (1-p)^(1-y), y ∈ {0, 1}.

    Args:
        y: 0 또는 1 (배열 가능).
        p: 1이 나올 확률.

    Returns:
        y와 같은 shape의 확률.
    """
    y = np.asarray(y)
    return p**y * (1 - p) ** (1 - y)


def plot_bernoulli_pmfs():
    """B(0.3)과 B(0.7)의 PMF 막대그래프를 한 Figure에 그린다.

    Returns:
        Figure. outputs/prob_bernoulli_pmf.png 로도 저장한다.
    """
    ys = np.array([0, 1])
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for shift, p, c in [(-width / 2, 0.3, BLUE), (width / 2, 0.7, RED)]:
        probs = bernoulli_pmf(ys, p)
        bars = ax.bar(ys + shift, probs, width, color=c, label=f"B({p})")
        ax.bar_label(bars, fmt="%.1f")
        print(f"B({p}): P(0) = {probs[0]:.1f}, P(1) = {probs[1]:.1f}, 합 = {probs.sum():.1f}")
    ax.set(xticks=[0, 1], xticklabels=["y = 0 (실패)", "y = 1 (성공)"], ylim=(0, 1),
           ylabel="확률 P(Y = y)", title="베르누이 분포 PMF: B(0.3)과 B(0.7)은 서로 좌우 대칭")
    ax.legend()
    save_figure(fig, "prob_bernoulli_pmf")
    return fig


# ---------------------------------------------------------------------------
# 3. softmax
# ---------------------------------------------------------------------------

def softmax(z):
    """수치적으로 안정한 소프트맥스. 최댓값을 빼서 오버플로를 막는다.

    softmax(z)_k = exp(z_k - max(z)) / Σ_j exp(z_j - max(z))

    Args:
        z: 점수 벡터. shape (K,).

    Returns:
        확률 벡터. shape (K,), 합 = 1.
    """
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def softmax_naive(z):
    """최댓값을 빼지 않은 소프트맥스 (비교용). 큰 입력에서 nan이 된다."""
    e = np.exp(np.asarray(z, dtype=float))
    return e / np.sum(e)


def softmax_test_cases(rng):
    """softmax 합 검증에 쓸 입력들. 무작위 입력은 rng에서 뽑는다."""
    return {
        "작은 점수 [2, 1, 0.1]": np.array([2.0, 1.0, 0.1]),
        "음수 포함 [-3, 0, 3, 1]": np.array([-3.0, 0.0, 3.0, 1.0]),
        "큰 점수 [1000, 1001, 1002]": np.array([1000.0, 1001.0, 1002.0]),
        "무작위 10개 (seed 42)": rng.randn(10) * 5,
    }


def check_softmax_sum(tests, tol=SOFTMAX_TOL):
    """각 입력에 대해 |Σ softmax(z) - 1| <= tol 인지 확인한다.

    Args:
        tests: {이름: 점수 벡터} dict.
        tol: 허용 오차 (기본 1e-6).

    Returns:
        모두 통과하면 True.
    """
    all_ok = True
    for name, z in tests.items():
        p = softmax(z)
        diff = abs(p.sum() - 1.0)
        ok = diff <= tol
        all_ok &= ok
        with np.errstate(over="ignore", invalid="ignore"):
            naive = softmax_naive(z)
        print(f"{name:<26} 합 = {p.sum():.10f}, |합 - 1| = {diff:.1e} → {'통과' if ok else '실패'}"
              f" | 안정화 안 한 버전: {'nan 발생' if np.isnan(naive).any() else '정상'}")
    print("\n[2, 1, 0.1] →", np.round(softmax([2.0, 1.0, 0.1]), 4))
    return all_ok


def plot_softmax(z=(2.0, 1.0, 0.1)):
    """점수 z와 softmax(z) 확률을 나란히 막대그래프로 그린다."""
    z = np.asarray(z, dtype=float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.bar(["A", "B", "C"], z, color=GRAY)
    a1.set(title="입력 점수 z (logit)", ylabel="점수")
    probs = softmax(z)
    bars = a2.bar(["A", "B", "C"], probs, color=[RED, ORANGE, BLUE])
    a2.bar_label(bars, fmt="%.3f")
    a2.set(title=f"softmax(z): 합 = {probs.sum():.6f}", ylabel="확률", ylim=(0, 1))
    save_figure(fig, "prob_softmax")
    return fig


# ---------------------------------------------------------------------------
# 4. MSE 최소화 = 정규분포 MLE
# ---------------------------------------------------------------------------

def make_regression_data(rng, n=50, sigma=0.5):
    """y = 2x + 1 + N(0, σ²) 잡음 데이터를 만든다. x는 [-2, 2] 균등분포."""
    x_data = rng.uniform(-2, 2, n)
    y_data = 2 * x_data + 1 + rng.normal(0, sigma, n)
    return x_data, y_data


def check_mse_mle(x_data, y_data, sigma=0.5):
    """기울기 w를 바꿔 가며 MSE와 NLL(음의 로그 우도)을 계산하고, 두 최솟값의 위치를 비교한다.

    Returns:
        (w_mse, w_nll, fig).
    """
    n = len(x_data)
    ws = np.linspace(1.0, 3.0, 2001)
    mse = np.array([np.mean((y_data - (w * x_data + 1)) ** 2) for w in ws])
    nll = np.array([-np.sum(np.log(normal_pdf(y_data, w * x_data + 1, sigma**2))) for w in ws])

    w_mse, w_nll = ws[np.argmin(mse)], ws[np.argmin(nll)]
    print(f"MSE를 최소로 만드는 w = {w_mse:.4f}")
    print(f"NLL을 최소로 만드는 w = {w_nll:.4f}  (우도 최대)")
    print(f"→ {'같은 w' if np.isclose(w_mse, w_nll) else '다른 w'}")
    print(f"NLL = n/(2σ²) × MSE + 상수 확인: 최대 차이 = "
          f"{np.max(np.abs(nll - (n / (2 * sigma**2) * mse + n * np.log(np.sqrt(2 * np.pi * sigma**2))))):.1e}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    a1.scatter(x_data, y_data, s=12, color=GRAY, label="데이터")
    xx = np.linspace(-2, 2, 2)
    a1.plot(xx, w_mse * xx + 1, color=RED, label=f"y = {w_mse:.3f}x + 1")
    a1.set(xlabel="x", ylabel="y", title="데이터와 최적 직선")
    a1.legend(fontsize=9)
    a2.plot(ws, mse, color=BLUE)
    a2.axvline(w_mse, color=RED, ls="--")
    a2.set(xlabel="기울기 w", ylabel="MSE", title=f"MSE 최소: w = {w_mse:.3f}")
    a3.plot(ws, nll, color=GREEN)
    a3.axvline(w_nll, color=RED, ls="--")
    a3.set(xlabel="기울기 w", ylabel="음의 로그 우도 (NLL)", title=f"NLL 최소 (우도 최대): w = {w_nll:.3f}")
    fig.suptitle("정규분포 가정에서 MSE 최소화와 MLE는 같은 답을 준다", y=1.02)
    save_figure(fig, "prob_mse_mle")
    return w_mse, w_nll, fig


# ---------------------------------------------------------------------------
# 5. Cross-Entropy 최소화 = 베르누이/카테고리 MLE
# ---------------------------------------------------------------------------

def check_bce_mle(coins=(1, 1, 1, 0, 1, 1, 0, 1, 0, 1)):
    """동전 결과로 p를 바꿔 가며 BCE와 우도를 계산하고, BCE 최소 = 우도 최대 = 표본 비율인지 본다.

    Returns:
        (p_bce, p_lik, fig).
    """
    coins = np.asarray(coins)
    ps = np.linspace(0.01, 0.99, 981)
    bce = np.array([-np.mean(coins * np.log(p) + (1 - coins) * np.log(1 - p)) for p in ps])
    likelihood = np.array([np.prod(bernoulli_pmf(coins, p)) for p in ps])
    p_bce, p_lik = ps[np.argmin(bce)], ps[np.argmax(likelihood)]
    print(f"BCE 최소 p = {p_bce:.2f}, 우도 최대 p = {p_lik:.2f}, 표본 비율 = {coins.mean():.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ps, likelihood, color=GREEN)
    a1.axvline(p_lik, color=RED, ls="--")
    a1.set(xlabel="앞면 확률 p", ylabel="우도 Π p(y_i)", title=f"우도 최대: p = {p_lik:.2f}")
    a2.plot(ps, bce, color=BLUE)
    a2.axvline(p_bce, color=RED, ls="--")
    a2.set(xlabel="앞면 확률 p", ylabel="BCE (평균)", title=f"BCE 최소: p = {p_bce:.2f}")
    fig.suptitle("베르누이 가정에서 BCE 최소화 = MLE (10번 중 앞면 7번)", y=1.02)
    save_figure(fig, "prob_bce_mle")
    return p_bce, p_lik, fig


def cross_entropy(z, y_onehot):
    """소프트맥스 확률에 대한 Cross-Entropy -Σ y_k log softmax(z)_k."""
    return -np.sum(y_onehot * np.log(softmax(z)))


def check_softmax_ce_grad(z=(2.0, 1.0, 0.1), y_onehot=(0.0, 1.0, 0.0)):
    """softmax + CE 의 기울기 공식 ŷ - y 를 중심차분과 비교한다.

    Returns:
        공식과 중심차분의 최대 차이.
    """
    z, y_onehot = np.asarray(z, dtype=float), np.asarray(y_onehot, dtype=float)
    analytic = softmax(z) - y_onehot
    numeric = numerical_gradient(lambda v: cross_entropy(v, y_onehot), z)
    k = int(np.argmax(y_onehot))
    print("CE =", round(cross_entropy(z, y_onehot), 4), "= -log(정답 확률", round(softmax(z)[k], 4), ")")
    print("공식 ŷ - y      :", np.round(analytic, 6))
    print("중심차분 기울기  :", np.round(numeric, 6))
    diff = np.max(np.abs(analytic - numeric))
    print("최대 차이        :", f"{diff:.1e}")
    return diff


# ---------------------------------------------------------------------------
# 6. (보너스) 정보 이론
# ---------------------------------------------------------------------------

def entropy(p):
    """엔트로피 H(p) = -Σ p log p (p = 0 인 항은 0으로 본다)."""
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return -np.sum(p[nz] * np.log(p[nz]))


def kl_divergence(p, q):
    """KL(p || q) = Σ p log(p / q) (p = 0 인 항은 0으로 본다)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nz = p > 0
    return np.sum(p[nz] * np.log(p[nz] / q[nz]))


def cross_entropy_pq(p, q):
    """Cross-Entropy H(p, q) = -Σ p log q."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nz = p > 0
    return -np.sum(p[nz] * np.log(q[nz]))


def check_info_theory(p=(0.7, 0.2, 0.1)):
    """여러 q에 대해 CE(p, q) = H(p) + KL(p || q) 가 성립하는지 확인한다."""
    p = np.asarray(p, dtype=float)
    all_ok = True
    for name, q in [("q = p", p), ("q = [0.5, 0.3, 0.2]", np.array([0.5, 0.3, 0.2])),
                    ("q = 균등 [1/3, 1/3, 1/3]", np.ones(3) / 3)]:
        H, KL, CE = entropy(p), kl_divergence(p, q), cross_entropy_pq(p, q)
        ok = np.isclose(CE, H + KL)
        all_ok &= ok
        print(f"{name:<26} H(p) = {H:.4f}, KL = {KL:.4f}, CE = {CE:.4f}, H + KL = {H + KL:.4f} "
              f"→ {'CE = H + KL' if ok else '불일치'}")
    return all_ok


def main():
    setup_style()
    rng = np.random.RandomState(SEED)   # softmax 무작위 입력 → 회귀 데이터 순서로 뽑는다

    print("[1] 정규분포 PDF 2종")
    plot_normal_pdfs()
    print("\n[2] 베르누이 PMF 2종")
    plot_bernoulli_pmfs()
    print(f"\n[3] softmax 합 = 1 확인 (허용 오차 {SOFTMAX_TOL})")
    assert check_softmax_sum(softmax_test_cases(rng))
    plot_softmax()
    print("\n[4] MSE 최소화 = 정규분포 MLE")
    w_mse, w_nll, _ = check_mse_mle(*make_regression_data(rng))
    assert np.isclose(w_mse, w_nll)
    print("\n[5] BCE 최소화 = 베르누이 MLE")
    p_bce, p_lik, _ = check_bce_mle()
    assert np.isclose(p_bce, p_lik)
    print("\n[5] softmax + CE 기울기 = ŷ - y")
    assert check_softmax_ce_grad() < 1e-8
    print("\n[6] (보너스) CE = H + KL")
    assert check_info_theory()
    plt.close("all")


if __name__ == "__main__":
    main()
