"""선형대수 모듈: 행렬 변환 시각화, Power Iteration, SVD 이미지 압축.

수식 요약
    회전:       R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]      det = 1
    스케일링:   S(sx, sy) = [[sx, 0], [0, sy]]                det = sx · sy
    전단:       Sh(k) = [[1, k], [0, 1]]   (x' = x + k·y)     det = 1
    넓이 비율:  (변환 후 넓이) / (변환 전 넓이) = |det(A)|
    신발끈 공식: 넓이 = ½ |Σ (xᵢ·yᵢ₊₁ - xᵢ₊₁·yᵢ)|
    Power Iteration: v ← A v / ‖A v‖,  λ ≈ vᵀ A v  (레일리 몫)
    SVD:        A = U Σ Vᵀ,  rank-k 근사 A_k = Σ_{i=1..k} σᵢ uᵢ vᵢᵀ

np.linalg.eig 는 Power Iteration 결과를 검증하는 데에만 사용한다.

실행:
    uv run python src/linear_algebra.py
"""

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

from plot_style import BLUE, GRAY, GREEN, ORANGE, RED, plain_log_ticks, save_figure, setup_style

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. 행렬 변환
# ---------------------------------------------------------------------------

def rotation_matrix(theta):
    """θ(라디안)만큼 반시계 방향으로 돌리는 회전 행렬 R(θ)를 만든다.

    R(θ) = [[cos θ, -sin θ],
            [sin θ,  cos θ]]

    Args:
        theta: 회전 각도 (라디안).

    Returns:
        shape (2, 2) 배열.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling_matrix(sx, sy):
    """x축으로 sx배, y축으로 sy배 늘이는 스케일링 행렬 S(sx, sy)를 만든다.

    S(sx, sy) = [[sx, 0],
                 [0, sy]]

    Args:
        sx: x 방향 배율.
        sy: y 방향 배율.

    Returns:
        shape (2, 2) 배열.
    """
    return np.array([[sx, 0.0], [0.0, sy]])


def shear_matrix(k):
    """y값에 비례해 x를 밀어 주는 전단 행렬 Sh(k)를 만든다.

    Sh(k) = [[1, k],
             [0, 1]]     →  x' = x + k·y,  y' = y

    Args:
        k: 전단 계수.

    Returns:
        shape (2, 2) 배열.
    """
    return np.array([[1.0, k], [0.0, 1.0]])


def unit_circle(n=100):
    """단위 원 위의 점 n개를 만든다 (과제 안내: np.linspace(0, 2π, 100)).

    Args:
        n: 점의 개수.

    Returns:
        shape (2, n) 배열. 각 열이 점 하나 (x, y).
    """
    theta = np.linspace(0, 2 * np.pi, n)
    return np.vstack([np.cos(theta), np.sin(theta)])


def polygon_area(points):
    """신발끈 공식으로 다각형의 넓이를 구한다.

    넓이 = ½ |Σ (xᵢ·yᵢ₊₁ - xᵢ₊₁·yᵢ)|   (마지막 점은 첫 점과 이어진다)

    Args:
        points: shape (2, n) 배열. 꼭짓점이 순서대로 들어 있어야 한다.

    Returns:
        넓이 (float).
    """
    x, y = points
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def check_area_ratio(A, n=100):
    """det(A)와 '변환 후 넓이 / 변환 전 넓이'를 비교한다.

    Args:
        A: shape (2, 2) 변환 행렬.
        n: 단위 원을 근사할 점의 개수.

    Returns:
        (det, 넓이 비율, 상대 오차) 튜플. 상대 오차 = |비율 - |det|| / |det|.
    """
    circle = unit_circle(n)
    det = np.linalg.det(A)
    ratio = polygon_area(A @ circle) / polygon_area(circle)
    error = abs(ratio - abs(det)) / abs(det)
    return det, ratio, error


def plot_transform(A, title, name, save=True):
    """단위 원과 격자에 행렬 A를 적용해 변환 전후를 한 Figure에 겹쳐 그린다.

    - 회색 점: 변환 전 격자 (np.meshgrid), 주황 점: 변환 후 격자
    - 파란 원: 변환 전 단위 원, 빨간 도형: 변환 후
    - 화살표: 기준 벡터 î=(1,0), ĵ=(0,1)이 도착한 곳 = A의 열

    Args:
        A: shape (2, 2) 변환 행렬.
        title: 그림 제목.
        name: 저장할 파일 이름 (확장자 제외).
        save: True면 outputs/<name>.png 로 저장한다.

    Returns:
        (Figure, det, 넓이 비율, 상대 오차) 튜플.
    """
    circle = unit_circle()
    moved = A @ circle
    det, ratio, error = check_area_ratio(A)

    gx, gy = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    grid = np.vstack([gx.ravel(), gy.ravel()])
    grid_moved = A @ grid

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(*grid, s=6, color=GRAY, alpha=0.5, label="변환 전 격자")
    ax.scatter(*grid_moved, s=6, color=ORANGE, alpha=0.6, label="변환 후 격자")
    ax.plot(*circle, color=BLUE, lw=2, label="변환 전: 단위 원")
    ax.fill(*moved, color=RED, alpha=0.12)
    ax.plot(*moved, color=RED, lw=2, label="변환 후")
    for col, c, lab in [(A[:, 0], RED, r"$A\hat{\imath}$"), (A[:, 1], GREEN, r"$A\hat{\jmath}$")]:
        ax.annotate("", xy=col, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.text(col[0] * 1.08, col[1] * 1.08, f"{lab}=({col[0]:.2f}, {col[1]:.2f})",
                color=c, fontsize=8)
    lim = max(2.3, np.abs(moved).max() * 1.15)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
           title=f"{title}\ndet = {det:.4f}, 넓이 비율 = {ratio:.4f}, 오차 = {error:.2%}")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.legend(fontsize=8, loc="lower right")
    print(f"[{title}] det = {det:.4f}, 넓이 비율 = {ratio:.4f}, "
          f"오차 = {error:.4%} → {'통과' if error < 0.01 else '실패'} (기준 1%)")
    if save:
        save_figure(fig, name)
    return fig, det, ratio, error


# ---------------------------------------------------------------------------
# 2. Power Iteration
# ---------------------------------------------------------------------------

def power_iteration(A, num_iters=1000, tol=1e-6):
    """Power Iteration으로 절댓값이 가장 큰 고유값과 고유벡터를 구한다.

    알고리즘
        1. 무작위 벡터 v₀ 에서 시작한다 (np.random.seed(42)로 재현 가능).
        2. w = A v_k
        3. v_{k+1} = w / ‖w‖                (정규화: 길이가 폭발/소멸하지 않게)
        4. λ_{k+1} = v_{k+1}ᵀ A v_{k+1}       (레일리 몫)
        5. 벡터 변화량 ‖v_{k+1} - v_k‖ 가 tol 보다 작으면 멈춘다.
           (고유값은 벡터보다 훨씬 빨리 수렴하므로, 벡터 기준으로 멈춰야 고유벡터도 정확하다.)

    Args:
        A: shape (n, n) 정사각 행렬.
        num_iters: 최대 반복 횟수.
        tol: 수렴 판정 기준.

    Returns:
        (고유값, 고유벡터, 반복 횟수, 매 반복의 고유값 추정 리스트) 튜플.
        고유벡터는 길이 1이고, 비교하기 쉽도록 가장 큰 성분이 양수가 되게 부호를 맞춘다.
    """
    A = np.asarray(A, dtype=float)
    v = np.random.rand(A.shape[0])
    v /= np.linalg.norm(v)
    lam = v @ A @ v
    history = [lam]
    for it in range(1, num_iters + 1):
        w = A @ v
        new_v = w / np.linalg.norm(w)
        if new_v @ v < 0:  # 음수 고유값이면 매번 부호가 바뀌므로 방향만 비교한다
            new_v = -new_v
        lam = new_v @ A @ new_v
        history.append(lam)
        converged = np.linalg.norm(new_v - v) < tol
        v = new_v
        if converged:
            break
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return lam, v, it, history


def verify_with_eig(A, lam, v):
    """Power Iteration 결과를 np.linalg.eig 와 비교한다 (검증 전용).

    고유벡터는 부호가 정해져 있지 않으므로(v와 -v 모두 고유벡터) 부호를 맞춘 뒤 비교한다.

    Args:
        A: 원래 행렬.
        lam: Power Iteration으로 구한 고유값.
        v: Power Iteration으로 구한 고유벡터.

    Returns:
        (eig 고유값, eig 고유벡터, 고유값 상대 오차, 고유벡터 오차) 튜플.
        고유벡터 오차 = 1 - |cos(두 벡터 사이 각)|.
    """
    vals, vecs = np.linalg.eig(A)
    i = np.argmax(np.abs(vals))
    ref_lam = vals[i].real
    ref_v = vecs[:, i].real
    if ref_v[np.argmax(np.abs(ref_v))] < 0:
        ref_v = -ref_v
    lam_err = abs(lam - ref_lam) / abs(ref_lam)
    vec_err = 1 - abs(v @ ref_v) / (np.linalg.norm(v) * np.linalg.norm(ref_v))
    return ref_lam, ref_v, lam_err, vec_err


def run_power_iteration_demo(save=True):
    """예시 행렬들에 Power Iteration을 적용하고 eig와 비교, 수렴 과정을 그린다.

    Args:
        save: True면 outputs/linalg_power_iteration.png 로 저장한다.

    Returns:
        Figure.
    """
    mats = {
        "A = [[4, 1], [1, 3]]": np.array([[4.0, 1.0], [1.0, 3.0]]),
        "B = [[2, 1], [1, 2]]": np.array([[2.0, 1.0], [1.0, 2.0]]),
        "C (3×3)": np.array([[6.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 1.0]]),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (label, M), c in zip(mats.items(), [RED, BLUE, GREEN]):
        lam, v, iters, hist = power_iteration(M)
        ref_lam, ref_v, lam_err, vec_err = verify_with_eig(M, lam, v)
        ok = lam_err < 0.05
        print(f"[Power Iteration] {label}")
        print(f"  Power Iteration ({iters}회, tol=1e-6): λ = {lam:.4f}, v = {np.round(v, 4)}")
        print(f"  np.linalg.eig               : λ = {ref_lam:.4f}, v = {np.round(ref_v, 4)}")
        print(f"  고유값 오차 = {lam_err:.2e}, 고유벡터 오차(1-|cos|) = {vec_err:.2e} "
              f"→ {'통과' if ok else '실패'} (기준 5%)")
        ax.plot(hist, "o-", ms=3, color=c, label=f"{label}: λ={lam:.4f} ({iters}회)")
        ax.axhline(ref_lam, color=c, ls=":", lw=1)
    ax.set(xlabel="반복 횟수", ylabel="고유값 추정 (레일리 몫)",
           title="Power Iteration: 추정값이 가장 큰 고유값(점선, eig)으로 수렴")
    ax.legend(fontsize=8)
    if save:
        save_figure(fig, "linalg_power_iteration")
    return fig


# ---------------------------------------------------------------------------
# 3. SVD 이미지 압축
# ---------------------------------------------------------------------------

def load_grayscale_image(size=64):
    """Matplotlib에 포함된 공개 샘플 사진(grace_hopper.jpg)을 size×size 흑백으로 만든다.

    인터넷에서 내려받지 않고, Matplotlib 설치본에 들어 있는 이미지를 쓴다.
    1. 가운데를 정사각형으로 자른다.
    2. 밝기 = 0.299 R + 0.587 G + 0.114 B 로 흑백 변환한다.
    3. 블록 평균으로 size×size 로 줄이고 0~1 범위로 맞춘다.

    Args:
        size: 결과 이미지의 한 변 크기 (과제 조건: 64 이하).

    Returns:
        shape (size, size), 값 범위 [0, 1] 배열.
    """
    path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    rgb = plt.imread(path).astype(float) / 255.0
    h, w, _ = rgb.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    rgb = rgb[top:top + side, left:left + side]
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    block = side // size
    gray = gray[:block * size, :block * size]
    small = gray.reshape(size, block, size, block).mean(axis=(1, 3))
    return (small - small.min()) / (small.max() - small.min())


def svd_compress(image, k):
    """SVD로 이미지를 rank-k 근사해 압축, 복원한다.

    A = U Σ Vᵀ 에서 큰 특이값 k개만 남긴다:  A_k = U[:, :k] · diag(σ₁..σ_k) · Vᵀ[:k, :]

    k가 rank(특이값 개수)보다 크면 남길 성분이 없으므로 rank로 줄인다.
    (64×64 이미지의 rank는 최대 64라서 k = 100은 k = 64와 같다.)

    Args:
        image: shape (m, n) 흑백 이미지.
        k: 남길 특이값 개수.

    Returns:
        dict: reconstructed(복원 이미지), k_used(실제 사용한 k), rel_error(상대 오차
        ‖A - A_k‖_F / ‖A‖_F), storage_ratio(저장할 숫자 개수 비율 k(m+n+1)/(mn)),
        energy(남긴 특이값 제곱의 비율), singular_values.
    """
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    k_used = min(k, len(S))
    approx = U[:, :k_used] @ np.diag(S[:k_used]) @ Vt[:k_used, :]
    m, n = image.shape
    return {
        "reconstructed": np.clip(approx, 0, 1),
        "k_used": k_used,
        "rel_error": np.linalg.norm(image - approx) / np.linalg.norm(image),
        "storage_ratio": k_used * (m + n + 1) / (m * n),
        "energy": np.sum(S[:k_used] ** 2) / np.sum(S**2),
        "singular_values": S,
    }


def plot_svd_compression(image, ks=(10, 50, 100), save=True):
    """원본과 여러 k의 복원 결과를 한 Figure에 비교하고, 특이값 그래프를 함께 그린다.

    Args:
        image: shape (m, n) 흑백 이미지.
        ks: 비교할 k 값들 (과제: 10, 50, 100).
        save: True면 outputs/linalg_svd_compression.png 로 저장한다.

    Returns:
        (Figure, k별 결과 dict 리스트) 튜플.
    """
    results = [svd_compress(image, k) for k in ks]
    S = results[0]["singular_values"]
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, len(ks) + 1, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"원본 {image.shape[0]}×{image.shape[1]}\n(rank = {np.sum(S > 1e-10)})")
    ax.axis("off")
    for j, (k, r) in enumerate(zip(ks, results), start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(r["reconstructed"], cmap="gray", vmin=0, vmax=1)
        note = f" → {r['k_used']} (rank 한계)" if r["k_used"] < k else ""
        warn = " (원본보다 큼)" if r["storage_ratio"] > 1 else ""
        ax.set_title(f"k = {k}{note}\n오차 {r['rel_error']:.1%}, 저장량 {r['storage_ratio']:.0%}{warn}",
                     fontsize=10)
        ax.axis("off")
        print(f"[SVD] k = {k:>3} (사용 {r['k_used']:>2}): 상대 오차 = {r['rel_error']:.4f}, "
              f"저장량 = {r['storage_ratio']:.1%}, 에너지 = {r['energy']:.4f}")
    ax = fig.add_subplot(gs[1, : (len(ks) + 1) // 2])
    ax.semilogy(np.arange(1, len(S) + 1), S, "o-", ms=3, color=BLUE)
    for k, c in zip(ks, [RED, GREEN, ORANGE]):
        if k <= len(S):
            ax.axvline(k, color=c, ls="--", lw=1, label=f"k = {k}")
    plain_log_ticks(ax, "y")
    ax.set(xlabel="순서 i", ylabel=r"특이값 $\sigma_i$ (로그)", title="특이값은 앞쪽이 훨씬 크다")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, (len(ks) + 1) // 2:])
    energy = np.cumsum(S**2) / np.sum(S**2)
    ax.plot(np.arange(1, len(S) + 1), energy, color=RED)
    ax.set(xlabel="남긴 개수 k", ylabel="남긴 에너지 비율", ylim=(min(energy) - 0.01, 1.005),
           title="앞쪽 몇 개만 남겨도 대부분이 보존된다")
    if save:
        save_figure(fig, "linalg_svd_compression")
    return fig, results


def main():
    """과제 요구 사항을 차례대로 실행하고 그림을 outputs/에 저장한다."""
    setup_style()
    plot_transform(rotation_matrix(np.pi / 4), "회전 R(45°)", "linalg_rotation")
    plot_transform(scaling_matrix(2, 0.5), "스케일링 S(2, 0.5)", "linalg_scaling")
    plot_transform(shear_matrix(1.0), "전단 Sh(1)", "linalg_shear")
    # 위 세 행렬은 모두 det = 1 이라, det ≠ 1 인 행렬로도 넓이 비율을 확인한다.
    plot_transform(np.array([[2.0, 1.0], [0.0, 1.5]]), "일반 행렬 [[2, 1], [0, 1.5]]", "linalg_general")
    run_power_iteration_demo()
    plot_svd_compression(load_grayscale_image(64))


if __name__ == "__main__":
    main()
