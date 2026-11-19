"""2층 신경망(입력 2 → 은닉 2 → 출력 1) 역전파: 고정 예제, 손계산 기록, NumPy 검증.

노트북 `notebooks/backprop_derivation.ipynb`가 이 모듈을 불러 쓴다. 수식 유도는 노트북에,
계산과 검증 코드는 이 파일에 있다. 단독 실행도 된다.

실행:
    uv run python src/backprop.py

고정 예제 (과제 명세)
    x  = [1, 0]                    shape (2,)
    W1 = [[0.1, 0.2], [0.3, 0.4]]  shape (2, 2)
    b1 = [0, 0]                    shape (2,)
    W2 = [0.5, 0.6]                shape (2,)
    b2 = 0                         scalar
    y  = 1 (정답)                  scalar
    활성화: sigmoid, 손실: BCE

손계산 기록 (중간값은 소수 6자리로 계산하고 결과는 소수 4자리로 반올림)
    순전파
      z1 = W1 x + b1   = [0.1×1 + 0.2×0, 0.3×1 + 0.4×0]               = [0.1000, 0.3000]   (2,)
      a1 = σ(z1)       = [1/(1+e^-0.1), 1/(1+e^-0.3)]
                       = [1/1.904837, 1/1.740818] = [0.524979, 0.574443] → [0.5250, 0.5744]   (2,)
      z2 = W2·a1 + b2  = 0.5×0.524979 + 0.6×0.574443 = 0.607155       → 0.6072             scalar
      ŷ  = σ(z2)       = 1/(1+e^-0.607155) = 1/1.544918 = 0.647292     → 0.6473             scalar
      L  = -log(ŷ)     = -log(0.647292) = 0.434958                     → 0.4350             scalar
    역전파 (출력층 → 입력층, 연쇄 법칙)
      ① ∂L/∂ŷ  = -y/ŷ + (1-y)/(1-ŷ) = -1/0.647292                    → -1.5449            scalar
      ② ∂L/∂z2 = ∂L/∂ŷ × σ'(z2) = ŷ - y = 0.647292 - 1                → -0.3527            scalar
      ③ ∂L/∂W2 = ∂L/∂z2 × a1 = -0.352708 × [0.524979, 0.574443]      → [-0.1852, -0.2026] (2,)
         ∂L/∂b2 = ∂L/∂z2                                              → -0.3527            scalar
      ④ ∂L/∂a1 = ∂L/∂z2 × W2 = -0.352708 × [0.5, 0.6]                → [-0.1764, -0.2116] (2,)
      ⑤ σ'(z1) = a1(1-a1) = [0.249376, 0.244458]
         ∂L/∂z1 = ∂L/∂a1 ⊙ σ'(z1) = [-0.176354×0.249376, -0.211625×0.244458]
                                                                      → [-0.0440, -0.0517] (2,)
      ⑥ ∂L/∂W1 = outer(∂L/∂z1, x)                                     → [[-0.0440, 0], [-0.0517, 0]] (2, 2)
         ∂L/∂b1 = ∂L/∂z1                                              → [-0.0440, -0.0517] (2,)

    과제 PDF 예시와 다른 점: z2는 0.6072(PDF 0.6071), ∂L/∂z1[1]은 -0.0517(PDF -0.0513)이다.
    0.6071은 a1을 4자리로 먼저 반올림했을 때 나오는 값이다. NumPy 계산과 중심차분 검증이
    모두 위 손계산 값을 가리킨다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculus import numerical_gradient  # noqa: E402

np.random.seed(42)

# 고정 예제
X = np.array([1.0, 0.0])                  # (2,)
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])   # (2, 2)
B1 = np.array([0.0, 0.0])                 # (2,)
W2 = np.array([0.5, 0.6])                 # (2,)
B2 = 0.0                                  # scalar
Y_TRUE = 1.0                              # scalar

# 손계산 결과 (소수 4자리 반올림). 위 docstring의 손계산 기록과 같은 값이다.
HAND_RESULTS = {
    "z1": [0.1000, 0.3000], "a1": [0.5250, 0.5744], "z2": 0.6072, "y_pred": 0.6473, "loss": 0.4350,
    "dL/dy_pred": -1.5449, "dL/dz2": -0.3527, "dL/dW2": [-0.1852, -0.2026], "dL/db2": -0.3527,
    "dL/da1": [-0.1764, -0.2116], "dL/dz1": [-0.0440, -0.0517],
    "dL/dW1": [[-0.0440, 0.0], [-0.0517, 0.0]], "dL/db1": [-0.0440, -0.0517],
}

PARAM_NAMES = ["W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]", "b1[0]", "b1[1]", "W2[0]", "W2[1]", "b2"]


def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z)). 배열이면 원소별로 적용한다."""
    return 1.0 / (1.0 + np.exp(-z))


def forward(x, W1, b1, W2, b2, y):
    """2층 신경망 순전파.

    Args:
        x: 입력 (2,). W1: (2, 2). b1: (2,). W2: (2,). b2: scalar. y: 정답 scalar.

    Returns:
        중간값 dict: z1 (2,), a1 (2,), z2 (scalar), y_pred (scalar), loss (scalar).
    """
    z1 = W1 @ x + b1                 # (2,2) @ (2,) + (2,) -> (2,)
    a1 = sigmoid(z1)                 # (2,)
    z2 = W2 @ a1 + b2                # (2,) · (2,) + scalar -> scalar
    y_pred = sigmoid(z2)             # scalar
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))  # BCE, scalar
    return {"z1": z1, "a1": a1, "z2": z2, "y_pred": y_pred, "loss": loss}


def backward(x, W2, y, cache):
    """2층 신경망 역전파. 손계산 기록의 ①~⑥을 그대로 옮긴다.

    Args:
        x: 입력 (2,). W2: (2,). y: 정답 scalar. cache: forward()의 반환값.

    Returns:
        기울기 dict. 각 기울기의 shape은 대응하는 변수의 shape과 같다.
    """
    a1, y_pred = cache["a1"], cache["y_pred"]
    dL_dy_pred = -y / y_pred + (1 - y) / (1 - y_pred)   # ① scalar
    dL_dz2 = y_pred - y                                  # ② scalar  (= ① × σ'(z2))
    dL_dW2 = dL_dz2 * a1                                 # ③ (2,)
    dL_db2 = dL_dz2                                      # ③ scalar
    dL_da1 = dL_dz2 * W2                                 # ④ (2,)
    dL_dz1 = dL_da1 * a1 * (1 - a1)                      # ⑤ (2,)  원소별 곱
    dL_dW1 = np.outer(dL_dz1, x)                         # ⑥ (2,2)
    dL_db1 = dL_dz1                                      # ⑥ (2,)
    return {"dL/dy_pred": dL_dy_pred, "dL/dz2": dL_dz2, "dL/dW2": dL_dW2, "dL/db2": dL_db2,
            "dL/da1": dL_da1, "dL/dz1": dL_dz1, "dL/dW1": dL_dW1, "dL/db1": dL_db1}


def run_fixed_example(verbose=True):
    """고정 예제로 순전파와 역전파를 계산한다.

    Returns:
        (cache, grads): 순전파 중간값 dict, 기울기 dict.
    """
    cache = forward(X, W1, B1, W2, B2, Y_TRUE)
    grads = backward(X, W2, Y_TRUE, cache)
    if verbose:
        print("순전파")
        for k, v in cache.items():
            print(f"  {k:<7} shape={str(np.shape(v)):<6} = {np.round(v, 4) + 0.0}")  # + 0.0: -0.0 표시 방지
        print("역전파")
        for k, v in grads.items():
            print(f"  {k:<11} shape={str(np.shape(v)):<6} = {(np.round(v, 4) + 0.0).tolist()}")
    return cache, grads


def compare_with_hand(computed, hand=HAND_RESULTS, decimals=4, verbose=True):
    """NumPy 계산 결과를 손계산 결과와 소수 decimals자리까지 비교한다.

    Args:
        computed: 순전파 중간값과 기울기를 합친 dict.
        hand: 손계산 결과 dict (기본: HAND_RESULTS).
        decimals: 비교할 소수 자릿수 (기본 4).

    Returns:
        모든 값이 일치하면 True.
    """
    all_ok = True
    for k, h in hand.items():
        mine = np.round(computed[k], decimals) + 0.0      # +0.0: -0.0을 0.0으로
        ok = np.array_equal(mine, np.array(h) + 0.0)
        all_ok &= ok
        if verbose:
            print(f"{k:<11} 손계산 {str(h):<30} NumPy {str(mine.tolist()):<30} {'일치' if ok else '불일치'}")
    if verbose:
        print("\n결과:", f"모든 값이 소수점 {decimals}자리까지 일치" if all_ok else "불일치 항목 있음")
    return all_ok


def flatten_params(W1, b1, W2, b2):
    """파라미터를 한 줄 벡터(W1 4개, b1 2개, W2 2개, b2 1개 = 9개)로 펼친다."""
    return np.concatenate([np.ravel(W1), b1, W2, [b2]])


def loss_of(params, x=X, y=Y_TRUE):
    """한 줄로 펼친 파라미터 9개를 받아 손실을 돌려준다 (중심차분용)."""
    return forward(x, params[:4].reshape(2, 2), params[4:6], params[6:8], params[8], y)["loss"]


def gradient_check(grads, verbose=True):
    """역전파 기울기를 src/calculus.py의 중심차분 기울기와 비교한다.

    Returns:
        상대 오차 ||g_역전파 - g_중심차분|| / (||g_역전파|| + ||g_중심차분||).
    """
    theta = flatten_params(W1, B1, W2, B2)
    g_numeric = numerical_gradient(loss_of, theta)
    g_backprop = flatten_params(grads["dL/dW1"], grads["dL/db1"], grads["dL/dW2"], grads["dL/db2"])
    rel = np.linalg.norm(g_backprop - g_numeric) / (np.linalg.norm(g_backprop) + np.linalg.norm(g_numeric))
    if verbose:
        for n, gb, gn in zip(PARAM_NAMES, g_backprop, g_numeric):
            print(f"{n:<8} 역전파 {gb:+.8f}   중심차분 {gn:+.8f}   차이 {abs(gb - gn):.1e}")
        print(f"\n상대 오차 = {rel:.2e} → {'역전파 공식 검증 완료' if rel < 1e-7 else '확인 필요'}")
    return rel


def main():
    print("=" * 60)
    print("[1] 고정 예제 순전파 / 역전파 (NumPy)")
    print("=" * 60)
    cache, grads = run_fixed_example()
    print()
    print("=" * 60)
    print("[2] 손계산과 NumPy 결과 비교 (소수 4자리)")
    print("=" * 60)
    assert compare_with_hand({**cache, **grads})
    print()
    print("=" * 60)
    print("[3] 중심차분 gradient check")
    print("=" * 60)
    assert gradient_check(grads) < 1e-7


if __name__ == "__main__":
    main()
