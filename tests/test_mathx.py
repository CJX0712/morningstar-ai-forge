"""L1 线性代数 + L3 优化：自研 forge 原语的「不变量」单元测试。

零重写 / 集大成：稠密代数复用 numpy，但此处手写那些能用一条可验证
不变量独立证明正确性的关键原语，并用不变量 + 跨库（numpy）对照自证。
"""
import numpy as np
import pytest

from morningstar_ai.mathx.linalg import (
    cholesky, cholesky_inverse, jacobi_eigen, logsumexp, softmax,
    log_softmax, orthogonality_error, eigen_residual, is_symmetric_psd,
)
from morningstar_ai.mathx.optimize import (
    numerical_gradient, gradient_check, gradient_descent, check_monotone_decrease,
)


def _spd(n, seed=0):
    """构造对称正定阵 A = M Mᵀ + n I。"""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


# ---------------- L1 线性代数 ----------------
def test_cholesky_residual():
    """Cholesky 不变量：A = L Lᵀ ⇒ ‖A − L Lᵀ‖_F ≈ 0。"""
    A = _spd(6, seed=1)
    L = cholesky(A)
    res = np.linalg.norm(A - L @ L.T)
    assert res < 1e-10


def test_cholesky_inverse():
    """逆矩阵不变量：A⁻¹ A = I。"""
    A = _spd(5, seed=2)
    L = cholesky(A)
    Ainv = cholesky_inverse(L)
    assert np.linalg.norm(Ainv @ A - np.eye(5)) < 1e-10


def test_jacobi_eigen_vs_numpy():
    """Jacobi 特征值分解：特征值 / 特征向量 / 残差 三重不变量对照 numpy。"""
    rng = np.random.default_rng(3)
    for n in (2, 5, 40):
        A = rng.standard_normal((n, n))
        A = A @ A.T
        ev, V, info = jacobi_eigen(A)
        ev_ref, _ = np.linalg.eigh(A)
        # 特征值一致（机器精度）
        assert np.max(np.abs(ev - ev_ref)) < 1e-9
        # 特征向量正交归一
        assert orthogonality_error(V) < 1e-8
        # 残差 ‖A vᵢ − λᵢ vᵢ‖ ≈ ε
        rel = max(np.linalg.norm(A @ V[:, i] - ev[i] * V[:, i]) / abs(ev[i])
                  for i in range(n))
        assert rel < 1e-7
        assert info["off_diag"] < 1e-9


def test_jacobi_degenerate_a_pp_eq_a_qq():
    """退化情形 a_pp == a_qq：必须取 φ=π/4 而非冻结。"""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    ev, V, _ = jacobi_eigen(A)
    assert np.allclose(np.sort(ev), [1.0, 3.0], atol=1e-10)


def test_logsumexp_stability():
    """logsumexp 数值稳定：大偏移量下与暴力 log(sum(exp)) 一致但不溢出。"""
    x = np.array([1000.0, 1000.0 - 1e-6, 999.0])
    ls = logsumexp(x)
    # 直接算会溢出，用移位法对照
    m = x.max()
    ref = m + np.log(np.sum(np.exp(x - m)))
    assert abs(ls - ref) < 1e-10


def test_softmax_row_stochastic():
    """softmax 不变量：每行和为 1，且对常数平移不变。"""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 7))
    P = softmax(X, axis=1)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)
    # 平移不变
    P2 = softmax(X + 3.0, axis=1)
    assert np.allclose(P, P2, atol=1e-12)


def test_log_softmax_identity():
    """log_softmax(x) = x − logsumexp(x)（按行）。"""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((3, 5))
    ls = log_softmax(X, axis=1)
    ref = X - logsumexp(X, axis=1, keepdims=True)
    assert np.allclose(ls, ref, atol=1e-12)


def test_is_symmetric_psd():
    psd, _ = is_symmetric_psd(_spd(4, seed=7))
    assert psd is True
    not_psd, _ = is_symmetric_psd(-np.eye(3))
    assert not_psd is False


# ---------------- L3 优化 ----------------
def test_gradient_check_quadratic():
    """梯度检验：解析梯度 vs 中心差分数值梯度，maxRelErr ≈ 1e-7。"""
    # f(w) = wᵀ Q w + bᵀ w，解析梯度 2 Q w + b
    rng = np.random.default_rng(8)
    Q = rng.standard_normal((5, 5)); Q = Q @ Q.T + np.eye(5)
    b = rng.standard_normal(5)
    f = lambda w: float(w @ Q @ w + b @ w)
    g = lambda w: 2 * Q @ w + b
    w0 = rng.standard_normal(5)
    rep = gradient_check(f, g, w0)
    assert rep["passed"]
    assert rep["max_rel_err"] < 1e-5


def test_gradient_descent_monotone():
    """梯度下降不变量：凸二次目标损失单调非增。"""
    Q = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, -1.0])
    f = lambda w: float(w @ Q @ w + b @ w)
    g = lambda w: 2 * Q @ w + b
    out = gradient_descent(f, g, np.array([2.0, 2.0]), lr=0.1, n_iter=500)
    assert check_monotone_decrease(out["f_hist"])
    # 收敛到真实极小值 w* = −½ Q⁻¹ b
    wstar = -0.5 * np.linalg.solve(Q, b)
    assert np.linalg.norm(out["x"] - wstar) < 1e-4
