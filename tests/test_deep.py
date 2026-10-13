"""L9 深度学习：MLP 反向传播 + 缩放点积注意力，不变量 + 可选 torch 跨库。"""
import numpy as np
import pytest

from morningstar_ai import deep, _random
from ._helpers import all_passed


def test_mlp_regression():
    Xr, yr = _random.make_regression(n_samples=200, n_features=3, noise=0.1, seed=3)
    mlp = deep.MLP(hidden=12, n_iter=4000, lr=0.05, task="regression", seed=0).fit(Xr, yr)
    inv = mlp.check_invariants()
    assert all_passed(inv), inv
    assert mlp.compare_torch(Xr)["passed"], mlp.compare_torch(Xr)


def test_mlp_classification():
    Xc, yc = _random.make_classification(n_samples=300, n_features=4, n_classes=2, seed=1)
    mlp = deep.MLP(hidden=10, n_iter=3000, lr=0.05, task="classification", seed=1).fit(Xc, yc)
    assert all_passed(mlp.check_invariants()), mlp.check_invariants()


def test_attention_invariants():
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(4, 3, 5))
    K = rng.normal(size=(4, 3, 5))
    V = rng.normal(size=(4, 3, 6))
    att = deep.Attention()
    assert all_passed(att.check_invariants(Q, K, V)), att.check_invariants(Q, K, V)


def test_attention_cross_torch():
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(2, 4, 5))
    K = rng.normal(size=(2, 4, 5))
    V = rng.normal(size=(2, 4, 6))
    att = deep.Attention()
    assert att.compare_torch(Q, K, V)["passed"], att.compare_torch(Q, K, V)
