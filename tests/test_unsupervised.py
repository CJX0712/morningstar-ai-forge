"""L7–L8 无监督学习：不变量 + 跨库（sklearn）交叉验证。"""
import numpy as np
import pytest

from morningstar_ai import unsupervised, _random
from ._helpers import all_passed


def _blobs():
    return _random.make_blobs(n_samples=300, n_features=3, centers=3, seed=2)


def test_kmeans():
    Xb, yb = _blobs()
    km = unsupervised.KMeans(3, seed=2).fit(Xb)
    assert all_passed(km.check_invariants())
    assert km.compare_sklearn(Xb, yb)["passed"]


def test_pca():
    Xb, yb = _blobs()
    pca = unsupervised.PCA(2).fit(Xb)
    assert all_passed(pca.check_invariants())
    assert pca.compare_sklearn(Xb)["passed"]


def test_pca_eckart_young():
    """Eckart-Young：秩-k 截断重构误差 = 理论丢弃能量（含 (N-1)/(N·d) 因子）。"""
    Xb, yb = _blobs()
    pca = unsupervised.PCA(1).fit(Xb)
    assert all_passed(pca.check_invariants())


def test_gmm():
    Xb, yb = _blobs()
    gmm = unsupervised.GaussianMixture(3, seed=2).fit(Xb)
    assert all_passed(gmm.check_invariants())
    assert gmm.compare_sklearn(Xb)["passed"]
