"""L4–L6 监督学习：不变量 + 跨库（sklearn）交叉验证。"""
import numpy as np
import pytest

from morningstar_ai import supervised, _random
from ._helpers import all_passed


def _data():
    Xc, yc = _random.make_classification(n_samples=300, n_features=4, n_classes=2, seed=1)
    Xr, yr = _random.make_regression(n_samples=200, n_features=3, noise=0.1, seed=3)
    return Xc, yc, Xr, yr


def test_linear_regression():
    Xc, yc, Xr, yr = _data()
    lin = supervised.LinearRegression().fit(Xr, yr)
    assert all_passed(lin.check_invariants())
    assert lin.compare_numpy(Xr, yr)["passed"]


def test_logistic_regression():
    Xc, yc, Xr, yr = _data()
    lr = supervised.LogisticRegression(n_iter=4000, lr=0.3).fit(Xc, yc)
    assert all_passed(lr.check_invariants())
    assert lr.compare_sklearn(Xc, yc)["passed"]


def test_gaussian_nb():
    Xc, yc, Xr, yr = _data()
    nb = supervised.GaussianNB().fit(Xc, yc)
    assert nb.compare_sklearn(Xc, yc)["passed"]


def test_knn():
    Xc, yc, Xr, yr = _data()
    kn = supervised.KNeighborsClassifier(5).fit(Xc, yc)
    assert all_passed(kn.check_invariants())
    assert kn.compare_sklearn(Xc, yc)["passed"]


def test_svc():
    Xc, yc, Xr, yr = _data()
    sv = supervised.SVC(C=1.0, kernel="linear").fit(Xc, yc)
    assert all_passed(sv.check_invariants())
    assert sv.compare_sklearn(Xc, yc)["passed"]


def test_svc_rbf_consistency():
    """RBF 核下与 sklearn 预测仍高度一致（集大成：复用 sklearn 作为黄金参照）。"""
    Xc, yc, Xr, yr = _data()
    sv = supervised.SVC(C=1.0, kernel="rbf", gamma=0.5).fit(Xc, yc)
    assert sv.compare_sklearn(Xc, yc)["passed"]
