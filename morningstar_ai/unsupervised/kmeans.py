"""K-Means —— forge Lloyd 算法 + 不变量。

不变量：
  * SSE（簇内平方和）随迭代单调非增
  * 收敛后簇分配稳定（不变）
  * 跨库校验：与 sklearn.KMeans 在已知 blob 上 ARI 高、inertia 接近
"""
from __future__ import annotations

import numpy as np


def _kmeans_plus_plus(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    return np.array(centers)


class KMeans:
    def __init__(self, n_clusters=3, n_iter=300, seed=0, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.seed = seed
        self.tol = tol

    def _assign(self, X, C):
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        C = _kmeans_plus_plus(X, self.n_clusters, rng)
        self.inertia_hist_ = []
        labels = None
        for _ in range(self.n_iter):
            labels = self._assign(X, C)
            newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                             else C[j] for j in range(self.n_clusters)])
            shift = np.linalg.norm(newC - C)
            C = newC
            d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            inertia = float(np.sum(d[np.arange(len(X)), labels] ** 2))
            self.inertia_hist_.append(inertia)
            if shift < self.tol:
                break
        self.cluster_centers_ = C
        self.labels_ = labels
        self.inertia_ = self.inertia_hist_[-1]
        return self

    def predict(self, X):
        return self._assign(np.asarray(X, dtype=np.float64), self.cluster_centers_)

    def check_invariants(self) -> dict:
        hist = np.array(self.inertia_hist_)
        mono = bool(np.all(np.diff(hist) <= 1e-8))
        return {"sse_monotone_non_increasing": {"value": mono, "passed": mono},
                "final_inertia": self.inertia_}

    def compare_sklearn(self, X, y=None) -> dict:
        from sklearn.cluster import KMeans as SKKM
        from sklearn.metrics import adjusted_rand_score
        X = np.asarray(X, dtype=np.float64)
        sk = SKKM(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(X)
        our = self.predict(X)
        res = {"our_inertia": self.inertia_,
               "sklearn_inertia": float(sk.inertia_),
               "passed": bool(abs(self.inertia_ - sk.inertia_) < 1e-6 * max(1, sk.inertia_))}
        if y is not None:
            res["ARI_vs_true"] = float(adjusted_rand_score(y, our))
            res["ARI_sklearn_vs_true"] = float(adjusted_rand_score(y, sk.labels_))
        return res
