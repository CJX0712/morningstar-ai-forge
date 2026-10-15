# MorningStar-AI · 世界顶级 AI 系统底座

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/CJX0712/morningstar-ai-forge/blob/main/LICENSE) [![Language](https://img.shields.io/badge/Python-3776AB?logo=python)](https://github.com/CJX0712/morningstar-ai-forge) [![Verified](https://img.shields.io/badge/invariants-verified-brightgreen)](https://github.com/CJX0712) [![Author](https://img.shields.io/badge/author-晨星-blueviolet)](https://github.com/CJX0712)

</div>


> 作者：**晨星 (MorningStar)**  ·  License：[MIT](LICENSE)  ·  Homepage：[github.com/CJX0712/morningstar-ai-forge](https://github.com/CJX0712/morningstar-ai-forge)

晨明星 —— 天亮前最后还亮着的那颗。本项目以「**零重写、集大成、可验证、闭环交付**」为唯一信条，锻造一座从数学基座到深度学习的完整 AI 系统：

- **零重写**：数值重活（稠密线性代数 / QP / EM / 聚类）**直接复用世界顶级库** `numpy` / `scipy` / `scikit-learn`（可选 `torch`），绝不从零造轮子。
- **集大成**：另有一层 **forge 纯实现**，手写那些「能用一条可验证不变量独立证明正确性」的关键原语，并**以数学不变量 + 跨库交叉验证**证明其与世界顶级库逐位一致。
- **可验证**：每个算法至少有一条**可被独立实现交叉验证的硬不变量**（解析 vs 数值、自研 vs 库参照、EM/SSE 单调性、KKT、残差→0），全部 **10/10 模块不变量全绿 + 10/10 跨库一致**。
- **闭环交付**：成果完整开源，可运行、可复现、可验证。

---

## 十层架构

| 层 | 领域 | 模块 | 世界级参照 |
|----|------|------|-----------|
| **L1** | 线性代数 | `morningstar_ai.mathx.linalg` | numpy / LAPACK |
| **L3** | 微积分与优化 | `morningstar_ai.mathx.optimize` | 链式法则、梯度检验 |
| **L4–6** | 监督学习 | `morningstar_ai.supervised.*` | scikit-learn |
| **L7–8** | 无监督学习 | `morningstar_ai.unsupervised.*` | scikit-learn |
| **L9** | 深度学习 | `morningstar_ai.deep.*` | PyTorch（可选） |
| **L10** | 集成验证 | `morningstar_ai.benchmarks` | 不变量 + 跨库闭环 |

### 模块与不变量矩阵

| 模块 | 核心算法 | 数学不变量（硬标准） | 跨库参照 |
|------|---------|---------------------|---------|
| **Cholesky** | A = L Lᵀ | `‖A − L Lᵀ‖_F ≈ 0`；`A⁻¹A = I` | numpy |
| **Jacobi 特征分解** | 循环旋转对角化 | `‖VᵀV − I‖_F ≈ ε`；残差 `‖Avᵢ − λᵢvᵢ‖/|λᵢ| ≈ ε` | `numpy.linalg.eigh` |
| **logsumexp / softmax** | 数值稳定 | 不溢出；softmax 行随机、平移不变 | numpy |
| **LinearRegression** | 正规方程（Cholesky） | 残差 ⊥ X：`Xᵀ(y − Xŵ) = 0` | sklearn |
| **LogisticRegression** | 梯度下降 + logsumexp | 梯度检验 `maxRelErr ≈ 1e-7`；损失单调非增 | sklearn |
| **GaussianNB** | 高斯朴素贝叶斯 | 后验和 = 1 | sklearn |
| **KNeighbors** | k-NN | k=1 训练误差 = 0（过拟合边界） | sklearn |
| **SVC** | 对偶 QP（scipy SLSQP）+ SMO | KKT 互补松弛；支持向量 `yᵢf(xᵢ) = 1` | sklearn |
| **KMeans** | Lloyd + k-means++ | SSE 单调非增 | sklearn |
| **PCA** | 协方差 + Jacobi 特征 | 协方差对称半正定；特征向量正交；Eckart-Young 最优 | sklearn |
| **GMM** | EM | 对数似然单调非减；责任度行和 = 1 | sklearn |
| **MLP** | 反向传播 | 梯度检验 `maxRelErr ≈ 1e-7`；损失单调非增 | PyTorch |
| **Attention** | 缩放点积注意力 | 权重行随机；自洽；大输入数值稳定 | PyTorch |

---

## 快速开始

### 安装

```bash
pip install -e .            # 核心依赖 numpy/scipy/scikit-learn
pip install -e ".[dev]"     # + pytest / torch（跨库校验）
```

### 一键自检：不变量 + 跨库验证

```bash
python -m morningstar_ai.benchmarks.run
```

预期输出（全部 ✅，`ALL_PASS: True`）：

```
=== MorningStar-AI · 不变量 + 跨库验证 ===
  LinearRegression           不变量 ✅  跨库 ✅
  LogisticRegression         不变量 ✅  跨库 ✅
  GaussianNB                 不变量 ✅  跨库 ✅
  KNeighbors(k=5)            不变量 ✅  跨库 ✅
  SVC(linear)                不变量 ✅  跨库 ✅
  KMeans                     不变量 ✅  跨库 ✅
  PCA                        不变量 ✅  跨库 ✅
  GMM(EM)                    不变量 ✅  跨库 ✅
  MLP(regression)            不变量 ✅  跨库 ✅
  Attention(SDPA)            不变量 ✅  跨库 ✅

总计 10 模块 | 不变量全绿 10/10 | 跨库一致 10/10
```

### 测试

```bash
pytest -q
```

### 用法示例

```python
import numpy as np
from morningstar_ai import supervised, unsupervised, deep

# 监督学习
clf = supervised.LogisticRegression(n_iter=4000, lr=0.3).fit(X, y)
svc = supervised.SVC(C=1.0, kernel="linear").fit(X, y)

# 无监督学习
pca = unsupervised.PCA(2).fit(X)          # 降维
km  = unsupervised.KMeans(3, seed=0).fit(X)
gmm = unsupervised.GaussianMixture(3).fit(X)

# 深度学习
mlp = deep.MLP(hidden=12, task="regression").fit(X, y)

# 每个模块都自带 check_invariants() 与 compare_<库>() 自证
clf.check_invariants(); clf.compare_sklearn(X, y)
```

---

## 目录结构

```
morningstar-ai/
├── morningstar_ai/
│   ├── mathx/          # L1 线性代数 / L3 优化（forge 纯实现 + 不变量）
│   ├── supervised/     # L4–6 监督学习
│   ├── unsupervised/   # L7–8 无监督学习
│   ├── deep/           # L9 深度学习（MLP / Attention）
│   ├── benchmarks/     # L10 集成验证编排
│   └── _random.py      # 可复现随机数据源
├── tests/              # pytest 测试套件（25 用例全绿）
├── docs/
│   ├── ARCHITECTURE.md # 十层架构详解
│   └── MATH.md         # 数学不变量手册
├── .github/workflows/  # CI（多 Python 版本 pytest）
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

更多细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 与 [`docs/MATH.md`](docs/MATH.md)。

---

## 设计信条

> **不靠信仰，靠可复现的硬数据。** 每一行自研代码都必须能用一条可被独立实现交叉验证的不变量证明其正确性；凡有更强的现成方案，立即替换自写。

* **数学不变量**：残差 → 0、正交性 → I、SSE 单调非增、EM 对数似然单调非减、KKT 互补松弛、Eckart-Young 最优、`(N-1)/N` 因子。
* **跨库交叉验证**：自研 forge 模块 vs `numpy` / `scikit-learn` / `PyTorch` 逐位对照。
* **可复现**：所有随机源封装于 `_random.py`，seed 固定，结果可复现。

## License

MIT © 2026 晨星 (MorningStar) — 详见 [LICENSE](LICENSE)。