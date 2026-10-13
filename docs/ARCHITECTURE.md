# ARCHITECTURE · 十层分解架构

MorningStar-AI 以**十层分解**组织一座完整 AI 系统。核心哲学是**零重写 + 集大成**：

```
基础数学（线代 / 概率 / 优化 / 数值 / 信息论）
        │
        ├─→ 监督学习   X → y      （分类 / 回归）
        ├─→ 无监督学习 X → 结构    （聚类 / 降维 / 密度估计）
        ├─→ 强化学习   状态→动作→奖励（决策 / 控制）   [规划中]
        ├─→ 概率图模型 变量→条件依赖（推断 / 生成）      [规划中]
        │
        └─→ 深度学习（神经网络）──→ 序列 / 生成 / 视觉 ──→ 大语言模型 / 多模态 [规划中]
```

## 分层职责与复用策略

| 层 | 职责 | 实现策略 |
|----|------|---------|
| **L1** 线性代数 | 矩阵分解、特征分解、稳定数值原语 | 稠密代数复用 `numpy`（BLAS/LAPACK）；`cholesky` / `jacobi_eigen` / `logsumexp` 为 forge 纯实现，用不变量自证 |
| **L3** 微积分与优化 | 梯度、梯度检验、梯度下降 | `numerical_gradient`（中心差分）/ `gradient_check` / `gradient_descent` forge 纯实现 |
| **L4–6** 监督学习 | 分类 / 回归 | forge 实现 + `sklearn` 跨库参照 |
| **L7–8** 无监督学习 | 聚类 / 降维 / 密度估计 | forge 实现 + `sklearn` 跨库参照 |
| **L9** 深度学习 | MLP / Attention | forge 反向传播 + 可选 `torch` 跨库参照 |
| **L10** 集成验证 | 不变量 + 跨库闭环编排 | `benchmarks/run.py` |

## 关键设计决策

### 1. 双重实现：库参照 + forge 纯实现

对每个算法，仓库内存在两条互相印证的计算路径：

- **库路径（集大成）**：直接调用世界顶级库（`sklearn` / `scipy` / `numpy` / `torch`），作为**黄金参照**。
- **forge 路径（可验证）**：纯 `numpy` 手写核心算法，暴露可证明的不变量。

`benchmarks` 层把两者放在同一数据上对比，任何一处数值漂移都会立即暴露。

### 2. 不变量驱动的正确性

正确性不靠"看起来对"，而靠**可被独立实现交叉验证的硬不变量**。见 [`MATH.md`](MATH.md)。例如：

- Jacobi 特征分解：`VᵀV = I` **且** `‖A vᵢ − λᵢ vᵢ‖/|λᵢ| ≈ ε` **且** 特征值与 `numpy.linalg.eigh` 逐位一致。
- SVC：KKT 互补松弛全局成立 **且** 与 `sklearn.SVC` 训练集预测一致 ≥ 95%。
- PCA：Eckart-Young 秩-k 重构误差精确等于理论丢弃能量 `(N-1)/(N·d)·Σ_{j>k} λⱼ`。

### 3. 可复现性

所有随机数据由 `_random.py` 统一封装（`np.random.default_rng(seed)`），seed 固定 ⇒ 基准与测试可复现。

### 4. 数值稳定性优先

- `logsumexp` 防溢出；`softmax` 平移不变。
- Cholesky 遇非正定加微岭项容错。
- SVC 对偶 QP 用 `scipy SLSQP` 精确求解；`_smo` 为 forge 原型备选。
- MLP 标准化输入/目标，配合学习率衰减保证损失单调非增。

## 目录映射

```
morningstar_ai/
├── mathx/linalg.py       L1 线性代数
├── mathx/optimize.py     L3 优化
├── supervised/           L4–6（linear / logistic / naive_bayes / knn / svm）
├── unsupervised/         L7–8（kmeans / pca / gmm）
├── deep/                 L9（neural_net / attention）
├── benchmarks/           L10 集成验证
└── _random.py            可复现随机源
```

## 演进路线（规划中）

- **强化学习**：Q-Learning / DQN / PPO，Bellman 残差 → 0 为不变量。
- **概率图模型**：贝叶斯网推断、变分推断。
- **大语言模型 / 多模态**：Transformer、扩散模型，统一以本十层底座为数学基座。
