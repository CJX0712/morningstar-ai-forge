# MATH · 数学不变量手册

> **不靠信仰，靠可复现的硬数据。** 每个算法至少有一条可被独立实现交叉验证的不变量。本手册给出每条不变量的数学陈述、验证方式与代码入口。

---

## 1. 线性代数（L1）

### 1.1 Cholesky 分解 `A = L Lᵀ`

- **不变量**：`‖A − L Lᵀ‖_F ≈ 0`（机器精度）。
- **逆矩阵**：`A⁻¹A = I`，其中 `A⁻¹ = (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹`。
- **代码**：`mathx.linalg.cholesky` / `cholesky_inverse`。
- **验证**：`tests/test_mathx.py::test_cholesky_residual`。

### 1.2 Jacobi 循环旋转特征分解

对对称阵 `A`，用正交旋转 `J` 逐对 `(p,q)` 零化离对角元：`A' = Jᵀ A J`。

- **特征值一致**：与 `numpy.linalg.eigh` 逐位一致（`max|λ − λ_ref| < 1e-9`）。
- **正交性**：`‖VᵀV − I‖_F ≈ ε`。
- **残差**：`max_i ‖A vᵢ − λᵢ vᵢ‖₂ / |λᵢ| ≈ ε`。
- **退化处理**：`a_pp = a_qq` 时取旋转角 `φ = π/4`（`t=1`），避免 `θ=0 → t=0` 卡死。
- **代码**：`mathx.linalg.jacobi_eigen`。

### 1.3 logsumexp / softmax

- **logsumexp**：`log Σ exp(xᵢ) = m + log Σ exp(xᵢ − m)`，其中 `m = max xᵢ`，防溢出。
- **softmax**：逐行和为 1（行随机）；对常数平移不变 `softmax(x + c) = softmax(x)`。
- **代码**：`mathx.linalg.logsumexp` / `softmax` / `log_softmax`。

---

## 2. 优化（L3）

### 2.1 梯度检验

- **不变量**：解析梯度 `∇f` 与中心差分 `∂f/∂xᵢ ≈ (f(x+εeᵢ) − f(x−εeᵢ))/(2ε)` 的相对误差 `≈ 1e-7`。
- **判官标准**：对显著分量 `maxRelErr < 1e-5`；近零分量用绝对误差兜底 `max_abs < 1e-7`。
- **代码**：`mathx.optimize.gradient_check`。

### 2.2 梯度下降单调性

- **不变量**：凸目标下 `f_hist` 单调非增（`np.all(diff ≤ tol)`）。
- **收敛**：二次目标收敛到真实极小值 `w* = −½ Q⁻¹ b`。
- **代码**：`mathx.optimize.gradient_descent` / `check_monotone_decrease`。

---

## 3. 监督学习（L4–6）

### 3.1 线性回归

- **正规方程**：`w = (XᵀX)⁻¹ Xᵀ y`，用 Cholesky 求逆。
- **不变量**：残差 ⊥ 设计矩阵 ⇒ `Xᵀ(y − Xŵ) = 0`。
- **代码**：`supervised.linear.LinearRegression`。

### 3.2 逻辑回归

- **损失**：稳定交叉熵（基于 logsumexp）。
- **不变量**：梯度检验 `≈ 1e-7` + 损失单调非增。
- **代码**：`supervised.logistic.LogisticRegression`。

### 3.3 高斯朴素贝叶斯

- **不变量**：先验 × 似然 → 后验，后验概率和 = 1。
- **代码**：`supervised.naive_bayes.GaussianNB`。

### 3.4 k-近邻

- **不变量**：`k = 1` 时训练集误差 = 0（过拟合边界）；`k → ∞` 收敛到贝叶斯误差。
- **代码**：`supervised.knn.KNeighborsClassifier`。

### 3.5 支持向量机（对偶 + KKT）

对偶问题：

```
max_α  Σαᵢ − ½ Σ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
s.t.   0 ≤ αᵢ ≤ C,  Σ αᵢ yᵢ = 0
```

- **KKT 互补松弛**：
  - `αᵢ = 0` ⇒ `yᵢ f(xᵢ) ≥ 1`
  - `0 < αᵢ < C` ⇒ `yᵢ f(xᵢ) = 1`
  - `αᵢ = C` ⇒ `yᵢ f(xᵢ) ≤ 1`
- **偏置**：`b = yᵢ − Σⱼ αⱼyⱼK(xᵢ,xⱼ)`（边界内支持向量均值）。
- **求解器**：默认 `scipy.optimize.minimize(SLSQP)` 精确求解对偶 QP（集大成）；`_smo` 为 forge SMO 原型。
- **边界判定**：容差化（`α ≤ 1e-6` 视为 0，`α ≥ C − 1e-6` 视为 C），兼容优化器返回的 `α ≈ 0/α ≈ C` 近边界值。
- **代码**：`supervised.svm.SVC`。

---

## 4. 无监督学习（L7–8）

### 4.1 K-Means

- **不变量**：SSE（簇内平方和）随迭代**单调非增**。
- **初始化**：k-means++。
- **代码**：`unsupervised.kmeans.KMeans`。

### 4.2 PCA + Eckart-Young

- **协方差**：`C = XcᵀXc/(N−1)`，对称半正定。
- **特征**：Jacobi 特征分解，特征向量正交归一。
- **Eckart-Young 最优性**：秩-k 截断是 Frobenius 意义下的最优低秩近似，重构误差（逐元素均值）= `(N−1)/(N·d)·Σ_{j>k} λⱼ`。
  > 因子说明：`‖Xc − Xc_k‖_F² = Σ_{j>k} σⱼ² = (N−1)·Σ_{j>k} λⱼ`；`reconstruction_error` 用 `np.mean`（对 `N·d` 个元素取均值），故再除以 `N·d`。
- **代码**：`unsupervised.pca.PCA`。

### 4.3 高斯混合（EM）

- **EM 定理**：对数似然**单调非减**。
- **不变量**：责任度每行和 = 1；混合权重和 = 1；协方差对称半正定。
- **代码**：`unsupervised.gmm.GaussianMixture`。

---

## 5. 深度学习（L9）

### 5.1 MLP 反向传播

- **链式法则**（反向传播之母）：`δ^{(out)}` → `δ^{(hidden)}` → 各层梯度。
- **不变量**：梯度检验 `maxRelErr ≈ 1e-7` + 标准化输入/目标后损失单调非增。
- **稳定性**：sigmoid 用分段指数防溢出；tanh 激活。
- **代码**：`deep.neural_net.MLP`。

### 5.2 缩放点积注意力

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
```

- **不变量**：权重矩阵行随机（每行和为 1）；输出与显式 `softmax@V` 自洽；大输入数值稳定（基于 logsumexp）。
- **代码**：`deep.attention.Attention` / `scaled_dot_product_attention`。

---

## 6. 跨库交叉验证总纲

| 自研 forge 模块 | 黄金参照 | 判定 |
|----------------|---------|------|
| LinearRegression | sklearn | 权重/预测一致 |
| LogisticRegression | sklearn | 预测一致 |
| GaussianNB / KNN / SVC | sklearn | 预测一致 |
| KMeans / PCA / GMM | sklearn | 惯性 / 方差比 / 对数似然一致 |
| MLP / Attention | PyTorch（可选） | 梯度一致 `< 1e-6` |
| jacobi_eigen | numpy.linalg.eigh | 特征值一致 `< 1e-9` |

全部判定在 `benchmarks.run_all()` 中编排，输出 `不变量全绿 10/10 | 跨库一致 10/10`。