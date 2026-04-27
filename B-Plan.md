# B\_Plan.md: 电力现货电价预测项目执行计划书

## 1\. 角色定义与项目目标

  - 角色名称：模型工程师 (Member B)
  - 核心任务：负责从 Day 4 至 Day 11 的开发工作，涵盖高阶特征工程挖掘与 ResNet-MLP 深度学习模型构建。
  - 项目目标：基于 7 天历史（672 个时间点）提取多维特征，预测未来 24 小时（96 个时间点）的电价序列，并提供置信度评估，支撑角色 C 的储能套利策略。

-----

## 2\. 每日执行计划 (Day 4 - Day 11)

### 第一阶段：特征工程与数据调优 (Day 4 - Day 7)

目标：在 A 提供的对齐数据基础上，注入物理逻辑与统计规律，并将特征维度控制在 50 维以内。

| 天数 | 任务目标 | 关键步骤 | 交付物 |
|:---|:---|:---|:---|
| Day 4 | 特征筛选与相关性评估 | 1. 针对 `aligned_15min_full.csv` 进行相关性分析 (Pearson/Spearman)<br>2. 识别并剔除与电价弱相关的气象维度<br>3. 确定首批 50 维入模特征清单 | `feature_mask.json` |
| Day 5 | 物理特征与时域编码 | 1. 计算净负荷 (Net Load = Load - Wind - Solar)<br>2. 实现时间周期的 Sin/Cos 编码<br>3. 响应储能约束，计算 2 小时 (8点) 滚动电价统计 | `features.py` (v1) |
| Day 6 | 滞后特征与长程挖掘 | 1. 注入 D-1, D-7 同期滞后特征 (Lag features)<br>2. 计算气象变量的差分特征 (Delta features)<br>3. 完成缺失值与异常值 (Inf/NaN) 的向量化清洗 | `features.py` (v2) |
| Day 7 | 特征管线冻结与联调 | 1. 配合角色 A 完成 `align_pipeline.py` 的集成测试<br>2. 验证特征注入后的 `DataLoader` 吞吐效率<br>3. 持久化最终版的 `scaler.pkl` | `data/scaler.pkl` |

### 第二阶段：模型开发与性能集成 (Day 8 - Day 11)

目标：构建针对长序列输入的 ResNet-MLP 模型，实现高精度的电价预测。

| 天数 | 任务目标 | 关键步骤 | 交付物 |
|:---|:---|:---|:---|
| Day 8 | 模型架构设计与实现 | 1. 构建 1D-ResNet 残差块处理 672 维时序输入<br>2. 设计双头 (Dual-Head) 输出架构：预测值 + 置信度<br>3. 实现特征压缩层降低参数量 | `model.py` |
| Day 9 | 损失函数与训练策略 | 1. 实现 Huber Loss 提升对电价尖峰的鲁棒性<br>2. 配置学习率衰减 (Scheduler) 与早停机制<br>3. 编写训练脚本，记录 Loss 曲线 | `train.py` |
| Day 10 | 滚动验证 (Backtesting) | 1. 执行严格的时间序列 Walk-forward 验证<br>2. 评估 MAE、RMSE 及 2 小时极值捕捉率<br>3. 撰写验证报告，对比 Baseline 模型 | `val_report.md` |
| Day 11 | 推理接口封装与交付 | 1. 实现 `inference.py`，加载权重与 scaler<br>2. 确保输出格式为 `np.float64` 且内存连续<br>3. 与角色 C 进行套利脚本联调 | `inference.py` |

-----

## 3\. 核心模块实现思路

### 3.1 特征工程库 (`features.py`)

遵循 `coding_standards.md` 要求，禁止修改 `Dataset.py` 内部逻辑，采用注入式开发：

```python
def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 物理特征：净负荷是电价波动的核心驱动
    df['net_load_forecast'] = df['load_forecast'] - df['renewable_forecast']
    
    # 2. 储能适配：计算连续2小时(8点)的滚动均值
    df['price_rolling_mean_8'] = df['price'].rolling(window=8).mean()
    
    # 3. 命名规范：{col}_lag_{n}
    df['price_lag_96'] = df['price'].shift(96) # 前一天同期
    
    return df.replace([np.inf, -np.inf], 0).fillna(method='ffill')
```

### 3.2 模型架构 (`model.py`)

采用 1D 卷积与残差全连接网络，兼顾时序局部特征与全局非线性：

  - 输入：`[Batch, 672, N_features]`
  - 特征提取：通过多层 `1D-Conv` 将 672 点压缩至高维语义向量。
  - 预测头：`Price Head` 输出 96 点电价；`Confidence Head` 输出 96 点标准差。

-----

## 4\. 接口对齐节点

### 4.1 与角色 A (数据架构师) 对齐

  - Day 3 (已完成)：确认 `DataLoader` 输出形状为 `(Batch, 672, F)`。
  - Day 7：锁定 `feature_cols` 清单，确保 A 的 `DataModule` 能够正确切片特征矩阵。
  - 文件路径：统一使用 `data/aligned_15min_full.csv` 和 `data/scaler.pkl`。

### 4.2 与角色 C (策略算法师) 对齐

  - Day 11：确定 `predict()` 函数返回值为两个 `numpy` 数组。
  - 数据契约：
      - `prediction`: `np.ndarray` [96], 代表次日 00:00-23:45。
      - `confidence`: `np.ndarray` [96], 用于策略端的风险修正因子。
  - 环境同步：确保 `inference.py` 在角色 C 的 C++ 封装层中可被 Python C-API 正确调用。

-----

## 5\. 风险预警与对策

  - 特征泄露风险：在构建 `price_lag` 等特征时，必须严格执行 `shift` 操作。Day 6 将进行自检。
  - 梯度消失问题：针对 672 点的长输入，若训练不收敛，将在 Day 8 引入 `Batch Normalization` 和 `Residual Connection`。
  - 推理延迟：若 Python 推理过慢影响 C 端的策略搜索，Day 11 将尝试导出为 `ONNX` 格式。