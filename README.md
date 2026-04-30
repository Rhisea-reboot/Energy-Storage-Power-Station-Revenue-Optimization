# 电力现货电价预测与储能策略优化

基于 PyTorch 深度学习模型预测蒙西地区电力现货电价，结合 C++/Python 策略搜索算法制定储能充放电计划，实现收益最大化。

> 赛题说明见 `Plan.md`，分阶段执行计划见 `Achieve.md`（数据预处理）、`B-Plan.md`（模型训练）、`C_Plan.md`（策略算法）。AI Agent 协作规范见 `AGENTS.md`。

---

## 目录结构

```
.
├── Plan.md / Achieve.md / B-Plan.md / C_Plan.md   # 计划文档
├── AGENTS.md                                      # AI Agent 协作指南
├── coding_standards.md                            # 编码规范
├── README.md                                      # 本文档
│
├── run_pipeline.py                                # 端到端流水线入口
│
# ---------- 数据预处理 ----------
├── Test_Weather_processor.py        # NC 气象降维
├── Dataaligning.py                  # 多源 15min 对齐
├── Dataset.py                       # PyTorch Dataset 封装
│
# ---------- 特征工程 + 模型 ----------
├── features.py                      # 特征工程库
├── model.py                         # ResNet-MLP + Time Embedding 架构
├── train.py                         # 训练脚本（50 epoch，动态 confidence 权重）
├── inference.py                     # 推理接口（单日/批量/滚动预测）
├── lgb_baseline.py                  # LightGBM 基线（可选）
│
# ---------- 策略算法 ----------
└── power_price/
    ├── data/                        # 数据与中间产物
    │   ├── weather_raw/             # 原始 NC 气象文件
    │   ├── weather_features.csv     # 气象特征
    │   ├── aligned_15min_full.csv   # 对齐主表（Dataaligning.py 输出）
    │   ├── aligned_15min_processed.csv  # 特征工程后宽表（features.py 输出）
    │   ├── feature_mask.json        # 筛选后的特征列表（≤50 维，test 兼容）
    │   ├── scaler.pkl               # StandardScaler
    │   ├── pred.npy / conf.npy      # 单日推理输出（调试用）
    │   ├── preds/                   # 批量预测 .npy 存放目录
    │   ├── mengxi_node_price_selected.csv
    │   └── mengxi_boundary_anon_filtered.csv
    │
    ├── models/
    │   └── best_model.pt            # 模型权重
    │
    ├── output/                      # 策略输出
    │   └── output.csv               # 最终提交格式（5664 行）
    │
    ├── test_data/
    │   └── test_in_feature_ori.csv  # 原始 test 特征（8 列中文，59 天）
    │
    ├── strategy_core.cpp            # C++ 核心搜索
    ├── strategy_core.so             # 编译后的动态库
    ├── strategy_wrapper.py          # Python 封装（ctypes + 纯 Python 降级）
    ├── strategy_features.py         # 策略辅助特征
    ├── risk_mgmt.py                 # 风险管理
    ├── risk_config.yaml             # 风险参数配置
    ├── monte_carlo_validator.py     # Monte-Carlo 验证
    ├── ensemble_strategy.py         # 多模型集成
    ├── prepare_test_features.py     # Test 特征构造（原始 8 列 → 完整 50 维）
    └── main.py                      # 策略生成主入口
```

---

## 快速开始

### 环境安装

```bash
pip install torch pandas numpy scikit-learn xarray joblib pyyaml lightgbm
```

> **GPU 支持**：若 PyTorch 无法检测到 CUDA，请检查 `LD_LIBRARY_PATH` 是否包含 conda 的 libstdc++。已配置激活脚本：`/home/rhisea/miniconda3/etc/conda/activate.d/libstdcxx.sh`

C++ 策略库已预编译为 `power_price/strategy_core.so`。若需重新编译：

```bash
cd power_price
g++ -O2 -shared -fPIC -o strategy_core.so strategy_core.cpp
```

### 全流程运行（推荐）

```bash
# 1. 训练模型（50 epoch，GPU 加速约 10~20 分钟）
python run_pipeline.py --stage3 --epochs 50

# 2. 从原始 test 特征生成提交结果（59 天，5664 行）
python run_pipeline.py \
  --input-csv power_price/test_data/test_in_feature_ori.csv \
  --output power_price/output/output.csv
```

**输出验证**：`output.csv` 应为 `(5664, 3)`，对应 59 天 × 96 点/天。若只有 `(96, 3)`，说明未走 `--input-csv` 流程。

### 分阶段运行

```bash
# 仅数据预处理 + 特征工程
python run_pipeline.py --stage1
python run_pipeline.py --stage2

# 仅模型训练（可调 epochs、batch size、patience）
python run_pipeline.py --stage3 --epochs 100 --batch-size 128

# 从已有模型直接生成提交结果（跳过训练）
python run_pipeline.py \
  --input-csv power_price/test_data/test_in_feature_ori.csv \
  --output power_price/output/output.csv
```

---

## 核心模块说明

### 数据预处理

#### `Test_Weather_processor.py` — 气象数据降维

```bash
python Test_Weather_processor.py
```

- 输入：`power_price/data/weather_raw/*.nc`
- 输出：`power_price/data/weather_features.csv`
- 提取 6 类气象变量的空间统计（mean/max/min/std），构建风速合成、太阳辐射潜力等派生特征

#### `Dataaligning.py` — 多源数据对齐

```bash
python Dataaligning.py
```

- 输入：
  - 电价 `power_price/data/mengxi_node_price_selected.csv`（目标列 `A`）
  - 边界条件 `power_price/data/mengxi_boundary_anon_filtered.csv`（中文列名自动映射为英文）
  - 气象 `power_price/data/weather_features.csv`
- 输出：`power_price/data/aligned_15min_full.csv`
- 统一时区 `Asia/Shanghai`，重采样到 15 分钟，计算竞价空间、净负荷、新能源渗透率等衍生特征，添加 sin/cos 时间编码

#### `Dataset.py` — PyTorch 数据封装

```bash
python Dataset.py
```

- 输入：`power_price/data/aligned_15min_processed.csv`（特征工程后的宽表）
- 滑动窗口：输入 672 点（7 天）→ 输出 96 点（1 天）
- 按时间先后分割训练/验证（前 10 个月训练，后 2 个月验证），`shuffle=False`
- 训练集 `fit_transform`，验证集 `transform`（防数据泄露）

---

### 特征工程与模型

#### `features.py` — 特征工程

```bash
python -c "
import pandas as pd
from features import build_feature_pipeline
df = pd.read_csv('power_price/data/aligned_15min_full.csv', parse_dates=['times'], index_col='times')
df_out, cols = build_feature_pipeline(df, target_col='A')
print('Selected features:', len(cols))
"
```

- 注入时间周期编码、多尺度滚动统计（8/24/96/192/672）、差分特征、日内统计、峰谷比
- **自动排除 `A_*` 衍生特征**（`A_lag_1`、`A_rolling_mean_8` 等）：测试期不提供历史真实电价，这些特征在 test 期间无法正确计算
- 通过 Pearson 相关性筛选 Top-50 test-兼容特征，生成 `power_price/data/feature_mask.json`
- 输出 `power_price/data/aligned_15min_processed.csv`

#### `model.py` — ResNet-MLP + Time Embedding

```bash
python model.py
```

架构：
```
Input [B, 672, F]
  → Linear(F → hidden)
  → Conv1d(hidden → hidden, kernel=3, padding=1)
  → ResidualBlock(Conv1d) × layers
  → Conv1d(hidden → hidden, kernel=7, stride=7)   # 672→96
  → + TimeEmbedding(96, hidden)                    # 打破输出同质化
  → Linear(hidden → 1) per time-step              # [B, 96]
```

默认参数：`hidden=128`, `layers=4`, `feature_dim` 由 `feature_mask.json` 动态决定。

> **为什么加 Time Embedding**：早期版本去掉全局池化后改用时间压缩卷积，但 Conv1d 输出头导致 96 个时间步的预测高度同质化（std 仅 0.05）。Time Embedding 为每个输出时间步提供独立偏置，强制产生日内差异。

#### `train.py` — 训练脚本

```bash
python train.py
```

- 加载 `feature_mask.json` 中的特征进行训练
- Loss = Huber Loss + α × 置信度 MSE
  - α 动态：前 3 epoch 为 `0.2`（预热），之后为 `0.6`
- Optimizer: AdamW (lr=1e-3) + **CosineAnnealingLR**
- **Early Stopping**: 验证集 MSE 连续 `patience` 轮（默认 10）不下降则停止
- 验证集输出 **Conf-Error Correlation**（评估置信度质量）
- 保存模型基于 **ValMSE 最低**
- **注意**：默认 50 epoch。若 ValMSE 仍高，可增至 100 epoch 或调大 `patience`

#### `inference.py` — 推理接口

```bash
python inference.py
```

- 自动读取宽表最后 672 行作为历史窗口
- 输出 `pred.npy` + `conf.npy` 到 `power_price/data/`（**单日 96 点，仅用于调试**）

**编程式调用**：

```python
from inference import PricePredictor
import pandas as pd

predictor = PricePredictor(
    model_path="power_price/models/best_model.pt",
    scaler_path="power_price/data/scaler.pkl",
    mask_path="power_price/data/feature_mask.json"
)

# 单日预测（历史窗口最后 672 行）
df = pd.read_csv("power_price/data/aligned_15min_processed.csv", parse_dates=['times'], index_col='times')
pred, conf = predictor.predict(df)  # pred.shape == (96,)

# 批量/滚动预测（测试期必需：将预测值逐日写回 A 列）
preds, confs, tss = predictor.predict_batch(
    df_input=df_features,
    target_dates=pd.date_range('2026-01-01', periods=59, freq='D'),
    skip_insufficient=True,
    target_col="A"
)
```

---

### 策略算法（在 `power_price/` 目录下运行）

#### `main.py` — 策略生成主入口

**方式一：端到端（推荐，完整 59 天输出）**

```bash
cd power_price
python main.py --input-csv test_data/test_in_feature_ori.csv --output output/output.csv
```

- 自动完成：特征工程 → 模型预测 → 风险管理 → 策略搜索
- 输出：`times, 实时价格(预测), power`
- `power` 列：充电时段 = `-1000`，放电时段 = `+1000`，其余 = `0`
- **输出形状应为 `(5664, 3)`**

**方式二：从单日 .npy 调试（仅 96 行，开发调试用）**

```bash
cd power_price
python main.py --prediction data/pred.npy --confidence data/conf.npy --output output/output.csv
```

**方式三：批量 .npy 目录（需自行准备多天预测）**

```bash
cd power_price
python main.py --batch --prediction-dir data/preds/ --output output/output.csv
```

#### `prepare_test_features.py` — Test 特征构造

```python
from power_price.prepare_test_features import prepare_test_features

df_features = prepare_test_features(
    test_csv="power_price/test_data/test_in_feature_ori.csv"
)
```

- 中文列名映射为英文（如 `系统负荷预测值` → `load_forecast`）
- 测试期实际值未知：用 `forecast` 填充 `actual`（**注意：这导致 `*_forecast_error` 特征在 test 期间恒为 0**）
- 测试期真实电价缺失：`A` 列保持 `NaN`，`curtailment_flag` 等依赖 `A` 的特征置 0
- 从历史 aligned 数据拼接尾部上下文（≥7 天），确保滚动窗口特征计算正确
- **调用 `features.py`** 生成与训练一致的策略特征、滞后特征，避免列缺失

#### `strategy_wrapper.py` — 充放电策略搜索

```bash
cd power_price
python -c "
import numpy as np
from strategy_wrapper import optimize_charge_discharge
prices = np.random.randn(96) * 100 + 300
result = optimize_charge_discharge(prices)
print(result)
"
```

- 优先调用 C++ `strategy_core.so`，若不可用自动降级为纯 Python O(N²) 实现
- 搜索目标：最大化 `放电时段电价之和 - 充电时段电价之和`
- 支持置信度惩罚与收益阈值过滤

#### `risk_mgmt.py` — 风险修正

```python
from power_price.risk_mgmt import apply_risk_management

result = apply_risk_management(prediction, confidence, config={
    "risk_aversion": 0.1,
    "z_score": 1.28,
    "fixed_profit_threshold": None,
})
adjusted_prediction = result["adjusted_prediction"]
```

- 支持置信度折扣、分位数保守估计、自适应收益阈值、极端价格过滤
- 参数可通过 `risk_config.yaml` 持久化配置

---

## 手动串联（灵活调试）

```bash
# 1. 数据预处理
python Test_Weather_processor.py
python Dataaligning.py

# 2. 特征工程
python -c "
import pandas as pd
from features import build_feature_pipeline
df = pd.read_csv('power_price/data/aligned_15min_full.csv', parse_dates=['times'], index_col='times')
build_feature_pipeline(df, target_col='A')
"

# 3. 训练模型
python train.py

# 4. 生成完整提交结果（端到端）
cd power_price
python main.py --input-csv test_data/test_in_feature_ori.csv --output output/output.csv
```

---

## 关键接口契约

| 阶段 | 输出文件 | 接收方 | 数据格式 |
|:---|:---|:---|:---|
| Day 1 → Day 2 | `power_price/data/weather_features.csv` | `Dataaligning.py` | 小时级气象特征 |
| Day 2 → Day 3 | `power_price/data/aligned_15min_full.csv` | `features.py` | 15min 宽表，索引 `times`，目标列 `A` |
| Day 2 → Day 3 | `power_price/data/aligned_15min_processed.csv` | `Dataset.py` / `train.py` / `inference.py` | 特征工程后宽表，≤50 维特征 |
| Day 3/7 → Day 8 | `power_price/data/feature_mask.json` | `train.py` / `inference.py` | `list[str]`，≤50 维，无 `A_*` 特征 |
| Day 9 → Day 11 | `power_price/models/best_model.pt` | `inference.py` | PyTorch `state_dict` |
| Day 9 → Day 11 | `power_price/data/scaler.pkl` | `inference.py` | `sklearn.StandardScaler` |
| Day 11 → Day 15 | `power_price/data/pred.npy` | `main.py`（调试） | `np.ndarray [96]`, `float64` |
| Day 11 → Day 15 | `power_price/data/conf.npy` | `main.py`（调试） | `np.ndarray [96]`, `float64` |
| 提交输入 | `power_price/test_data/test_in_feature_ori.csv` | `prepare_test_features.py` | 8 列中文原始特征，59 天 |
| 提交输出 | `power_price/output/output.csv` | 赛题评测 | `times, 实时价格(预测), power`，5664 行 |

---

## 注意事项

1. **目标列名**：电价目标列统一使用 **`A`**（蒙西节点）。若数据使用 `price`，请同步修改各模块的 `target_col` / `price_col` 参数。

2. **时区处理**：NC 气象数据无 TZ 信息，视为 UTC 后转 `Asia/Shanghai`；Dataset 加载时去除时区以便日期比较。

3. **防止数据泄露**：
   - 训练/验证必须按时间先后分割，严禁随机打乱
   - `StandardScaler` 只能在训练集 `fit`，验证集只能 `transform`
   - DataLoader 中 `shuffle=False`

4. **路径一致性**：所有中间产物统一到 `power_price/data/`、`power_price/models/`、`power_price/output/` 三个目录，请勿在根目录散落文件。

5. **测试期无真实电价**：
   - `prepare_test_features.py` 中 test 期间 `A` 列保持 `NaN`
   - `inference.py` 的 `predict_batch` 支持 `target_col='A'` 滚动预测，每天预测后将预测值写回输入
   - `feature_mask.json` 已排除 `A_*` 衍生特征，确保 test 期间模型输入不依赖历史真实电价

6. **训练 epoch 数**：`train.py` 默认 50 epoch。若 ValMSE 仍高（如 >1.0），建议增至 100 epoch 或调大 `patience`。

7. **输出形状**：使用 `--input-csv` 时输出应为 5664 行（59 天）；使用 `--prediction` 单文件时输出为 96 行（1 天）。

8. **GPU 环境**：若导入 torch/numpy 时报 `GLIBCXX_3.4.29` 错误，请确保 conda 环境已激活（`LD_LIBRARY_PATH` 已由 `libstdcxx.sh` 自动配置）。
