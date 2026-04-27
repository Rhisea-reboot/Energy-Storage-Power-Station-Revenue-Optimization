# 电力现货电价预测与储能策略优化项目

本项目基于 **PyTorch** 深度学习模型预测蒙西地区电力现货电价，并结合 **C++/Python** 策略搜索算法制定储能系统充放电计划，实现收益最大化。

> 赛题说明见 `Plan.md`，分阶段执行计划见 `Achieve.md`（成员A）、`B-Plan.md`（成员B）、`C_Plan.md`（成员C）。

---

## 一、目录结构（数据/模型/输出分离）

```
.
├── Plan.md / Achieve.md / B-Plan.md / C_Plan.md   # 计划文档
├── coding_standards.md                              # 编码规范
├── README.md                                        # 本文档
│
├── run_pipeline.py                                  # 【端到端流水线入口】
│
# ---------- 阶段一：数据预处理（成员A） ----------
├── Test_Weather_processor.py        # Day 1: NC气象降维
├── Dataaligning.py                  # Day 2: 多源15min对齐
├── Dataset.py                       # Day 3: PyTorch Dataset封装
│
# ---------- 阶段二：特征工程 + 模型（成员B） ----------
├── features.py                      # Day 4-7: 特征工程
├── model.py                         # Day 8: ResNet-MLP架构
├── train.py                         # Day 9-10: 训练脚本
├── inference.py                     # Day 11: 推理接口
│
# ---------- 阶段三：策略算法（成员C） ----------
└── power_price/
    ├── data/                        # 【数据目录】原始数据 + 中间产物
    │   ├── weather_raw/             # 原始NC气象文件
    │   ├── mengxi_node_price_selected.csv      # 原始电价
    │   ├── mengxi_boundary_anon_filtered.csv   # 原始边界条件
    │   ├── weather_features.csv                # Day 1 输出
    │   ├── aligned_15min_full.csv              # Day 2 输出
    │   ├── aligned_15min_processed.csv         # 特征工程调试输出
    │   ├── feature_mask.json                   # Day 4-7 输出
    │   ├── scaler.pkl                          # Day 3/9 输出
    │   ├── pred.npy                            # Day 11 推理输出
    │   └── conf.npy                            # Day 11 推理输出
    │
    ├── models/                      # 【模型目录】
    │   └── best_model.pt            # Day 9-10 训练输出
    │
    ├── output/                      # 【策略输出目录】
    │   └── output.csv               # Day 15 最终提交格式
    │
    ├── test_data/                   # 【测试数据目录】
    │   └── test_in_feature_ori.csv  # 原始test特征（8列中文，无电价）
    │
    ├── prepare_test_features.py     # Day 15: 原始test特征→完整50维特征
    ├── strategy_core.cpp            # Day 12: C++核心搜索
    ├── strategy_core.so             # 编译后的动态库
    ├── strategy_wrapper.py          # Day 12: Python封装
    ├── strategy_features.py         # Day 5: 物理特征辅助
    ├── risk_mgmt.py                 # Day 13: 风险管理
    ├── risk_config.yaml             # Day 13: 风险参数配置
    ├── monte_carlo_validator.py     # Day 13: Monte-Carlo验证
    ├── ensemble_strategy.py         # Day 14: 多模型集成
    └── main.py                      # Day 15: 策略生成主入口
```

---

## 二、各文件详细分工（含输入/输出路径）

### 阶段一：数据预处理（成员 A）

#### 1. `Test_Weather_processor.py` — 气象数据降维（Day 1）

| 项目 | 路径 |
|:---|:---|
| **输入** | `power_price/data/weather_raw/*.nc` |
| **输出** | `power_price/data/weather_features.csv` |

**使用方法**：
```bash
python Test_Weather_processor.py
```
- 提取 NC 文件中 6 类气象变量的空间统计（mean/max/min/std）
- 构建风速合成、太阳辐射潜力等派生特征
- 输出为 1 小时分辨率 CSV（Day 2 会插值为 15 分钟）

---

#### 2. `Dataaligning.py` — 多源数据对齐（Day 2）

| 项目 | 路径 |
|:---|:---|
| **输入1 电价** | `power_price/data/mengxi_node_price_selected.csv`（目标列 `A`） |
| **输入2 边界条件** | `power_price/data/mengxi_boundary_anon_filtered.csv`（中文列名自动映射） |
| **输入3 气象** | `power_price/data/weather_features.csv`（Day 1 输出） |
| **输出** | `power_price/data/aligned_15min_full.csv` |

**使用方法**：
```bash
python Dataaligning.py
```
- 统一时区为 `Asia/Shanghai`，重采样到 15 分钟分辨率
- 计算竞价空间、净负荷、新能源渗透率、预测误差等电力系统衍生特征
- 添加正弦-余弦时间编码（`time_sin/cos`, `dow_sin/cos`, `month_sin/cos`）
- 输出宽表列数约 60~76 维（供后续特征筛选）

---

#### 3. `Dataset.py` — PyTorch 数据封装（Day 3）

| 项目 | 路径 |
|:---|:---|
| **输入** | `power_price/data/aligned_15min_full.csv` |
| **输出** | `power_price/data/scaler.pkl`（直接运行 `Dataset.py` 时生成） |

**使用方法**：
```bash
python Dataset.py
```
- 按时间先后分割训练/验证集（严禁随机打乱）
- 训练集 `fit_transform`，验证集 `transform`（防止数据泄露）
- 样本维度：`x: (batch, 672, F)` → `y: (batch, 96)`
- **注意**：实际训练时 `train.py` 会重新生成与 `feature_mask.json` 维度一致的 `scaler.pkl`

---

### 阶段二：特征工程与模型（成员 B）

#### 4. `features.py` — 特征工程库（Day 4-7）

| 项目 | 路径 |
|:---|:---|
| **输入** | `power_price/data/aligned_15min_full.csv` |
| **输出** | `power_price/data/feature_mask.json` |
| **调试输出** | `power_price/data/aligned_15min_processed.csv` |

**使用方法**：
```bash
python -c "
import pandas as pd
from features import build_feature_pipeline
df = pd.read_csv('power_price/data/aligned_15min_full.csv', parse_dates=['times'], index_col='times')
df_out, cols = build_feature_pipeline(df, target_col='A')
print('Selected features:', len(cols))
"
```
- 注入时间周期编码、策略滚动统计、滞后特征（`price_lag_96` 等）
- 通过 Pearson 相关性筛选 Top-50 特征，生成 `feature_mask.json`
- 该掩码供 `train.py`、`inference.py`、`Dataset.py` 共享

---

#### 5. `model.py` — ResNet-MLP 模型（Day 8）

| 项目 | 路径 |
|:---|:---|
| **输入** | 动态特征维度（由 `power_price/data/feature_mask.json` 决定） |
| **输出** | 无文件输出，仅验证维度 |

**使用方法**：
```bash
python model.py
```
- 输入：`[Batch, 672, feature_dim]`，默认 `feature_dim = 50`
- 输出：`price [batch, 96]` + `confidence [batch, 96]`
- 双头输出：电价预测头 + 置信度估计头（softplus 激活保证正值）

---

#### 6. `train.py` — 训练脚本（Day 9-10）

| 项目 | 路径 |
|:---|:---|
| **输入1 数据** | `power_price/data/aligned_15min_full.csv` |
| **输入2 特征掩码** | `power_price/data/feature_mask.json` |
| **输出1 模型权重** | `power_price/models/best_model.pt` |
| **输出2 Scaler** | `power_price/data/scaler.pkl`（与 feature_mask 同维度） |

**使用方法**：
```bash
python train.py
```
- 加载 `feature_mask.json` 中的 50 维特征进行训练
- Loss = Huber Loss + 0.2 × 置信度拟合 MSE
- 验证集 MSE 最低时保存模型
- **训练结束后自动保存 `scaler.pkl`**，供推理复用

---

#### 7. `inference.py` — 推理接口（Day 11）

| 项目 | 路径 |
|:---|:---|
| **输入1 数据** | `power_price/data/aligned_15min_full.csv` 或构造好的完整特征 DataFrame |
| **输入2 模型** | `power_price/models/best_model.pt` |
| **输入3 Scaler** | `power_price/data/scaler.pkl` |
| **输入4 特征掩码** | `power_price/data/feature_mask.json` |
| **输出1 预测** | `power_price/data/pred.npy`（运行 `inference.py` 时自动保存） |
| **输出2 置信度** | `power_price/data/conf.npy`（运行 `inference.py` 时自动保存） |

**使用方法**：
```bash
python inference.py
```
- 自动读取宽表最后 672 行作为历史窗口
- 输出 `prediction [96]` + `confidence [96]`
- **运行后自动保存** `pred.npy` 和 `conf.npy` 到 `power_price/data/`
- B → C 数据契约：`np.ndarray [96]`, dtype float64

**编程式调用**：
```python
from inference import PricePredictor
import pandas as pd

predictor = PricePredictor(
    model_path="power_price/models/best_model.pt",
    scaler_path="power_price/data/scaler.pkl",
    mask_path="power_price/data/feature_mask.json"
)
df = pd.read_csv("power_price/data/aligned_15min_full.csv", parse_dates=['times'], index_col='times')
pred, conf = predictor.predict(df)
```

**批量/滚动预测**（`predict_batch`）：
```python
preds, confs, tss = predictor.predict_batch(
    df_input=df_features,
    target_dates=pd.date_range('2026-01-01', periods=59, freq='D'),
    target_col="A"  # 启用滚动预测：每天预测后把电价写回 df_input
)
```

---

### 阶段三：策略算法（成员 C）

> 以下模块位于 `power_price/` 目录，建议在该目录下运行策略相关命令。

#### 8. `strategy_core.cpp` + `strategy_core.so` — C++ 核心搜索（Day 12）

| 项目 | 路径 |
|:---|:---|
| **输入** | 内存中的 `double[96]` 电价数组 |
| **输出** | 最优 `(tc, td)` 索引与预期收益 |

**编译方法**（如 `.so` 缺失或需更新）：
```bash
cd power_price
g++ -O2 -shared -fPIC -o strategy_core.so strategy_core.cpp
```

---

#### 9. `strategy_wrapper.py` — Python 封装层（Day 12）

| 项目 | 路径 |
|:---|:---|
| **输入** | `np.ndarray [96]` 电价 + 可选置信度 |
| **输出** | `dict` 包含 `charge_start`, `discharge_start`, `power_profile`, `expected_profit` |

**使用方法**：
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
- 优先调用 `strategy_core.so`（C++ 高性能）
- 若 C++ 库不可用，自动降级为纯 Python 实现

---

#### 10. `risk_mgmt.py` — 风险修正（Day 13）

| 项目 | 路径 |
|:---|:---|
| **输入** | `prediction [96]`, `confidence [96]`, `config dict` |
| **输出** | `dict` 包含保守估计、收益阈值、极端价格掩码等 |

**使用方法**：
```python
from power_price.risk_mgmt import apply_risk_management

result = apply_risk_management(prediction, confidence, config={
    "risk_aversion": 0.1,
    "z_score": 1.28,
    "fixed_profit_threshold": 5000,
})
adjusted_prediction = result["adjusted_prediction"]
```
- 支持置信度折扣、分位数保守估计、收益阈值、极端价格过滤
- 参数可通过 `power_price/risk_config.yaml` 持久化配置

---

#### 11. `main.py` — 端到端策略生成主入口（Day 15）

| 项目 | 路径 |
|:---|:---|
| **输入1 预测** | `data/pred.npy`（相对 `power_price/` 目录） |
| **输入2 置信度** | `data/conf.npy`（可选，默认使用 `pred.std() * 0.1`） |
| **输入3 原始特征** | `test_data/test_in_feature_ori.csv`（通过 `--input-csv`） |
| **输入4 配置** | `risk_config.yaml`（同目录，可选） |
| **输出** | `output/output.csv` |

**使用方法**：
```bash
cd power_price

# 方式A：从 .npy 文件（单日/批量）
python main.py --prediction data/pred.npy --confidence data/conf.npy --output output/output.csv

# 方式B：批量模式（多天的 .npy 文件放在同一目录）
python main.py --batch --prediction-dir data/preds/ --output output/output.csv

# 方式C：从原始 test 特征 CSV 端到端（训练完成后使用，推荐）
python main.py --input-csv test_data/test_in_feature_ori.csv --output output/output.csv
```
- 方式C 内部自动完成：特征工程 → 模型预测 → 策略搜索
- 输出格式：`times, 实时价格(预测), power`
- `power` 列：充电时段 = `-1000`，放电时段 = `+1000`，其余 = `0`

#### 12. `prepare_test_features.py` — Test 特征构造（Day 15）

| 项目 | 路径 |
|:---|:---|
| **输入1 test特征** | `test_data/test_in_feature_ori.csv`（8列中文原始预测特征） |
| **输入2 历史数据** | `data/aligned_15min_full.csv` |
| **输入3 气象数据** | `data/weather_features.csv` |
| **输出** | 完整特征 DataFrame（含历史上下文 + test 期间 50 维特征） |

**核心逻辑**：
- 中文列名映射为英文（`系统负荷预测值` → `load_forecast` 等）
- **测试期实际值未知**：用 `forecast` 填充 `actual`（最佳估计）
- **测试期真实电价缺失**：`A` 列在 test 期间保持 `NaN`，`curtailment_flag` 被正确置 0
- 从历史 aligned 数据拼接尾部上下文（≥7天），确保滚动窗口特征计算正确
- 重新计算全部派生特征（竞价空间、净负荷、渗透率等）与时间编码

**编程式调用**：
```python
from power_price.prepare_test_features import prepare_test_features
df_features = prepare_test_features(test_csv="power_price/test_data/test_in_feature_ori.csv")
```

---

## 三、端到端流水线（推荐）

`run_pipeline.py` 串联了以上所有阶段，按顺序调用各模块。

### 完整运行

```bash
# 全流程：数据检查 → 特征工程 → 模型训练 → 策略生成
python run_pipeline.py --all --epochs 10

# 从原始 test 特征直接生成提交结果（训练完成后使用）
python run_pipeline.py --input-csv power_price/test_data/test_in_feature_ori.csv --output power_price/output/output.csv
```

### 分阶段运行

```bash
# 仅数据预处理
python run_pipeline.py --stage1

# 仅特征工程（生成 power_price/data/feature_mask.json）
python run_pipeline.py --stage2

# 仅模型训练（生成 power_price/models/best_model.pt + power_price/data/scaler.pkl）
python run_pipeline.py --stage3 --epochs 20 --batch-size 64

# 策略生成演示（若模型已存在则真实推理，否则用模拟数据演示）
python run_pipeline.py --stage4-demo
```

### 手动串联（灵活调试）

```bash
# 1. 数据预处理（若 aligned_15min_full.csv 已存在可跳过）
python Test_Weather_processor.py
python Dataaligning.py

# 2. 特征工程（生成 power_price/data/feature_mask.json）
python -c "
import pandas as pd
from features import build_feature_pipeline
df = pd.read_csv('power_price/data/aligned_15min_full.csv', parse_dates=['times'], index_col='times')
build_feature_pipeline(df, target_col='A')
"

# 3. 训练模型（生成 power_price/models/best_model.pt + power_price/data/scaler.pkl）
python train.py

# 4. 推理（生成 power_price/data/pred.npy + conf.npy）
python inference.py

# 5. 生成策略（生成 power_price/output/output.csv）
cd power_price
python main.py --prediction data/pred.npy --confidence data/conf.npy --output output/output.csv
```

---

## 四、关键接口契约

| 阶段 | 输出文件 | 接收方 | 数据格式 |
|:---|:---|:---|:---|
| Day 1 → Day 2 | `power_price/data/weather_features.csv` | `Dataaligning.py` | 小时级气象特征表 |
| Day 2 → Day 3 | `power_price/data/aligned_15min_full.csv` | `Dataset.py` / `features.py` | 15min 统一宽表 |
| Day 3/7 → Day 8 | `power_price/data/feature_mask.json` | `train.py` / `inference.py` | ≤50 维特征名列表 |
| Day 9 → Day 11 | `power_price/models/best_model.pt` | `inference.py` | PyTorch state_dict |
| Day 9 → Day 11 | `power_price/data/scaler.pkl` | `inference.py` | sklearn StandardScaler |
| Day 11 → Day 15 | `power_price/data/pred.npy` | `main.py` | `np.ndarray [96]` float64 |
| Day 11 → Day 15 | `power_price/data/conf.npy` | `main.py` | `np.ndarray [96]` float64 |
| Day 15 输入 | `power_price/test_data/test_in_feature_ori.csv` | `prepare_test_features.py` | 原始预测特征（8列中文） |
| Day 15 → 提交 | `power_price/output/output.csv` | 赛题评测 | `times, 实时价格(预测), power` |

---

## 五、环境依赖

```bash
# 核心依赖
pip install torch pandas numpy scikit-learn xarray joblib

# 策略模块可选依赖
pip install pyyaml  # 用于读取 power_price/risk_config.yaml
```

**C++ 编译环境**：若需重新编译 `power_price/strategy_core.so`，要求 GCC 9+ 或兼容的 C++ 编译器。

---

## 六、注意事项

1. **目标列名**：本项目电价目标列使用 **`A`**（蒙西节点），与 `mengxi_node_price_selected.csv` 原始列名一致。若你的数据使用 `price` 作为列名，请在调用 `PowerDataAligner`、`PowerPriceDataModule`、`build_feature_pipeline` 时同步修改 `target_col` / `price_col` 参数。

2. **时区处理**：NC 气象数据无 TZ 信息，默认视为 UTC 后转 `Asia/Shanghai`；Dataset 加载时会去除时区以便日期比较。

3. **防止数据泄露**：
   - 训练/验证必须**按时间先后分割**，严禁随机打乱。
   - `StandardScaler` 只能在训练集 `fit`，验证集只能 `transform`。

4. **路径一致性**：所有中间产物已统一到 `power_price/data/`、`power_price/models/`、`power_price/output/` 三个目录下，**请勿在根目录下散落 `.npy` / `.json` 文件**。

5. **测试期无真实电价约束**：
   - 比赛规则规定测试集不提供历史真实电价，代码已严格遵循：
   - `prepare_test_features.py` 中 test 期间 `A` 列保持 `NaN`，不向前填充历史值
   - `inference.py` 的 `predict_batch` 支持 `target_col='A'` 滚动预测，每预测一天后将预测电价写回输入，供下一天使用
   - 当前 `feature_mask.json` 筛选出的 50 维特征不包含历史电价滞后项，因此模型本身不依赖前一日真实电价
