# AGENTS.md — AI Agent 协作指南

> 本文件面向 AI Agent，提供项目背景、接口契约与协作规范。人类用户请优先阅读 `README.md`。

---

## 1. 项目概述

### 1.1 赛题目标

构建**电力现货电价预测模型**与**储能充放电策略优化算法**，在蒙西地区电力现货市场实现收益最大化。

- **预测任务**：预测未来 24 小时（96 个 15 分钟分辨率点）的实时电价
- **策略任务**：在储能约束条件下制定充放电计划，最大化收益

**评测指标**：测试集所有天数收益的平均值

```
Profit = Σ(t=0 to 95) P_t × E_t
```

- `P_t`：真实实时电价
- `E_t`：充放电功率（+1000 放电，-1000 充电，0 不操作）

### 1.2 储能约束（硬性约束）

| 参数 | 值 |
|:---|:---|
| 储能容量 | 8000 MWh |
| 充放电功率 | ±1000 MW |
| 单次持续时间 | 连续 8 个时间点（2 小时） |
| 初始 SOC | 0 |
| 每日操作限制 | 最多 1 次完整充放电（充 + 放） |
| 充电开始时间 | 0 ≤ t_c ≤ 80 |
| 放电开始时间 | t_c + 8 ≤ t_d ≤ 88 |

---

## 2. 模块职责与数据流

```
weather_raw/*.nc
      ↓ Test_Weather_processor.py
weather_features.csv
      ↓ Dataaligning.py ( + 电价 + 边界条件 )
aligned_15min_full.csv
      ↓ features.py
aligned_15min_processed.csv + feature_mask.json (≤50维, 无A_*特征)
      ↓ Dataset.py + train.py
best_model.pt + scaler.pkl
      ↓ inference.py
pred.npy + conf.npy（调试用，单日96点）
      ↓ main.py / strategy_wrapper.py
output.csv（提交格式，5664行 = 59天×96点）
```

### 各模块职责

| 模块 | 职责 | 关键产出 |
|:---|:---|:---|
| `Test_Weather_processor.py` | NC 气象降维：提取 6 类变量的空间统计（mean/max/min/std）及派生特征 | `weather_features.csv` |
| `Dataaligning.py` | 多源 15min 对齐：统一时区、重采样、电力系统衍生特征（竞价空间、净负荷、渗透率等） | `aligned_15min_full.csv` |
| `Dataset.py` | PyTorch Dataset 封装：滑动窗口、StandardScaler、时序分割 | `scaler.pkl`（训练时由 `train.py` 最终保存） |
| `features.py` | 特征工程：时间编码（sin/cos）、多尺度滚动统计、差分特征、滞后特征、Top-K 筛选 | `feature_mask.json` |
| `model.py` | ResNet-MLP + Time Embedding 架构：1D 卷积残差块 + 时间压缩（672→96）+ 时间嵌入 + 双头输出 | — |
| `train.py` | 训练：Huber Loss + α×置信度 MSE（α 动态：前 3 epoch 0.2，之后 0.6），CosineAnnealingLR，Early Stopping | `best_model.pt` + `scaler.pkl` |
| `inference.py` | 推理接口：`PricePredictor` 支持单日/批量/滚动预测 | `pred.npy` + `conf.npy`（调试） |
| `power_price/main.py` | 策略主入口：风险管理 → 策略搜索 → 功率计划 → 提交 CSV | `output.csv`（5664 行） |
| `power_price/strategy_wrapper.py` | 策略搜索：优先调用 C++ `.so`，降级纯 Python O(N²) | `(tc, td, profit)` |
| `power_price/risk_mgmt.py` | 风险修正：置信度折扣、分位数保守估计、收益阈值、极端价格过滤 | `adjusted_prediction` |
| `power_price/prepare_test_features.py` | Test 特征构造：中文列名映射、forecast 填充 actual、拼接历史上下文、调用 `features.py` 生成一致特征 | 完整特征 DataFrame |

---

## 3. 技术栈

| 层级 | 技术 | 用途 |
|:---|:---|:---|
| 深度学习 | PyTorch | 电价预测模型 |
| 数据处理 | pandas, xarray, numpy | 数据预处理与特征工程 |
| 数值计算 | numpy, scikit-learn | 标准化、评估指标 |
| 策略算法 | C++ / Python ctypes | 充放电策略搜索 |
| 配置管理 | YAML | 风险参数配置 (`risk_config.yaml`) |

### 模型架构（以 `model.py` 为准）

```
Input: [Batch, 672, Feature_Dim]
       ↓
Linear(Feature_Dim → hidden)
       ↓
Conv1d(hidden → hidden, kernel=3, padding=1)
       ↓
ResidualBlock(Conv1d) × N
       ↓
Conv1d(hidden → hidden, kernel=7, stride=7)   # 时间压缩: 672→96
       ↓
+ TimeEmbedding(96, hidden)                    # 打破输出同质化
       ↓
Linear(hidden → 1) per time-step              # [Batch, 96]
```

默认参数：`hidden=128`, `layers=4`, `horizon=96`。

> **Time Embedding 必要性**：早期版本去掉全局池化后改用时间压缩卷积，但 Conv1d 输出头导致 96 个时间步的预测高度同质化（std 仅 0.05）。Time Embedding 为每个输出时间步提供独立偏置，强制产生日内差异。

---

## 4. 开发规范（必须遵守）

### 4.1 防数据泄露

1. **时序划分**：训练/验证必须**按时间先后分割**，严禁随机打乱
2. **StandardScaler**：训练集 `fit_transform`，验证集/测试集只能 `transform`
3. **DataLoader**：`shuffle=False`

### 4.2 命名规范

- **目标列名**：`A`（蒙西节点电价）
- **时间列名**：`times`
- **特征列名**：英文小写 + 下划线（snake_case）
- **中文 → 英文映射**（边界条件原始列名）：
  - `系统负荷实际值/预测值` → `load_actual/forecast`
  - `风光总加实际值/预测值` → `renewable_actual/forecast`
  - `联络线实际值/预测值` → `tie_line_actual/forecast`
  - `风电/光伏/水电/非市场化机组预测值` → `wind/solar/hydro/non_market_forecast`

### 4.3 路径统一

所有中间产物必须放到指定目录：

| 目录 | 用途 |
|:---|:---|
| `power_price/data/` | 数据与中间产物 |
| `power_price/models/` | 模型权重 |
| `power_price/output/` | 策略输出 CSV |
| `power_price/test_data/` | 测试输入 |

禁止在根目录散落 `.npy` / `.json` / `.pkl` 文件。

---

## 5. 接口契约

| 输出方 | 文件 | 接收方 | 格式 |
|:---|:---|:---|:---|
| Day 1 | `power_price/data/weather_features.csv` | `Dataaligning.py` | 小时级气象特征，含 `timestamp` 列 |
| Day 2 | `power_price/data/aligned_15min_full.csv` | `features.py` | 15min 宽表，索引 `times`，目标列 `A` |
| Day 2 | `power_price/data/aligned_15min_processed.csv` | `Dataset.py` / `train.py` / `inference.py` | 特征工程后宽表，≤50 维 |
| Day 3/7 | `power_price/data/feature_mask.json` | `train.py` / `inference.py` | `list[str]`，≤50 维，**无 `A_*` 特征** |
| Day 9 | `power_price/models/best_model.pt` | `inference.py` | PyTorch `state_dict` |
| Day 9 | `power_price/data/scaler.pkl` | `inference.py` | `sklearn.StandardScaler`，维度与 `feature_mask` 一致 |
| Day 11 | `power_price/data/pred.npy` | `main.py`（调试） | `np.ndarray [96]`, `float64` |
| Day 11 | `power_price/data/conf.npy` | `main.py`（调试） | `np.ndarray [96]`, `float64` |
| 提交输入 | `power_price/test_data/test_in_feature_ori.csv` | `prepare_test_features.py` | 8 列中文原始预测特征，59 天 |
| 提交输出 | `power_price/output/output.csv` | 赛题评测 | `times, 实时价格(预测), power`，**5664 行** |

### 关键编程接口

```python
# inference.py
class PricePredictor:
    def __init__(self, model_path, scaler_path, mask_path): ...
    def predict(self, df_input: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...
    def predict_batch(self, df_input, target_dates, skip_insufficient=True, target_col=None)
        -> tuple[list[np.ndarray], list[np.ndarray], list[pd.DatetimeIndex]]: ...

# power_price/strategy_wrapper.py
def optimize_charge_discharge(
    prices: np.ndarray, confidence=None, risk_aversion=0.1, profit_threshold=0.0
) -> dict: ...

# power_price/main.py
def run_single_day(prediction, confidence=None, timestamp_index=None, config=None) -> pd.DataFrame: ...
def run_batch(predictions_list, confidences_list=None, timestamps_list=None, config=None) -> pd.DataFrame: ...
```

---

## 6. 测试期特殊约束

比赛规则规定**测试集不提供历史真实电价**。代码已按此约束实现：

1. `prepare_test_features.py` 中 test 期间 `A` 列保持 `NaN`，`curtailment_flag` 等依赖 `A` 的特征被正确置 0
2. `prepare_test_features.py` 用 `forecast` 填充 `actual`，导致 `*_forecast_error` 特征在 test 期间恒为 0（**已知信息损失**）
3. `inference.py` 的 `predict_batch` 支持滚动预测（`target_col='A'`），每预测一天后将预测电价写回输入，供下一天使用
4. `feature_mask.json` **已排除所有 `A_*` 衍生特征**（`A_lag_96`、`A_rolling_mean_8` 等），确保 test 期间模型输入不依赖历史真实电价
5. `features.py` 的 `add_strategy_features` / `add_lag_features` 在生成特征前检查列是否已存在，避免与 `Dataaligning.py` 产生重复列（如 `net_load_forecast`）

---

## 7. Agent 协作检查清单

任务交接时确认：

- [ ] **输出文件**已生成在指定路径
- [ ] **数据格式**符合接口契约（shape, dtype, 列名）
- [ ] **无数据泄露**（标准化器参数来自训练集）
- [ ] **路径一致**（无散落文件在根目录）
- [ ] **文档更新**（README / AGENTS.md / 注释反映最新实现）

---

> **配套文档**：技术细节与分阶段计划见 `Plan.md` / `Achieve.md` / `B-Plan.md` / `C_Plan.md`，人类使用说明见 `README.md`。
