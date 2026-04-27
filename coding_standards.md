# 电力现货电价预测项目 — 数据处理与接口规范

> 本文档基于现有代码（`DetectNCfile.py`、`Dataaligning.py`、`Dataset.py`）制定，供三位成员在后续开发中遵循。  
> **目标**：确保数据流、接口命名、文件组织一致，降低集成成本。

---

## 1. 项目目录结构规范

```
.
├── data/                           # 原始数据与中间产物（不入git）
│   ├── weather_raw/                # NC原始文件（成员A处理）
│   ├── weather_features.csv        # Day1 输出：6变量气象特征
│   ├── mengxi_node_price_selected.csv
│   ├── mengxi_boundary_anon_filtered.csv
│   ├── aligned_15min_full.csv      # Day2 输出：对齐后的主数据表
│   └── scaler.pkl                  # Day3 输出：StandardScaler序列化文件
├── preprocess/                     # 数据预处理脚本（成员A）
│   ├── nc_extractor.py             # NC降维脚本（替代DetectNCfile.py）
│   ├── align_pipeline.py           # 主对齐流程（Dataaligning.py的正式版）
│   └── quality_report.py           # 数据质量检查脚本
├── features.py                     # 特征工程库（成员B/C）
├── model.py                        # ResNet-MLP架构（成员B）
├── train.py / inference.py         # 训练与推理脚本（成员B）
├── strategy.py / strategy_core.cpp # 充放电策略（成员C）
├── risk_mgmt.py                    # 风险修正模块（成员C）
├── main.py                         # 端到端入口（成员C）
└── utils/                          # 公共工具
    └── data_schema.py              # 数据schema常量定义（三人共享）
```

### 约定
- **所有中间CSV必须包含 `times` 列作为索引**，且列名统一为小写+下划线。
- **输出文件路径使用 `pathlib.Path`**，禁止硬编码字符串拼接路径。
- **scaler、模型权重、schema 必须持久化到磁盘**，供跨阶段复用。

---

## 2. 命名规范（基于现有代码）

### 2.1 文件命名
- Python 模块：全小写，下划线分隔，如 `dataaligning.py`、`quality_report.py`
- C++ 源文件：下划线分隔，如 `strategy_core.cpp`
- 数据文件：下划线分隔，阶段标注清晰，如 `aligned_15min_full.csv`

### 2.2 类命名
- 数据对齐类：`PowerDataAligner`
- 配置类：`DataAlignmentConfig`（若后续增加模型配置，沿用 `ModelConfig` / `TrainConfig`）
- Dataset 类：`PowerPriceDataset`
- 数据模块类：`PowerPriceDataModule`

### 2.3 函数/方法命名
- 动词开头，下划线分隔，返回类型注解必须写：
  ```python
  def load_raw_data(self, ...) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...
  def resample_to_15min(self, df: pd.DataFrame, data_type: str, method: str = 'linear') -> pd.DataFrame: ...
  def calculate_power_system_features(self, df: pd.DataFrame) -> pd.DataFrame: ...
  def get_train_loader(self) -> DataLoader: ...
  ```

### 2.4 变量/列名命名
| 类别 | 命名规则 | 示例 |
|------|----------|------|
| 原始实际值 | `{entity}_actual` | `load_actual`, `wind_actual` |
| 原始预测值 | `{entity}_forecast` | `load_forecast`, `solar_forecast` |
| 衍生物理特征 | 语义清晰，避免缩写 | `bidding_space_actual`, `renewable_penetration` |
| 时间循环编码 | `{granularity}_sin` / `{granularity}_cos` | `time_sin`, `dow_cos`, `month_sin` |
| 标记/布尔特征 | `is_{condition}` | `is_weekend`, `is_peak_hour`, `is_extreme_renewable` |
| 误差/偏差 | `{entity}_forecast_error` | `wind_forecast_error` |
| 比率/比例 | `{entity}_ratio` | `tie_line_ratio` |
| 滚动统计 | `rolling_{stat}_{window}`（新增时） | `rolling_std_8`, `rolling_mean_96` |

---

## 3. 数据格式与接口契约

### 3.1 阶段一：NC → 气象CSV（成员A，Day1）

**输入**：`weather_raw/*.nc`（xarray Dataset）  
**输出**：`weather_features.csv`

输出要求：
- 必须包含 `timestamp` 列（DatetimeIndex），由 xarray 的 `time` 维度展平而来
- 列名使用小写+下划线，如 `t2m_mean`, `t2m_max`, `t2m_min`, `t2m_std`
- 时区处理：NC 文件通常无 TZ，视为 UTC，转换到 `Asia/Shanghai`
- **禁止在NC处理阶段做插值**，保留原始时间分辨率（通常是1小时），插值在 Day2 统一完成

```python
# 推荐列名（6类气象变量）
WEATHER_BASE_COLS = [
    't2m_mean', 't2m_max', 't2m_min', 't2m_std',
    'ghi_mean', 'ghi_max', 'ghi_min', 'ghi_std',
    'sp_mean',  'sp_max',  'sp_min',  'sp_std',
    'tcc_mean', 'tcc_max', 'tcc_min', 'tcc_std',
    'wspd_mean', 'wspd_max'
]
```

### 3.2 阶段二：多源对齐 → 主数据表（成员A，Day2）

**输入**：电价CSV、边界CSV、气象CSV  
**输出**：`aligned_15min_full.csv`

核心接口：`PowerDataAligner.align_all_data(...)`

#### 对齐后 DataFrame 规范
| 属性 | 要求 |
|------|------|
| 索引名 | `times`（已设为 `DatetimeIndex`） |
| 时间分辨率 | 严格15分钟，每日96点 |
| 时区 | `Asia/Shanghai`（输出前去掉时区或保持统一） |
| 目标列 | 默认 `price` 或具体节点名如 `A` |
| 缺失值 | 核心列（电价、负荷、竞价空间）不允许有缺失 |

#### 现有特征分类清单（必须在schema中维护）
```python
# utils/data_schema.py 建议内容
PRICE_COLS = ['price', 'A']  # 蒙西节点电价列

BOUNDARY_ACTUAL_COLS = [
    'load_actual', 'renewable_actual', 'tie_line_actual',
    'wind_actual', 'solar_actual', 'hydro_actual', 'non_market_actual'
]

BOUNDARY_FORECAST_COLS = [
    'load_forecast', 'renewable_forecast', 'tie_line_forecast',
    'wind_forecast', 'solar_forecast', 'hydro_forecast', 'non_market_forecast'
]

DERIVED_POWER_COLS = [
    'bidding_space_actual', 'bidding_space_forecast',
    'net_load_actual', 'net_load_forecast',
    'renewable_penetration', 'supply_demand_gap', 'tie_line_ratio',
    'load_forecast_error', 'wind_forecast_error', 'solar_forecast_error',
    'renewable_forecast_error', 'penetration_sq', 'is_extreme_renewable',
    'midday_solar_risk', 'evening_ramp_stress',
    'expected_price_volatility', 'deep_peak_risk',
    'tie_line_margin', 'is_high_export',
    'wind_solar_correlation_7d', 'wind_solar_instant_ratio',
    'curtailment_flag', 'heating_season_rigid'
]

TIME_FEATURE_COLS = [
    'time_sin', 'time_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend', 'is_peak_hour'
]
```

### 3.3 阶段三：Dataset → Tensor（成员A，Day3）

**输入**：`aligned_15min_full.csv`  
**输出**：`torch.utils.data.DataLoader`

#### 核心接口
```python
class PowerPriceDataset(Dataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: torch.FloatTensor, shape (lookback_window, n_features)
            y: torch.FloatTensor, shape (forecast_horizon,)
        """

class PowerPriceDataModule:
    def get_train_loader(self) -> DataLoader: ...
    def get_val_loader(self) -> DataLoader: ...
    def get_feature_dim(self) -> int: ...
```

#### 严格约定（防止数据泄露）
1. **时间序列分割**：`train_df = df[df.index <= train_end_date]`，`val_df = df[df.index >= val_start_date]`，**严禁 `train_test_split` 随机打乱**。
2. **标准化器共享**：训练集 `fit_transform`，验证集/测试集必须使用 `transform(scaler=train_scaler)`。
3. **DataLoader 中 `shuffle=False`**：时间序列数据不允许打乱顺序。
4. **scaler 必须保存**：使用 `joblib.dump(scaler, 'data/scaler.pkl')`，供推理阶段复用。

#### A → B 的数据接口（Day3 锁定）
```python
# 数据模块输出的 batch 维度规范
x_tensor: torch.Tensor  # [Batch, Seq_Len, N_features]
                        # Seq_Len = 672 (7天×96点)
                        # N_features ≤ 50（B/C 增加特征后需同步更新此上限）
y_tensor: torch.Tensor  # [Batch, 96], 实际电价（未归一化）
```

### 3.4 阶段四：特征工程扩展（成员B/C，Day4-7）

新增特征必须通过 `features.py` 中的函数注入，**禁止直接修改 `Dataset.py` 的内部逻辑**。

#### 推荐接口
```python
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """成员B：sin/cos、滞后特征、滚动统计"""
    ...

def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """成员C：净负荷、渗透率、风险标记等物理特征"""
    ...
```

#### 特征注入流程
```python
# align_pipeline.py 中，在保存 csv 前调用
merged = aligner.align_all_data(...)
merged = add_temporal_features(merged)      # B负责
merged = add_strategy_features(merged)      # C负责
merged.to_csv(output_path)
```

#### 新增特征命名规范
- 滞后特征：`{col}_lag_{n}`，如 `price_lag_96`（前一天同期）
- 滚动统计：`{col}_rolling_{stat}_{window}`，如 `price_rolling_mean_96`
- 差分特征：`{col}_diff_{n}`，如 `price_diff_1`
- 物理组合特征：语义优先，如 `net_load_forecast`, `supply_demand_gap`

### 3.5 阶段五：模型输出（成员B，Day8-11）

#### B → C 的接口规范（Day11 必须确定）
```python
prediction: np.ndarray  # shape [96], dtype float64
                        # 含义：未来一天的96个15分钟电价预测值
confidence: np.ndarray  # shape [96], dtype float64, optional
                        # 含义：预测置信度/标准差，用于风险调节
```

模型推理函数推荐签名：
```python
def predict(model, x: torch.Tensor, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        x: 输入特征，shape (1, 672, F) 或 (672, F)
    Returns:
        prediction: [96] 电价预测
        confidence: [96] 预测不确定度
    """
```

### 3.6 阶段六：策略算法（成员C，Day12-15）

#### Python 接口（优先）
```python
def optimize_charge_discharge(
    prices: np.ndarray,           # [96] 预测电价
    confidence: np.ndarray,       # [96] 可选
    risk_params: dict             # 风险参数字典
) -> dict:
    """
    Returns:
        {
            'charge_start': int,      # 充电开始时段索引 [0,95]
            'discharge_start': int,   # 放电开始时段索引 [0,95]
            'expected_profit': float, # 预期收益
            'risk_adjusted_return': float
        }
    """
```

#### C++ 接口（若成员C选择用C++实现核心搜索）
```cpp
// strategy_core.cpp
extern "C" {
    double find_best_strategy(
        const double* prices, int n,
        int* charge_start, int* discharge_start,
        double* expected_profit
    );
}
```

封装层 `strategy_wrapper.py`：
```python
import ctypes
import numpy as np

lib = ctypes.CDLL('./strategy_core.so')

def optimize_charge_discharge(prices: np.ndarray) -> dict:
    n = len(prices)
    charge = ctypes.c_int(0)
    discharge = ctypes.c_int(0)
    profit = ctypes.c_double(0.0)
    lib.find_best_strategy(
        prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        ctypes.byref(charge),
        ctypes.byref(discharge),
        ctypes.byref(profit)
    )
    return {
        'charge_start': charge.value,
        'discharge_start': discharge.value,
        'expected_profit': profit.value
    }
```

---

## 4. 代码风格规范（针对C++背景团队）

### 4.1 Python 代码必须遵守的约定
1. **类型注解**：所有函数参数和返回值必须写类型注解（现有代码已做到，继续保持）。
2. **不要写裸循环处理 DataFrame**：能用 `pandas`/`numpy` 向量化的操作，严禁手写 `for` 循环。
   ```python
   # 错误示例
   for i in range(len(df)):
       df.loc[i, 'new_col'] = df.loc[i, 'a'] + df.loc[i, 'b']

   # 正确示例
   df['new_col'] = df['a'] + df['b']
   ```
3. **异常与边界检查**：C++ 风格的防御式编程是优点，但必须用 Pythonic 方式表达：
   ```python
   if df.empty:
       raise ValueError("DataFrame is empty")
   if 'price' not in df.columns:
       raise KeyError(f"Missing required column: price")
   ```
4. **无穷值与缺失值处理**：所有除法后必须 `replace([np.inf, -np.inf], 0)`，滚动计算必须带 `min_periods=1`。
5. **配置参数集中化**：不要散落魔法数字。已有 `DataAlignmentConfig`，模型/策略侧也请创建 `ModelConfig`、`StrategyConfig`。

### 4.2 日志与打印
- 调试信息使用 `print`，但需带统一前缀：`[ModuleName] message`
- 现有示例：`print("=== 计算电力系统衍生特征 ===")`、`print(f"[{mode}] Dataset initialized:")`
- 后续建议逐步迁移到 `logging` 模块，但短期内保持前缀一致即可。

### 4.3 常量定义
所有字符串常量、列名列表、阈值统一放到 `utils/data_schema.py`，禁止在函数内部硬编码：
```python
# 正确：从公共模块导入
from utils.data_schema import BOUNDARY_ACTUAL_COLS, TIME_FEATURE_COLS

# 错误：在函数里写死
exclude_cols = ['price', 'date', 'is_weekend', 'is_peak_hour']
```

---

## 5. 关键集成节点检查清单

### Day 3（Schema 锁定）
- [ ] `aligned_15min_full.csv` 列名固定，导出 `data_schema.json`
- [ ] `PowerPriceDataModule` 输出 `x.shape == (batch, 672, F)` 且 `y.shape == (batch, 96)`
- [ ] `scaler.pkl` 已保存，验证集复用无误

### Day 7（特征工程完成）
- [ ] 特征总数 `N_features ≤ 50`
- [ ] `features.py` 中所有函数都有单元测试（至少跑通不报错）
- [ ] 更新 `data_schema.py` 中的特征列表

### Day 11（模型输出可用）
- [ ] `inference.py` 能输出 `prediction: np.ndarray [96]` 和 `confidence: np.ndarray [96]`
- [ ] 模型权重文件已保存到 `checkpoints/`

### Day 15（最终集成）
- [ ] `main.py` 端到端跑通：NC/CSV → align → Dataset → model → strategy → result
- [ ] `requirements.txt` 和 `Dockerfile` 已提交

---

## 6. 附：现有代码中的具体约定摘录

### 6.1 时间窗口常量（已硬编码在 Dataset.py）
```python
lookback_window = 672    # 7天 × 96点/天
forecast_horizon = 96    # 1天 × 96点/天
batch_size = 32
```
> **注意**：若成员B/C 需要调整窗口（如改为14天历史），必须与成员A同步修改 `Dataset.py` 和 `Dataaligning.py` 中的验证逻辑。

### 6.2 时区处理（Dataaligning.py 已约定）
- 原始无TZ → 视为 UTC → 转 `Asia/Shanghai`
- 原始有TZ → `tz_convert('Asia/Shanghai')`
- Dataset 加载时会 `tz_localize(None)` 以去掉时区便于日期比较

### 6.3 数据填充策略（Dataaligning.py 已约定）
| 数据类型 | 策略 | 原因 |
|----------|------|------|
| 电价 (`price`) | `ffill().bfill()` | 严禁插值，保持跳变 |
| 边界实际值 | `ffill().bfill()` | 保持实际值真实性 |
| 气象数据 | `interpolate('linear')` + 均值填充 | 物理连续变量允许插值 |

### 6.4 价格列名特殊处理
`Dataset.py` 中 `target_col` 默认 `'price'`，但示例中使用 `'A'`（蒙西节点）。**请成员A在 Day3 最终确定目标列名，并在 `data_schema.py` 中注册**。

---

## 7. 快速参考：新增代码时的自测问题

在提交任何 PR 前，请逐条确认：

1. **接口**：我的函数签名是否带类型注解？返回值是否单一明确？
2. **命名**：列名是否全小写+下划线？是否与 `data_schema.py` 冲突？
3. **数据泄露**：是否对验证集做了任何拟合操作？（如 fit_transform、按全局统计填充）
4. **维度**：输出的 Tensor/ndarray shape 是否符合 A→B→C 的接口契约？
5. **C++思维陷阱**：是否有对 DataFrame/Series 的手写 for 循环？是否可改为 `apply`/`rolling`/`shift`？
6. **边界**：是否处理了 divide-by-zero、NaN、Inf、空表情况？
7. **持久化**：是否有需要跨阶段复用的对象（scaler、schema、模型权重）已保存到磁盘？
