# AGENTS.md - AI Agent 开发指南

> 本文件为AI Agent提供项目背景、技术路线和协作规范，确保多Agent协同开发时保持一致性。

---

## 一、项目概述

### 1.1 赛题目标

构建**电力现货电价预测模型**与**储能充放电策略优化算法**，在蒙西地区电力现货市场实现收益最大化。

**核心任务**：
- **预测任务**：预测未来24小时（96个15分钟分辨率点）的实时电价
- **策略任务**：在储能约束条件下制定充放电计划，最大化收益

**评测指标**：测试集所有天数收益的平均值

```
Profit = Σ(t=0 to 95) P_t × E_t
```

其中：
- `P_t`: 真实实时电价
- `E_t`: 充放电功率（+1000放电，-1000充电，0不操作）

### 1.2 储能约束条件（硬性约束）

| 参数 | 值 |
|:---|:---|
| 储能容量 | 8000 |
| 充放电功率 | ±1000 |
| 单次持续时间 | 连续8个时间点（2小时） |
| 初始SOC | 0 |
| 每日操作限制 | 最多1次完整充放电（充+放） |
| 充电开始时间 | 0 ≤ t_c ≤ 80 |
| 放电开始时间 | t_c + 8 ≤ t_d ≤ 88 |

---

## 二、角色分工与职责

本项目采用**三成员协作模式**，每个Agent负责一个阶段：

### 成员A - 数据预处理工程师
**负责阶段**：Day 1-3（阶段一：数据预处理与管线搭建）

**核心职责**：
- NC气象数据降维与特征提取
- 多源异构数据对齐（15分钟粒度）
- PyTorch Dataset封装与标准化

**关键产出**：
- `weather_features.csv`（气象特征表）
- `aligned_15min_full.csv`（对齐后的宽表）
- `scaler.pkl`（标准化器）

---

### 成员B - 模型工程师
**负责阶段**：Day 4-11（阶段二：深度特征工程 + 模型开发训练）

**核心职责**：
- 物理特征工程（净负荷、渗透率、预测偏差等）
- ResNet-MLP模型架构实现
- 训练流程与推理接口

**关键产出**：
- `feature_mask.json`（筛选后的特征列表）
- `model.py`（模型架构）
- `best_model.pt`（训练好的模型权重）
- `pred.npy` / `conf.npy`（预测结果与置信度）

---

### 成员C - 策略算法工程师
**负责阶段**：Day 12-15（阶段三：策略算法与鲁棒性优化）

**核心职责**：
- C++核心搜索算法实现
- 风险溢价与策略修正
- 模型集成与端到端封装

**关键产出**：
- `strategy_core.cpp/.so`（C++核心搜索库）
- `strategy_wrapper.py`（Python封装）
- `risk_mgmt.py`（风险管理）
- `output.csv`（最终提交格式）

---

## 三、技术栈与架构

### 3.1 核心技术栈

| 层级 | 技术 | 用途 |
|:---|:---|:---|
| 深度学习 | PyTorch | 电价预测模型 |
| 数据处理 | pandas, xarray, numpy | 数据预处理与特征工程 |
| 数值计算 | numpy, scikit-learn | 标准化、评估指标 |
| 策略算法 | C++ / Python | 充放电策略搜索 |
| 配置管理 | YAML | 风险参数配置 |

### 3.2 模型架构

**ResNet-MLP架构**：
```
Input: [Batch, 672, Feature_Dim]  # 672=过去7天15分钟点数
       ↓
Flatten + Linear(672×F → 512)
       ↓
Residual Block × N
       ↓
Output Heads:
  - Price Head: [Batch, 96]       # 电价预测
  - Confidence Head: [Batch, 96]  # 置信度估计(softplus激活)
```

### 3.3 数据流架构

```
raw_data/
├── weather_raw/*.nc          → Day 1 → weather_features.csv
├── mengxi_node_price.csv     → Day 2 → aligned_15min_full.csv
└── mengxi_boundary.csv              ↓
                               Day 3 → Dataset + scaler.pkl
                                      ↓
                               Day 4-7 → feature_mask.json
                                      ↓
                               Day 8-10 → best_model.pt
                                      ↓
                               Day 11 → pred.npy + conf.npy
                                      ↓
               test_in_feature_ori.csv → prepare_test_features.py → 完整特征
                                      ↓
                               Day 12-15 → output.csv
```

---

## 四、开发规范与约束

### 4.1 数据约束（必须遵守）

1. **时序划分**：训练/验证必须**按时间先后分割**，严禁随机打乱
2. **防数据泄露**：StandardScaler只能在训练集`fit`，验证集只能`transform`
3. **路径统一**：所有中间产物必须放到指定目录，禁止散落文件

### 4.2 接口契约

| 阶段 | 输出文件 | 接收方 | 数据格式 |
|:---|:---|:---|:---|
| Day 1 → Day 2 | `weather_features.csv` | Dataaligning.py | 小时级气象特征表 |
| Day 2 → Day 3 | `aligned_15min_full.csv` | Dataset.py | 15min统一宽表 |
| Day 3/7 → Day 8 | `feature_mask.json` | train.py | ≤50维特征名列表 |
| Day 9 → Day 11 | `best_model.pt` | inference.py | PyTorch state_dict |
| Day 9 → Day 11 | `scaler.pkl` | inference.py | sklearn StandardScaler |
| Day 11 → Day 15 | `pred.npy` | main.py | np.ndarray [96] float64 |
| Day 11 → Day 15 | `conf.npy` | main.py | np.ndarray [96] float64 |
| 提交输入 | `test_in_feature_ori.csv` | `prepare_test_features.py` | 原始预测特征（8列中文） |
| 提交输出 | `output.csv` | 赛题评测 | `times, 实时价格(预测), power` |

### 4.3 命名规范

- **目标列名**：电价目标列使用 `A`（蒙西节点）
- **时间列名**：统一使用 `times`
- **特征列名**：使用英文小写+下划线（snake_case）

---

## 五、各阶段详细任务

### 阶段一：数据预处理（Day 1-3）

#### Day 1: 气象数据降维
**任务**：处理NC文件，提取空间统计特征

**输入**：`power_price/data/weather_raw/*.nc`
**输出**：`power_price/data/weather_features.csv`

**关键操作**：
```python
# 使用xarray加载NC文件
# 提取mean, max, min, std统计量
# 构建派生特征：风速合成、太阳辐射潜力
```

#### Day 2: 多源数据对齐
**任务**：统一时区、重采样到15分钟、合并所有数据源

**输入**：
- 电价：`mengxi_node_price_selected.csv`
- 边界条件：`mengxi_boundary_anon_filtered.csv`
- 气象：`weather_features.csv`

**输出**：`power_price/data/aligned_15min_full.csv`

**关键特征**：
- 竞价空间、净负荷、新能源渗透率
- 预测误差特征
- 正弦-余弦时间编码

#### Day 3: PyTorch Dataset封装
**任务**：滑动窗口构建、标准化、训练/验证分割

**参数**：
- `lookback_window`: 672点（过去7天）
- `forecast_horizon`: 96点（未来1天）
- 分割比例：前10个月训练，第11个月验证

---

### 阶段二：特征工程与模型（Day 4-11）

#### Day 4: 周期性与时间特征
- 正余弦编码：Hour (0-23), DayOfWeek (0-6)
- 节假日One-hot编码

#### Day 5: 电力平衡物理特征
- **净负荷** = 负荷预测值 - 风电预测 - 光伏预测
- **能源渗透率** = (风电+光伏) / 负荷预测
- **预测偏差** = T-96时刻预测值与实际值差值

#### Day 6: 滞后与统计窗口
- 滞后特征：`price_lag_96`（昨日同期）、`price_lag_672`（上周同期）
- 滚动统计：过去24小时电价波动率、均值

#### Day 7: 特征筛选
- Pearson/Spearman相关性计算
- 筛选Top-50特征，生成`feature_mask.json`

#### Day 8: 模型架构
- ResNet-MLP实现
- 双头输出：电价预测 + 置信度估计

#### Day 9-10: 训练
- Loss: Huber Loss + 0.2 × 置信度MSE
- Optimizer: AdamW (lr=1e-3)
- Scheduler: CosineAnnealingLR
- Early Stopping: 验证集Loss连续10个Epoch不下降

#### Day 11: 推理
- 加载最后672行作为历史窗口
- 输出`pred.npy`和`conf.npy`

---

### 阶段三：策略算法（Day 12-15）

#### Day 12: 基础搜索算法
**算法**：O(N²)暴力枚举

```cpp
// 目标：max(Σ放电时段电价 - Σ充电时段电价)
for tc in [0, 80]:
    for td in [tc+8, 88]:
        profit = sum(P[td:td+8]) - sum(P[tc:tc+8])
        track max profit
```

#### Day 13: 风险溢价与策略修正
- **波动率调节**：高波动区间向平稳区域平移
- **收益阈值**：预测收益<X元时放弃操作

#### Day 14: 模型集成
- 3-5个不同Seed模型
- 加权平均预测结果

#### Day 15: 最终打包
- 封装全流程：数据读取→特征工程→模型推理→策略搜索→格式输出
- 确保10-30分钟内完成推理

---

## 六、文件清单与快速参考

### 根目录关键文件

| 文件 | 用途 |
|:---|:---|
| `Plan.md` | 完整15天执行计划 |
| `README.md` | 详细使用说明与接口文档 |
| `coding_standards.md` | 编码规范 |
| `run_pipeline.py` | 端到端流水线入口 |

### 阶段一文件（成员A）

| 文件 | 用途 |
|:---|:---|
| `Test_Weather_processor.py` | Day 1: NC气象降维 |
| `Dataaligning.py` | Day 2: 多源15min对齐 |
| `Dataset.py` | Day 3: PyTorch Dataset封装 |

### 阶段二文件（成员B）

| 文件 | 用途 |
|:---|:---|
| `features.py` | Day 4-7: 特征工程库 |
| `model.py` | Day 8: ResNet-MLP架构 |
| `train.py` | Day 9-10: 训练脚本 |
| `inference.py` | Day 11: 推理接口 |

### 阶段三文件（成员C）

| 文件 | 用途 |
|:---|:---|
| `power_price/prepare_test_features.py` | Day 15: 原始test特征→完整50维特征 |
| `power_price/strategy_core.cpp` | Day 12: C++核心搜索 |
| `power_price/strategy_wrapper.py` | Day 12: Python封装 |
| `power_price/risk_mgmt.py` | Day 13: 风险管理 |
| `power_price/ensemble_strategy.py` | Day 14: 多模型集成 |
| `power_price/main.py` | Day 15: 策略生成主入口 |

---

## 七、常用命令速查

### 环境安装
```bash
pip install torch pandas numpy scikit-learn xarray joblib pyyaml
```

### 编译C++策略库
```bash
cd power_price
g++ -O2 -shared -fPIC -o strategy_core.so strategy_core.cpp
```

### 端到端运行
```bash
# 全流程（训练+推理+策略）
python run_pipeline.py --all --epochs 10

# 分阶段
python run_pipeline.py --stage1  # 数据预处理
python run_pipeline.py --stage2  # 特征工程
python run_pipeline.py --stage3 --epochs 20  # 模型训练
python run_pipeline.py --stage4-demo  # 策略演示

# 从原始 test 特征直接生成提交结果（训练完成后使用）
python run_pipeline.py --input-csv power_price/test_data/test_in_feature_ori.csv --output power_price/output/output.csv
```

### 手动串联
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

# 3. 训练
python train.py

# 4. 推理（开发调试用）
python inference.py

# 5. 策略生成 —— 方式A：从 .npy 文件（单日/批量）
cd power_price
python main.py --prediction data/pred.npy --confidence data/conf.npy --output output/output.csv

# 5. 策略生成 —— 方式B：从原始 test 特征 CSV 端到端（推荐，训练完成后使用）
cd power_price
python main.py --input-csv test_data/test_in_feature_ori.csv --output output/output.csv
```

---

## 八、常见问题

### Q1: 目标列名不一致？
本项目电价目标列使用 **`A`**。若数据使用`price`作为列名，请同步修改各模块的`target_col`/`price_col`参数。

### Q2: 时区如何处理？
- NC气象数据：视为UTC后转`Asia/Shanghai`
- Dataset加载时：去除时区以便日期比较

### Q3: 如何防止数据泄露？
1. 训练/验证按时间先后分割，严禁随机打乱
2. StandardScaler只在训练集`fit`，验证集只能`transform`

### Q4: 路径不一致？
所有中间产物必须统一放到：
- `power_price/data/`
- `power_price/models/`
- `power_price/output/`

### Q5: 测试期为何没有真实电价？
根据比赛规则，**测试集中不提供历史真实电价**。代码已按此约束实现：
1. `prepare_test_features.py` 中 test 期间 `A` 列保持 `NaN`，`curtailment_flag` 等依赖 `A` 的特征被正确置 0
2. `inference.py` 的 `predict_batch` 支持滚动预测（`target_col='A'`），每预测一天后将预测电价写回输入，供下一天使用
3. 当前 `feature_mask.json` 选出的 50 维特征不直接包含历史电价滞后项，因此模型输入本身不依赖前一日真实电价

---

## 九、Agent协作检查清单

当一个Agent完成任务交接给下一个Agent时，请确认：

- [ ] **输出文件**已生成在指定路径
- [ ] **数据格式**符合接口契约（shape, dtype, 列名）
- [ ] **无数据泄露**（标准化器参数来自训练集）
- [ ] **路径一致**（无散落文件在根目录）
- [ ] **文档更新**（README/注释反映最新实现）

---

> **注意**：本文件与`Plan.md`和`README.md`配套使用。技术细节详见`Plan.md`，使用说明详见`README.md`。
