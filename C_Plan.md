 我将为你（成员C：策略算法师）制定一份详细的执行计划。首先让我快速浏览技能文档，确保计划符合项目规范。
 这份技能文档与当前任务无关。让我直接基于已上传的项目文档为你制定详细的策略算法师执行计划。

---

# 成员C（策略算法师）详细执行计划书

## 1. 角色定位与核心职责

| 维度 | 说明 |
|:---|:---|
| **核心任务** | Day 5物理特征辅助 + Day 12-15储能充放电策略全流程 |
| **技术特色** | 利用C++背景实现高性能策略搜索算法，Python封装调用 |
| **关键交付** | 收益最大化的充放电计划生成器 + 端到端集成管线 |
| **协作接口** | 接收B的`prediction[96]`+`confidence[96]`，输出`power[96]`策略 |

---

## 2. 前置依赖与输入规范（Day 11前必须确认）

### 2.1 从成员B接收的数据契约

```yaml
输入接口 (Day 11 联调确认):
  prediction: 
    type: np.ndarray
    shape: [96]
    dtype: float64
    含义: 次日00:00-23:45的电价预测值 (15分钟分辨率)
    取值范围: 蒙西现货电价通常 -50 ~ 1500 元/MWh
    
  confidence:
    type: np.ndarray  
    shape: [96]
    dtype: float64
    含义: 预测标准差或置信度评分 (越高=不确定性越大)
    可选: 若B未提供，默认填充0.1*prediction.std()
    
  timestamp_index:
    type: pd.DatetimeIndex
    freq: 15分钟
    tz: Asia/Shanghai
```

### 2.2 从成员A接收的数据契约（Day 5辅助特征阶段）

```yaml
辅助输入 (Day 5 物理特征计算):
  aligned_data: pd.DataFrame
    路径: power_price/data/aligned_15min_full.csv
    必需列: 
      - load_forecast, renewable_forecast, hydro_forecast
      - wind_actual, solar_actual, price (历史)
      - bidding_space_forecast, net_load_forecast
```

---

## 3. 分阶段执行计划

### 阶段一：物理特征辅助（Day 5，协作支持）

**目标**：计算与储能策略强相关的物理特征，注入到主数据表中

| 步骤 | 任务内容 | 技术要点 | 输出列名 |
|:---|:---|:---|:---|
| 5.1 | 净负荷深度调峰风险标记 | 高风电+低负荷+低温(供暖期)组合条件 | `deep_peak_risk` |
| 5.2 | 竞价空间波动率代理 | 未来2小时滚动标准差（储能充放电窗口） | `expected_price_volatility` |
| 5.3 | 极端渗透率标记 | renewable_penetration > 0.6 标记为极端 | `is_extreme_renewable` |
| 5.4 | 分时段供需结构 | 午间光伏风险(10-14点)、晚高峰爬坡压力(18-22点) | `midday_solar_risk`, `evening_ramp_stress` |
| 5.5 | 联络线裕度计算 | 容量 - 实际功率绝对值（评估外送能力） | `tie_line_margin` |

**协作动作**：
- 向成员A提交`add_strategy_features()`函数，集成到`align_pipeline.py`
- 确认特征列名符合`coding_standards.md`规范（小写+下划线）
- 验证特征计算无数据泄露（不使用未来信息）

---

### 阶段二：基础策略搜索算法（Day 12，核心启动）

**目标**：实现O(N²)暴力搜索，找到满足约束的最优充放电时机

#### 12.1 约束条件形式化

```yaml
储能系统约束:
  容量: 8000 MWh (固定)
  功率: ±1000 MW (放电+/充电-)
  持续时间: 连续8个时间点 (2小时)
  初始SOC: 0
  
操作限制:
  每日最多: 1次完整"充+放"循环 或 不操作
  充电开始: tc ∈ [0, 80] (确保8点充电+8点放电不越界)
  放电开始: td ∈ [tc+8, 88] (放电必须在充电结束后)
  
收益计算:
  Profit = Σ(Pt * Et) for t=0..95
  其中 Et ∈ {+1000(放电), -1000(充电), 0(不操作)}
```

#### 12.2 搜索空间定义

```yaml
搜索策略:
  模式: 双层循环枚举所有合法(tc, td)组合
  
  第一层 (充电窗口):
    tc从0到80遍历 (共81个可能的充电起始点)
    充电时段: [tc, tc+7], 功率=-1000
    
  第二层 (放电窗口):
    对每个tc, td从tc+8到88遍历
    放电时段: [td, td+7], 功率=+1000
    
  总搜索复杂度: O(81×81) ≈ 6561次评估/天
  
  评估函数: 计算该(tc,td)组合下的理论收益
    profit = sum(price[td:td+8])*1000 - sum(price[tc:tc+8])*1000
```

#### 12.3 C++核心实现方案（推荐）

考虑到团队C++背景，建议用C++实现核心搜索，Python封装：

```cpp
// strategy_core.cpp 接口设计
struct StrategyResult {
    int charge_start;      // 充电开始索引 [0,80]
    int discharge_start;   // 放电开始索引 [8,88]
    double expected_profit; // 预期收益 (元)
    double confidence_adjusted_return; // 风险调整后收益
    bool execute;          // 是否执行交易 (收益>阈值时为true)
};

// 主搜索函数
StrategyResult optimize_strategy(
    const double* prices,      // [96] 预测电价
    const double* confidence,  // [96] 预测置信度 (可为nullptr)
    int n,                     // 固定96
    double risk_aversion,      // 风险厌恶系数 (默认0.1)
    double profit_threshold      // 最低执行收益阈值 (默认0)
);
```

**Day 12 交付物**：
- `strategy_core.cpp`：纯C++搜索实现，O2优化编译为`.so`
- `strategy_wrapper.py`：ctypes封装层，提供Python接口
- 单元测试：验证约束满足（tc≤80, td≥tc+8, td≤88）

---

### 阶段三：风险溢价与鲁棒性策略（Day 13）

**目标**：引入预测不确定性，构建风险调整后的决策逻辑

#### 13.1 风险修正模型

```yaml
风险来源:
  1. 预测误差: B提供的confidence越高，实际收益偏离预期的风险越大
  2. 极端电价: 蒙西现货可能出现负电价或尖峰电价，需特殊处理
  3. 执行风险: 预测与实际的系统性偏差

风险修正策略:

  A. 置信度折扣 (Confidence Discounting):
     adjusted_profit = expected_profit - λ × Σ(confidence[t] × |Et|)
     其中λ为风险厌恶系数，建议0.05-0.2
     
  B. 分位数保守估计 (Quantile-based):
     不采用点预测，而是用 prediction - z×confidence 作为保守估计
     z=1.28对应80%置信度，z=1.96对应95%
     
  C. 收益阈值机制 (Profit Threshold):
     只有当 expected_profit > threshold 时才执行交易
     threshold建议: 历史收益分布的25%分位数 或 固定值(如5000元)
     
  D. 极端价格过滤:
     若预测出现负电价时段，强制标记为充电机会
     若预测出现>800元/MWh时段，强制标记为放电机会
```

#### 13.2 多情景模拟（Monte-Carlo验证）

```yaml
鲁棒性检验:
  输入: prediction[96], confidence[96]
  
  模拟方法:
    对每个候选(tc,td)组合，生成K个价格情景 (K=100)
    情景生成: price_scenario = prediction + ε, ε~N(0, confidence²)
    
  评估指标:
    - 期望收益: mean(profit_k)
    - 收益波动: std(profit_k)  
    - 下行风险: percentile(profit_k, 10) < 0 的概率
    - 夏普比率: mean(profit_k) / std(profit_k)
    
  决策规则:
    选择夏普比率最高且下行风险<20%的策略
```

**Day 13 交付物**：
- `risk_mgmt.py`：风险修正模块，实现上述A-D策略
- `monte_carlo_validator.py`：情景模拟与鲁棒性检验
- 风险参数配置文件 `risk_config.yaml`（便于调参）

---

### 阶段四：模型集成与多策略融合（Day 14）

**目标**：支持多模型预测输入，实现集成策略

#### 14.1 多模型加权接口

```yaml
输入扩展 (若B提供多模型预测):
  predictions: List[np.ndarray]  # M个模型的预测，每个[96]
  confidences: List[np.ndarray]   # M个模型的置信度
  model_weights: np.ndarray       # M个权重，sum=1
  
集成策略:
  1. 简单平均: ensemble = Σ(w_i × pred_i)
  2. 方差加权: w_i ∝ 1/variance_i (低方差模型权重更高)
  3. 分位数融合: 取各模型预测的分位数（如中位数更稳健）
  
  推荐: 先采用简单平均，若时间允许实现分位数融合
```

#### 14.2 策略-模型反馈闭环

```yaml
反馈机制 (可选增强):
  记录每日: 预测收益 vs 实际收益
  计算: 系统性偏差 = mean(actual_profit - predicted_profit)
  
  自适应调整:
    若连续3天系统性偏差>20%，自动上调风险厌恶系数λ
    若连续3天未执行交易但事后看应执行，下调收益阈值
```

**Day 14 交付物**：
- `ensemble_strategy.py`：多模型集成接口
- `adaptive_risk.py`：自适应风险参数调整（可选）
- 集成测试报告：验证多模型输入下的策略稳定性

---

### 阶段五：全流程打包与部署（Day 15）

**目标**：端到端管线打通，生成符合提交格式的输出

#### 15.1 主入口设计 (`main.py`)

```yaml
端到端流程:
  1. 加载配置 (paths, risk_params)
  2. 调用B的inference模块获取 prediction[96]
  3. 调用C的策略模块获取最优(tc, td)
  4. 生成 power[96] 序列:
       - 充电时段 [tc:tc+8]: -1000
       - 放电时段 [td:td+8]: +1000
       - 其余: 0
  5. 格式化为提交CSV:
       - 列名: times, 实时价格(预测), power
       - 索引: 北京时间时间戳 (15分钟频)
       - 文件名: output.csv
       
  6. 批量处理 (测试集D天):
       - 循环D次，每次生成96行
       - 垂直拼接为 96×D 行大表
```

#### 15.2 部署环境配置

```yaml
交付清单:
  strategy_core.so          # C++编译的动态库
  strategy_wrapper.py       # Python封装
  risk_mgmt.py              # 风险管理
  main.py                   # 端到端入口
  requirements.txt          # 依赖 (numpy, pandas, torch等)
  Dockerfile (可选)         # 容器化部署
  
环境要求:
  - Python 3.9+
  - GCC 9+ (用于编译C++扩展)
  - 内存: 8GB+ (处理NC文件时需要)
  - 支持WSL2或Linux环境
```

#### 15.3 关键验证检查点

```yaml
Day 15 集成测试清单:
  [ ] 单日测试: 输入96点预测，输出96点power，约束满足
  [ ] 批量测试: D=30天，输出CSV维度为 (2880, 3)
  [ ] 约束验证: 随机抽查10天，确认tc∈[0,80], td∈[tc+8,88]
  [ ] 收益计算: 用历史真实价格验证收益公式正确性
  [ ] 边界情况: 
      - 全零预测时的处理 (应输出全0或不操作)
      - 负电价时段的充电意愿
      - 极端尖峰(>1000)时的放电意愿
```

---

## 4. 技术实现规范（C++背景适配）

### 4.1 C++代码风格要求

```cpp
// strategy_core.cpp 规范
#include <vector>
#include <algorithm>
#include <cmath>

// 禁止裸指针，使用std::vector
using PriceSeries = std::vector<double>;

// 结构体明确命名
struct ChargeDischargeStrategy {
    int charge_start;
    int discharge_start;
    double profit;
    bool is_valid;
};

// 函数签名清晰，带const correctness
double calculate_strategy_profit(
    const PriceSeries& prices,
    int charge_start,
    int discharge_start
) noexcept;

// 主优化函数
extern "C" {
    // 导出符号供Python调用
    int find_optimal_strategy(
        const double* prices,
        int n,
        double risk_aversion,
        int* out_charge_start,
        int* out_discharge_start,
        double* out_profit
    );
}
```

### 4.2 Python封装层规范

```python
# strategy_wrapper.py
import ctypes
import numpy as np
from pathlib import Path

# 加载编译后的库
_LIB_PATH = Path(__file__).parent / "strategy_core.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

# 函数签名声明
_lib.find_optimal_strategy.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # prices
    ctypes.c_int,                     # n
    ctypes.c_double,                  # risk_aversion
    ctypes.POINTER(ctypes.c_int),     # out_charge_start
    ctypes.POINTER(ctypes.c_int),     # out_discharge_start
    ctypes.POINTER(ctypes.c_double),   # out_profit
]
_lib.find_optimal_strategy.restype = ctypes.c_int  # 返回状态码0=成功

def optimize_charge_discharge(
    prices: np.ndarray,
    confidence: np.ndarray | None = None,
    risk_aversion: float = 0.1,
    profit_threshold: float = 0.0
) -> dict:
    """
    Python接口：调用C++核心搜索
    
    Args:
        prices: [96] 预测电价
        confidence: [96] 预测置信度 (可选)
        risk_aversion: 风险厌恶系数 (0-1)
        profit_threshold: 最低执行收益阈值
        
    Returns:
        {
            'charge_start': int | None,
            'discharge_start': int | None, 
            'power_profile': np.ndarray [96],
            'expected_profit': float,
            'execute': bool
        }
    """
    # 输入验证
    assert prices.shape == (96,), f"prices must be [96], got {prices.shape}"
    assert np.issubdtype(prices.dtype, np.floating), "prices must be float"
    
    # 若未提供confidence，使用默认值
    if confidence is None:
        confidence = np.full(96, prices.std() * 0.1)
    
    # 准备输出变量
    c_start = ctypes.c_int(0)
    d_start = ctypes.c_int(0)
    profit = ctypes.c_double(0.0)
    
    # 调用C++库
    status = _lib.find_optimal_strategy(
        prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        96,
        risk_aversion,
        ctypes.byref(c_start),
        ctypes.byref(d_start),
        ctypes.byref(profit)
    )
    
    if status != 0:
        raise RuntimeError(f"C++ library returned error code: {status}")
    
    # 构建power序列
    power = np.zeros(96, dtype=np.int32)
    execute = profit.value > profit_threshold
    
    if execute:
        power[c_start.value : c_start.value+8] = -1000
        power[d_start.value : d_start.value+8] = +1000
    
    return {
        'charge_start': c_start.value if execute else None,
        'discharge_start': d_start.value if execute else None,
        'power_profile': power,
        'expected_profit': profit.value,
        'execute': execute
    }
```

---

## 5. 协作节点与同步机制

### 5.1 关键接口确认时间表

| 日期 | 协作对象 | 确认内容 | 验证方式 |
|:---|:---|:---|:---|
| Day 3 | A | 确认`aligned_15min_full.csv`列名稳定 | 检查`data_schema.py` |
| Day 5 | A | 物理特征注入成功，`deep_peak_risk`等列存在 | 跑通`align_pipeline.py` |
| Day 7 | A+B | 确认`feature_cols`清单≤50维 | 打印`DataModule`特征数 |
| Day 11 | B | 确认`prediction`+`confidence`输出格式 | 联调`inference.py` |
| Day 12 | B | C++搜索模块接收B的输出无误 | 单日端到端测试 |
| Day 14 | B | 多模型集成接口稳定 | 批量测试D=7天 |
| Day 15 | A+B | 最终集成测试通过，CSV格式正确 | 提交格式验证 |

### 5.2 代码审查重点（C++思维检查清单）

```yaml
审查成员A的数据Pipeline (你审查A):
  - 检查点: 向量化效率，避免Python for循环处理DataFrame
  - C++视角: 检查是否有O(N²)的DataFrame操作，建议用numpy向量化
  
审查成员B的模型代码 (你审查B):
  - 检查点: 梯度流、tensor维度匹配
  - C++视角: 检查是否有内存泄漏风险（Python引用循环），检查tensor连续性
  
自我审查 (A+B审查你):
  - 检查点: 边界条件、数组越界、内存安全
  - C++视角: 确保所有数组访问有边界检查，ctypes接口类型安全
```

---

## 6. 风险预警与应急预案

| 风险点 | 概率 | 应对方案 |
|:---|:---|:---|
| B的模型延迟交付 | 中 | Day 12先用历史均值或Baseline预测跑通策略管线，Day 14替换为真实预测 |
| C++编译环境问题 | 中 | 准备纯Python备用实现（O(N²)搜索在Python中6561次循环可接受） |
| 预测精度不足导致策略失效 | 中 | Day 13强化风险阈值，宁可不交易也不错误交易 |
| 负电价/尖峰预测缺失 | 低 | 在`risk_mgmt.py`中硬编码规则：价格<0强制充电，>800强制放电 |
| WSL文件IO瓶颈 | 低 | 建议A将NC处理放在WSL本地目录(`~/data`)而非Windows挂载 |

---

## 7. 快速参考：核心算法伪代码

### 7.1 基础搜索算法（C++风格）

```cpp
// 暴力搜索最优(tc, td)
StrategyResult find_optimal_strategy(const vector<double>& prices) {
    double best_profit = -INFINITY;
    int best_tc = -1, best_td = -1;
    
    for (int tc = 0; tc <= 80; ++tc) {        // 充电开始
        for (int td = tc + 8; td <= 88; ++td) {  // 放电开始
            // 计算该组合收益
            double charge_cost = 0.0;
            for (int t = tc; t < tc + 8; ++t) 
                charge_cost += prices[t] * 1000;  // 充电支出
            
            double discharge_revenue = 0.0;
            for (int t = td; t < td + 8; ++t)
                discharge_revenue += prices[t] * 1000;  // 放电收入
            
            double profit = discharge_revenue - charge_cost;
            
            if (profit > best_profit) {
                best_profit = profit;
                best_tc = tc;
                best_td = td;
            }
        }
    }
    
    return {best_tc, best_td, best_profit, best_profit > 0};
}
```

### 7.2 风险调整后搜索（含置信度）

```cpp
// 风险调整收益 = 期望收益 - 风险惩罚
double risk_adjusted_profit(
    const vector<double>& prices,
    const vector<double>& confidence,
    int tc, int td,
    double lambda
) {
    double expected_profit = calculate_raw_profit(prices, tc, td);
    
    // 风险惩罚 = λ × Σ(confidence × |功率|)
    double risk_penalty = 0.0;
    for (int t = tc; t < tc + 8; ++t) risk_penalty += confidence[t] * 1000;
    for (int t = td; t < td + 8; ++t) risk_penalty += confidence[t] * 1000;
    
    return expected_profit - lambda * risk_penalty;
}
```

---

## 8. 交付物清单总览

| 阶段 | 文件名 | 说明 |
|:---|:---|:---|
| Day 5 | `strategy_features.py` | 物理特征计算函数（提交给A集成） |
| Day 12 | `strategy_core.cpp` | C++核心搜索算法 |
| Day 12 | `strategy_core.so` | 编译后的动态库 |
| Day 12 | `strategy_wrapper.py` | Python ctypes封装 |
| Day 13 | `risk_mgmt.py` | 风险修正与阈值逻辑 |
| Day 13 | `monte_carlo_validator.py` | 鲁棒性检验工具 |
| Day 14 | `ensemble_strategy.py` | 多模型集成接口（可选） |
| Day 15 | `main.py` | 端到端主入口 |
| Day 15 | `requirements.txt` | Python依赖 |
| Day 15 | `README.md` | 使用说明与运行命令 |

---

此计划严格遵循`coding_standards.md`的接口规范，充分利用团队C++背景实现高性能策略搜索，同时保持Python层的灵活性。建议Day 12优先完成基础搜索，Day 13再叠加风险逻辑，确保每个阶段都有可运行的交付物。