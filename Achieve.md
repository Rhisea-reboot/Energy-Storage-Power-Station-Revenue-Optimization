# AI4S 电力现货电价预测 — 前三天数据处理方法论分析

本文档系统梳理 Day 1 ~ Day 3 在 `Test_Weather_processor.py`、`Dataaligning.py`、`Dataset.py` 中实现的数据处理管线。重点阐述**数据处理方法**、**工具选择**以及**各阶段数据的使用方式**，而非代码实现细节。

---

## Day 1：气象数据降维与特征提取

### 1.1 核心目标

将高维的数值天气预报（NWP）NetCDF（NC）格点数据，转化为可用于电价预测的低维时间序列特征表。NC 文件通常包含 `[时间, 预报时效, 变量通道, 纬度, 经度]` 等多维结构，直接使用会导致模型输入维度爆炸且包含大量空间冗余信息。

### 1.2 工具选择

- **xarray**：NC 文件的标准读取与多维切片工具，支持按维度名称（`time`、`lead_time`、`channel`、`lat`、`lon`）进行语义化索引，避免硬编码轴序号带来的错误。
- **pandas**：将提取出的标量时间序列组织为带时间索引的 DataFrame，便于后续拼接、去重、重采样。
- **numpy**：在空间维度上进行聚合统计（mean、max、min、std）。

### 1.3 数据处理方法

#### 1.3.1 时间戳还原

NC 气象预报文件中的时间通常由两个维度共同决定：
- **基准时间（base_time）**：模型起报时间。
- **预报时效（lead_time）**：相对于起报时间的未来偏移量（小时）。

处理方法为将二者相加，得到**实际预报所指向的物理时间戳**，确保后续与电价、边界条件数据对齐时不会发生时间错位。

#### 1.3.2 空间降维：2D 网格 → 1D 标量

对每个气象变量在每个时间点上的二维空间场（经纬度网格）进行统计聚合，将空间信息压缩为少量代表性标量。设某气象变量在经度 $x$、纬度 $y$ 方向上的网格值为 $V_{x,y}$，网格点总数为 $N$：

- **空间均值（mean）**：表征全省/全区域的整体气象状态，是模型最关心的宏观输入。
  $$
  \text{mean} = \frac{1}{N} \sum_{x,y} V_{x,y}
  $$

- **空间最大值（max）与最小值（min）**：捕捉区域内的极端气象条件，对极端电价（尖峰、深谷）有解释力。
  $$
  \text{max} = \max_{x,y}(V_{x,y}), \quad \text{min} = \min_{x,y}(V_{x,y})
  $$

- **空间标准差（std）**：反映气象变量在空间分布上的不均匀性。例如，云量 std 大说明省内部分地区晴朗、部分地区阴雨，可能导致光伏出力分化，进而影响电价空间分布。
  $$
  \text{std} = \sqrt{\frac{1}{N} \sum_{x,y} \left(V_{x,y} - \text{mean}\right)^2}
  $$

#### 1.3.3 派生特征构建

在原始气象变量统计量的基础上，结合电力气象学知识构建具有物理意义的派生特征：

- **风速合成（wspd）**：将 100m 高度层的 u 分量（东西向风速）和 v 分量（南北向风速）合成为风速标量，再对该风速场进行空间聚合。这是因为风功率与风速直接相关，而非与单一分量相关。
  $$
  \text{wspd}_{x,y} = \sqrt{u100_{x,y}^2 + v100_{x,y}^2}
  $$
  随后对 $\text{wspd}_{x,y}$ 全场计算 mean、max（代码中实现了 `wspd_mean`、`wspd_max`）。

- **温度单位转换**：将开尔文（K）转换为摄氏（℃），更贴近业务直觉。
  $$
  T_{℃} = T_{K} - 273.15
  $$
  该转换作用于 `t2m_mean`、`t2m_max`、`t2m_min`。

- **温度跨度（temp_spread）**：既反映空间温差，也与日内温差相关，可用于解释空调负荷的空间差异。
  $$
  \text{temp\_spread} = \text{t2m\_max} - \text{t2m\_min}
  $$

- **太阳辐射潜力（solar_potential）**：总水平辐射（GHI）是光伏出力的能量来源，总云量（TCC）是主要衰减因子。二者相乘可近似反映“有效太阳辐射”，与光伏实际出力高度相关。
  $$
  \text{solar\_potential} = \text{ghi\_mean} \times \left(1 - \text{tcc\_mean}\right)
  $$

- **风功率潜力（wind_power_potential）**：根据贝茨极限前的风功率公式，风功率与风速的三次方成正比，因此该特征能更好地映射到风电出力对电价的抑制作用。
  $$
  \text{wind\_power\_potential} = \left(\text{wspd\_mean}\right)^3
  $$

#### 1.3.4 时间序列规整

多个 NC 文件拼接后，进行以下处理：
- **去重**：同一实际时间戳可能出现在不同起报文件的重叠预报时效中，保留第一条记录即可。
- **排序**：按时间索引升序排列，保证序列单调性。
- **重采样与插值**：将小时级气象序列通过**线性插值**升频到更高频率（如 15 分钟）。气象变量具有物理连续性，线性插值是合理且保守的做法。

### 1.4 各数据使用方式

| 阶段 | 数据形态 | 说明 |
|------|----------|------|
| **输入** | 原始 NC 格点文件 | 包含多变量、多时效、多空间格点的数值天气预报 |
| **中间处理** | 每个文件 → 时间序列 DataFrame | 按 `base_time + lead_time` 映射到物理时间，每行对应一个时间点，列是各变量的 mean/max/min/std |
| **输出** | `weather_features.csv` | 统一时间轴、统一频率的气象特征宽表，作为 Day 2 的气象输入源 |

---

## Day 2：多源数据对齐（15 分钟粒度）

### 2.1 核心目标

将异源、异频、异时区的三类数据（电价、电力系统边界条件、气象特征）对齐到统一的 15 分钟时间轴上，并在此过程中构造具有强物理解释力的电力系统衍生特征，最终输出一条可用于深度学习建模的干净宽表。

### 2.2 工具选择

- **pandas**：时间序列对齐的核心工具，负责 CSV 读取、时区转换、重采样（resample/reindex）、插值（interpolate）、前向填充（ffill）、以及基于最近时间戳的合并（`merge_asof`）。
- **numpy**：处理除零产生的 `inf`、计算比例与误差等数值运算。

### 2.3 数据处理方法

#### 2.3.1 时区统一

三类数据来源不同，时间戳的时区信息可能不一致（电价/边界条件常为北京时间，气象数据可能为 UTC）。统一时区是避免对齐偏移的根本步骤：
- 对已带时区信息的数据，直接转换为 `Asia/Shanghai`。
- 对无时区信息的数据，**保守假设为 UTC**，再转换到北京时区。气象数据常常缺少 TZ 信息，这种假设可以防止因本地时间与 UTC 混淆而导致的 8 小时系统性偏移。

#### 2.3.2 分辨率对齐：以电价为基准的差异化重采样策略

目标频率为 **15 分钟**，每日 96 个点。由于三类数据的原始频率和物理性质不同，必须采用差异化的重采样策略，不能统一线性插值：

- **电价数据（Price）**：
  - **严禁线性插值**。电价是市场出清结果，在竞价边界变化时可能出现跳变（如峰谷切换时刻）。线性插值会在跳变点人为制造不存在的过渡值，扭曲真实的市场信号。
  - **处理方法**：前向填充（ffill）+ 后向填充（bfill）。即在 15 分钟粒度上，重复该时段已知的最新电价。

- **边界条件数据（Boundary）**：
  - 分为**实际值（actual）**和**预测值（forecast）**。
  - **实际值严禁插值**：系统负荷、风光实际出力等来自 SCADA/计量系统，是物理量测值，插值会制造虚假数据。
  - **预测值优先 ffill**：日前预测通常在每日固定时刻发布，全天保持不变或仅有少量更新，因此前向填充更符合业务实际；若存在小时级更新，少量线性插值亦可接受，但本实现统一采用 ffill 以保证保守性。

- **气象数据（Weather）**：
  - Day 1 输出为小时级，需要降频到 15 分钟。
  - **允许线性插值**。温度、辐射、风速等气象变量在物理上是连续变化的，小时级到 15 分钟的线性插值误差极小。
  - **缺失值均值填充**：若插值后仍存在缺失（如边界 NaN），用该列全局均值填充，避免模型因 NaN 而崩溃。

#### 2.3.3 多源数据合并：以时间戳为键的左连接

合并顺序和基准的选择至关重要：
- **以电价时间戳为左表（基准）**，因为电价是预测目标，必须保证其时间戳完整无缺。
- 先合并边界条件，再合并气象数据。
- 使用 `pd.merge_asof(..., direction='nearest')` 进行**最近时间戳匹配**，并设置容忍度（tolerance）：
  - 边界条件容忍 1 分钟：电力系统数据时间精度高，必须严格对齐。
  - 气象数据容忍 15 分钟：气象数据粒度较粗，允许稍大的时间偏差。

这种合并方式能优雅处理三路数据时间戳不完全重合的问题，避免因为秒级差异而产生大量 NaN。

#### 2.3.4 电力系统衍生特征计算

在合并后的宽表基础上，依据电力市场物理规律构造核心解释变量，这些变量往往是比原始负荷、风光更直接的电价驱动因子：

- **竞价空间（Bidding Space）**：
  市场清算中真正需要火电机组和其他可调资源去满足的电量空间，是电价的第一预测因子。竞价空间越小，供给越紧张，电价越高。
  $$
  \text{bidding\_space\_actual} = \text{load\_actual} - \text{renewable\_actual} - \text{hydro\_actual} - \text{non\_market\_actual}
  $$
  $$
  \text{bidding\_space\_forecast} = \text{load\_forecast} - \text{renewable\_forecast} - \text{hydro\_forecast} - \text{non\_market\_forecast}
  $$

- **净负荷（Net Load）**：
  反映扣除新能源后的残余需求，是调度部门最关心的指标之一。
  $$
  \text{net\_load\_actual} = \text{load\_actual} - \text{renewable\_actual}
  $$
  $$
  \text{net\_load\_forecast} = \text{load\_forecast} - \text{renewable\_forecast}
  $$

- **新能源渗透率（renewable_penetration）**：
  渗透率越高，说明系统对新能源的依赖度越大，电价越容易受新能源出力波动影响。对除零产生的 `inf` 替换为 0。
  $$
  \text{renewable\_penetration} = \frac{\text{renewable\_actual}}{\text{load\_actual}} \quad \xrightarrow{\text{除零处理}} \quad \text{replace}(\pm\infty,\; 0)
  $$

- **供需缺口（supply_demand_gap）**：
  反映日前预测层面的供需平衡状态。正缺口意味着预测供不应求，可能推升日前电价；负缺口则可能压低电价。
  $$
  \text{supply\_demand\_gap} = \text{load\_forecast} - \left(\text{renewable\_forecast} + \text{hydro\_forecast} + \text{tie\_line\_forecast} + \text{non\_market\_forecast}\right)
  $$

- **联络线净受电比例（tie_line_ratio）**：
  反映外省电力输入对本地市场的支撑程度。
  $$
  \text{tie\_line\_ratio} = \frac{\text{tie\_line\_actual}}{\text{load\_actual}} \quad \xrightarrow{\text{除零处理}} \quad \text{replace}(\pm\infty,\; 0)
  $$

- **预测误差系列（Forecast Error）**：
  日前预测偏差会影响实时市场行为，从而传导至电价。
  $$
  \begin{aligned}
  \text{load\_forecast\_error} &= \text{load\_actual} - \text{load\_forecast} \\
  \text{wind\_forecast\_error} &= \text{wind\_actual} - \text{wind\_forecast} \\
  \text{solar\_forecast\_error} &= \text{solar\_actual} - \text{solar\_forecast} \\
  \text{renewable\_forecast\_error} &= \text{renewable\_actual} - \text{renewable\_forecast}
  \end{aligned}
  $$

#### 2.3.5 精细化特征（蒙西市场高 ROI）

在基础物理特征之上，针对**蒙西市场新能源占比极高、负电价频发**以及**储能套利策略优化**的特殊需求，进一步构造非线性、分时段、策略导向的精细化特征。这些特征对捕捉极端电价（尖峰、深谷）和价差幅度至关重要。

##### 1. 非线性边际效应

新能源渗透率对电价的压制具有**非线性**：从 50% 提升到 60% 的边际冲击，远大于从 10% 到 20%。

- **渗透率平方（penetration_sq）**：
  $$
  \text{penetration\_sq} = \left(\text{renewable\_penetration}\right)^2
  $$
  放大高渗透率时段的边际恶化信号，帮助模型区分"高渗透"与"极高渗透"。

- **极端新能源指示器（is_extreme_renewable）**：
  $$
  \text{is\_extreme\_renewable} = \mathbf{1}_{\{\text{renewable\_penetration} > 0.6\}}
  $$
  直接标记负电价高发场景（蒙西常见）。

##### 2. 分时段供需结构特征

负荷与新能源的**时间匹配度**比绝对值更重要。通过时段条件掩码，构造只在关键时段生效的特征：

- **中午光伏冲击（midday_solar_risk）**：10:00–14:00 光伏大发但负荷未完全起来，形成鸭子曲线的谷底。
  $$
  \text{midday\_solar\_risk} = \frac{\text{solar\_actual}}{\text{load\_actual}} \cdot \mathbf{1}_{\{10 \le \text{hour} \le 14\}}
  $$

- **晚峰调节压力（evening_ramp_stress）**：18:00–22:00 光伏归零但负荷高峰，净负荷占比越高，电价尖峰风险越大。
  $$
  \text{evening\_ramp\_stress} = \frac{\text{net\_load\_actual}}{\text{load\_actual}} \cdot \mathbf{1}_{\{18 \le \text{hour} \le 22\}}
  $$

##### 3. 储能策略相关特征

储能收益取决于**价差幅度**而不仅是绝对价格，因此需要能解释高低电价分离度的特征。

- **预期价格波动（expected_price_volatility）**：
  用竞价空间的短期滚动标准差作为未来电价波动的代理变量。
  $$
  \text{expected\_price\_volatility}_t = \text{Std}\left(\text{bidding\_space\_forecast}_{t-7:t}\right)
  $$
  滚动窗口为 8 个 15 分钟点（2 小时）。波动越大，套利空间越大。

- **深度调峰风险（deep_peak_risk）**：
  蒙西特色场景：冬季供暖期 + 风电大发 + 低负荷，极易触发负电价。
  $$
  \text{deep\_peak\_risk} = \mathbf{1}_{\{\text{wind} > q_{0.8}(\text{wind}) \;\land\; \text{load} < q_{0.3}(\text{load}) \;\land\; T_{\text{t2m}} < 5\,^\circ\text{C}\}}
  $$

##### 4. 联络线精细特征

蒙西作为送端电网，联络线外送功率直接影响本地供需松紧。

- **联络线裕度（tie_line_margin）**：
  $$
  \text{tie\_line\_margin} = C - |\text{tie\_line\_actual}|
  $$
  其中 $C$ 为联络线容量（若未提供，按历史实际功率绝对值 95% 分位数 × 1.1 自动估计）。裕度越小，本地新能源消纳压力越大。

- **高外送指示（is_high_export）**：
  $$
  \text{is\_high\_export} = \mathbf{1}_{\{|\text{tie\_line\_actual}| > 0.9 \, C\}}
  $$

##### 5. 蒙西市场特异性特征

- **风光互补指数（wind_solar_correlation_7d）**：
  滚动 7 日（$7 \times 96$ 点）风电与光伏实际出力的皮尔逊相关系数。
  $$
  \text{wind\_solar\_correlation\_7d} = \text{Corr}_{672}\left(\text{wind\_actual}, \text{solar\_actual}\right)
  $$
  相关系数越低，风光互补性越好，系统净负荷波动越小，电价越稳定。序列头部窗口不足时用 0 填充。

- **即时风光比（wind_solar_instant_ratio）**：
  $$
  \text{wind\_solar\_instant\_ratio} = \frac{\text{wind\_actual}}{\text{solar\_actual} + 10^{-6}}
  $$
  作为互补指数的局部代理，解决冷启动问题。

- **弃风弃光隐性成本标记（curtailment_flag）**：
  当实际新能源大幅低于预测且电价极低时，反映系统阻塞或消纳困难。
  $$
  \text{curtailment\_flag} = \mathbf{1}_{\{\text{renewable\_actual} < 0.8 \, \text{renewable\_forecast} \;\land\; \text{price} < 50\}}
  $$

- **供热期刚性约束（heating_season_rigid）**：
  蒙西冬季（11 月–次年 3 月）凌晨 0:00–6:00，供热机组最小出力高，压低凌晨电价。
  $$
  \text{heating\_season\_rigid} = \mathbf{1}_{\{\text{month} \in [11,12,1,2,3] \;\land\; 0 \le \text{hour} \le 6\}}
  $$

#### 2.3.6 周期性时间特征编码

电力系统具有强烈的周期性（日周期、周周期、年周期）。为了让模型正确理解时间的循环特性，避免将 23:45 误认为离 00:00 很远，采用**正弦-余弦编码**。设 $t$ 为当前时间戳：

- **日内位置**：`time_of_day` 取值范围为 $0 \sim 95$（15 分钟粒度，1 天 96 点）
  $$
  \text{time\_of\_day} = \text{hour}(t) \times 4 + \frac{\text{minute}(t)}{15}
  $$
  $$
  \text{time\_sin} = \sin\left(\frac{2\pi \cdot \text{time\_of\_day}}{96}\right), \quad \text{time\_cos} = \cos\left(\frac{2\pi \cdot \text{time\_of\_day}}{96}\right)
  $$

- **周内位置**：`day_of_week` 取值范围为 $0 \sim 6$（周一到周日）
  $$
  \text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)
  $$

- **年内位置**：`month` 取值范围为 $1 \sim 12$
  $$
  \text{month\_sin} = \sin\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right)
  $$

同时加入两个布尔标记：
- **是否周末**：
  $$
  \text{is\_weekend} = \mathbf{1}_{\{\text{day\_of\_week} \,\in\, \{5,6\}\}}
  $$
- **是否高峰时段**：
  $$
  \text{is\_peak\_hour} = \mathbf{1}_{\{8 \,\le\, \text{hour}(t) \,\le\, 22\}}
  $$

#### 2.3.7 数据质量验证

对齐完成后，必须进行完整性校验：
- **每日点数检查**：每天必须恰好 96 个点。若不足，说明原始数据存在缺失或重采样异常。
- **时间间隔检查**：相邻时间戳的间隔必须严格为 15 分钟。任何非 15 分钟的间隔都意味着存在时间戳跳跃或重复。
- **关键特征缺失检查**：电价列、`load_actual`、`load_forecast`、`bidding_space_forecast` 等核心列不允许存在缺失值。

### 2.4 各数据使用方式

| 阶段 | 数据形态 | 说明 |
|------|----------|------|
| **输入 1** | 历史电价 CSV | 15 分钟级，目标变量，作为所有对齐操作的时间基准 |
| **输入 2** | 边界条件 CSV | 包含系统负荷、风光、联络线、水电、非市场化机组的实际值与预测值 |
| **输入 3** | `weather_features.csv`（Day 1 输出） | 小时级气象特征 |
| **中间处理** | 三路数据分别重采样到 15 分钟 | 差异化插值/填充策略 |
| **中间处理** | 宽表合并 + 衍生特征 + 时间编码 | 以电价为左表，按最近时间戳 merge |
| **输出** | `aligned_15min_full.csv` | 统一 15 分钟轴、包含全部特征的高质量宽表，供 Day 3 使用 |

---

## Day 3：PyTorch Dataset 实现

### 3.1 核心目标

将 Day 2 输出的长序列宽表，转化为深度学习模型可直接消费的 `(X, y)` 样本对，并完成训练/验证分割、标准化和 DataLoader 封装。

### 3.2 工具选择

- **pandas / numpy**：读取 CSV、提取特征矩阵和目标向量、时间索引校验。
- **sklearn.preprocessing.StandardScaler**：对特征进行 Z-score 标准化，消除不同特征量纲差异。
- **torch.utils.data.Dataset / DataLoader**：PyTorch 标准数据接口，支持批量加载、多进程、内存优化。

### 3.3 数据处理方法

#### 3.3.1 滑动窗口样本构建

将长序列切割为有监督学习样本。设对齐后的完整时间序列为 $\{t_i\}_{i=0}^{N-1}$，特征矩阵为 $\mathbf{X} \in \mathbb{R}^{N \times F}$，目标电价为 $\mathbf{y} \in \mathbb{R}^{N}$：

- **历史窗口（lookback_window）**：默认 $L = 672$ 个点 = 7 天 × 96 点/天。即模型可以“看到”过去 7 天的全部信息。
- **预测窗口（forecast_horizon）**：默认 $H = 96$ 个点 = 1 天 × 96 点/天。即模型需要预测未来 1 天每 15 分钟的电价。
- **滑动步长（stride）**：设为 $s = 1$（每 15 分钟滑动一次），最大化样本量。若数据量过大，可适当增大 $s$ 以平衡训练效率。

**有效样本索引**：对于起始索引 $i$，要求历史窗口与预测窗口均完整且时间连续：
$$
i \in \{0, s, 2s, \dots, N - L - H\}
$$
且满足时间连续性校验（防止跨越缺失期）：
$$
t_{i+L-1} - t_i = 15\,\text{min} \times (L - 1), \quad t_{i+L+H-1} - t_{i+L-1} = 15\,\text{min} \times H
$$

**单个样本的数学形式**：
- 输入：
  $$
  \mathbf{x}^{(i)} = \mathbf{X}[i : i+L] \in \mathbb{R}^{L \times F}
  $$
- 输出：
  $$
  \mathbf{y}^{(i)} = \mathbf{y}[i+L : i+L+H] \in \mathbb{R}^{H}
  $$

默认维度：
- $\mathbf{x}^{(i)}$：$(672, F)$
- $\mathbf{y}^{(i)}$：$(96,)$

#### 3.3.2 标准化：严防数据泄露

电力现货电价预测中，不同特征的量纲差异巨大（如温度几十度、负荷几千万千瓦、渗透率 0~1）。如果不进行标准化，模型会被大数值特征主导，学习不稳定。

采用 **StandardScaler（Z-score 标准化）**：
\[ x' = \frac{x - \mu}{\sigma} \]

**最关键的原则：防止数据泄露（Data Leakage）**：
- **训练集**：在训练特征上 `fit` 并 `transform`，计算得到均值 `μ` 和标准差 `σ`。
- **验证集 / 测试集**：只能使用训练集已经 `fit` 好的 scaler 进行 `transform`，**严禁在验证集上重新 fit**。如果验证集参与 fit，模型会提前“看到”验证集的分布信息，导致评估指标虚高，无法反映真实泛化能力。

实现中通过将训练集的 scaler 对象显式传入验证集 Dataset 来确保这一点。

#### 3.3.3 时间序列分割（非随机）

电力预测是典型的时间序列预测任务，未来数据在训练时不可见。因此：
- **必须按时间先后分割**，如前 10 个月作为训练集，后 2 个月作为验证集。
- **严禁随机打乱（shuffle=False）**。时间序列样本之间存在强自相关性，随机打乱会让模型在训练时“看到未来”，造成灾难性的数据泄露。

分割后还需要检查训练集截止日期与验证集开始日期之间的时间间隔，确保没有产生意料之外的数据断档。

#### 3.3.4 DataLoader 配置

- `shuffle=False`：再次强调，时间序列数据不打乱批次顺序。
- `pin_memory=True`：将数据预加载到 page-locked 内存，加速向 GPU 的传输。
- `num_workers`：调试时设为 0 以便定位问题；生产环境可增大以加速数据读取。

### 3.4 各数据使用方式

| 阶段 | 数据形态 | 说明 |
|------|----------|------|
| **输入** | `aligned_15min_full.csv`（Day 2 输出） | 统一 15 分钟宽表 |
| **分割** | train_df / val_df | 按时间先后切割，如 `index <= '2025-10-31'` 为训练集，之后为验证集 |
| **特征矩阵** | `(N, F)` | N 为总时间步，F 为特征维度 |
| **目标向量** | `(N,)` | 电价序列 |
| **标准化** | 训练集 fit scaler；验证集 transform | 防止数据泄露 |
| **样本对** | `X: (batch, 672, F)`，`y: (batch, 96)` | 每个 batch 包含过去 7 天预测未来 1 天的样本 |
| **输出** | `DataLoader` 对象 + `scaler.pkl` | 供 PyTorch 模型训练使用；scaler 持久化保存，供推理时反标准化 |

---

## 总结：三天管线的数据流与关键设计

```
Day 1: 气象 NC 文件
       ↓ (xarray 读取 → 空间聚合 mean/max/min/std → 派生特征 → 线性插值)
       weather_features.csv (小时级气象特征表)

       ↓

Day 2: 历史电价 CSV + 边界条件 CSV + weather_features.csv
       ↓ (时区统一 → 差异化 15min 重采样 → merge_asof 对齐 → 电力衍生特征 → 时间编码)
       aligned_15min_full.csv (统一 15min 宽表)

       ↓

Day 3: aligned_15min_full.csv
       ↓ (时间序列分割 → 滑动窗口 → StandardScaler 标准化 → PyTorch Dataset/DataLoader)
       模型训练可用数据 (batch, 672, F) → (batch, 96)
```

### 关键设计原则回顾

1. **气象空间降维**：用统计聚合（mean/max/min/std）将 2D 格点压缩为 1D 标量，既保留区域整体信息，又捕捉空间异质性。
2. **差异化插值策略**：电价与边界实际值严禁插值（ffill），气象数据允许线性插值——这是由不同数据类型的物理/市场属性决定的。
3. **以电价为基准的合并**：保证目标变量时间轴的完整性，其他数据源通过最近时间戳匹配补充。
4. **物理驱动特征优先**：竞价空间、净负荷、新能源渗透率等衍生特征，是比原始数据更直接的电价解释变量。
5. **精细化与市场特异性**：通过非线性变换（如 penetration_sq）、分时段掩码（midday_solar_risk / evening_ramp_stress）、策略代理变量（expected_price_volatility）以及蒙西特有场景标记（deep_peak_risk / heating_season_rigid），显著提升模型对极端电价和储能套利空间的捕捉能力。
6. **时间序列防泄露**：训练/验证按时间分割，标准化参数仅从训练集提取，DataLoader 不打乱顺序——这三点是保证模型评估有效性的底线。
