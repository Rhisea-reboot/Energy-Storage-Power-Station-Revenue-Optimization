/**
 * strategy_core.cpp
 * =================
 * Day 12 储能充放电策略核心搜索算法 (C++ 实现)
 *
 * 约束:
 *   - 充电功率: -1000 MW, 放电功率: +1000 MW
 *   - 连续 8 个时间点 (2 小时)
 *   - 充电开始 tc ∈ [0, 80], 放电开始 td ∈ [tc+8, 88]
 *   - 每日最多 1 次完整 "充+放" 循环或不操作
 *
 * 导出接口:
 *   - find_optimal_strategy()
 *   - find_optimal_strategy_with_confidence()
 */

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

// ------------------------------------------------------------------
// 内部类型别名
// ------------------------------------------------------------------
using PriceSeries = std::vector<double>;

struct StrategyResult {
    int charge_start;       // 充电开始索引 [0,80]
    int discharge_start;    // 放电开始索引 [8,88]
    double expected_profit; // 预期收益 (元)
    double confidence_adjusted_return; // 风险调整后收益
    bool execute;           // 是否执行交易
};

// ------------------------------------------------------------------
// 内部辅助函数
// ------------------------------------------------------------------

/**
 * 计算指定 (tc, td) 组合的原始期望收益
 */
static double calculate_raw_profit(
    const PriceSeries& prices,
    int charge_start,
    int discharge_start
) noexcept {
    double charge_cost = 0.0;
    for (int t = charge_start; t < charge_start + 8; ++t) {
        charge_cost += prices[t] * 1000.0;
    }

    double discharge_revenue = 0.0;
    for (int t = discharge_start; t < discharge_start + 8; ++t) {
        discharge_revenue += prices[t] * 1000.0;
    }

    return discharge_revenue - charge_cost;
}

/**
 * 计算风险调整后收益
 *   adjusted_profit = expected_profit - lambda * Σ(confidence[t] * |Et|)
 */
static double risk_adjusted_profit(
    const PriceSeries& prices,
    const PriceSeries& confidence,
    int charge_start,
    int discharge_start,
    double lambda
) noexcept {
    double expected_profit = calculate_raw_profit(
        prices, charge_start, discharge_start
    );

    double risk_penalty = 0.0;
    for (int t = charge_start; t < charge_start + 8; ++t) {
        risk_penalty += confidence[t] * 1000.0;
    }
    for (int t = discharge_start; t < discharge_start + 8; ++t) {
        risk_penalty += confidence[t] * 1000.0;
    }

    return expected_profit - lambda * risk_penalty;
}

/**
 * 核心搜索函数：遍历所有合法 (tc, td) 组合，返回最优策略
 */
static StrategyResult optimize_strategy_internal(
    const PriceSeries& prices,
    const PriceSeries* confidence_ptr,  // 可为 nullptr
    double risk_aversion,
    double profit_threshold
) noexcept {
    const int N = static_cast<int>(prices.size());
    const int DURATION = 8;
    const int MAX_TC = N - DURATION * 2;        // 80
    const int MAX_TD = N - DURATION;            // 88

    double best_score = -std::numeric_limits<double>::infinity();
    int best_tc = -1;
    int best_td = -1;
    double best_raw_profit = 0.0;

    for (int tc = 0; tc <= MAX_TC; ++tc) {
        for (int td = tc + DURATION; td <= MAX_TD; ++td) {
            double score;
            double raw_profit = calculate_raw_profit(prices, tc, td);

            if (confidence_ptr != nullptr) {
                score = risk_adjusted_profit(
                    prices, *confidence_ptr, tc, td, risk_aversion
                );
            } else {
                score = raw_profit;
            }

            if (score > best_score) {
                best_score = score;
                best_tc = tc;
                best_td = td;
                best_raw_profit = raw_profit;
            }
        }
    }

    bool execute = (best_score > profit_threshold) && (best_tc >= 0);

    return {
        best_tc,
        best_td,
        best_raw_profit,
        best_score,
        execute
    };
}

// ------------------------------------------------------------------
// C 导出接口 (供 Python ctypes 调用)
// ------------------------------------------------------------------

extern "C" {

/**
 * 基础搜索接口（不带置信度）
 *
 * 参数:
 *   prices          : [n] 预测电价数组
 *   n               : 数组长度 (固定 96)
 *   risk_aversion   : 风险厌恶系数 (此处仅作占位，不生效)
 *   out_charge_start: 输出最优充电开始索引
 *   out_discharge_start: 输出最优放电开始索引
 *   out_profit      : 输出最优原始期望收益
 *
 * 返回:
 *   0 = 成功, 非0 = 错误
 */
int find_optimal_strategy(
    const double* prices,
    int n,
    double risk_aversion,
    int* out_charge_start,
    int* out_discharge_start,
    double* out_profit
) {
    if (prices == nullptr || out_charge_start == nullptr
        || out_discharge_start == nullptr || out_profit == nullptr) {
        return -1;
    }
    if (n <= 0) {
        return -2;
    }

    PriceSeries price_vec(prices, prices + n);

    StrategyResult result = optimize_strategy_internal(
        price_vec,
        nullptr,          // 无置信度
        risk_aversion,
        0.0               // 基础搜索不设阈值，由 Python 层判断
    );

    *out_charge_start = result.charge_start;
    *out_discharge_start = result.discharge_start;
    *out_profit = result.expected_profit;

    return 0;
}

/**
 * 扩展搜索接口（带置信度）
 *
 * 参数在基础接口上增加 confidence 数组
 */
int find_optimal_strategy_with_confidence(
    const double* prices,
    const double* confidence,
    int n,
    double risk_aversion,
    double profit_threshold,
    int* out_charge_start,
    int* out_discharge_start,
    double* out_profit
) {
    if (prices == nullptr || confidence == nullptr
        || out_charge_start == nullptr
        || out_discharge_start == nullptr || out_profit == nullptr) {
        return -1;
    }
    if (n <= 0) {
        return -2;
    }

    PriceSeries price_vec(prices, prices + n);
    PriceSeries conf_vec(confidence, confidence + n);

    StrategyResult result = optimize_strategy_internal(
        price_vec,
        &conf_vec,
        risk_aversion,
        profit_threshold
    );

    *out_charge_start = result.charge_start;
    *out_discharge_start = result.discharge_start;
    *out_profit = result.confidence_adjusted_return;

    return 0;
}

} // extern "C"
