"""
monte_carlo_validator.py
========================
Day 13 鲁棒性检验工具 — 多情景 Monte-Carlo 模拟。

对每个候选 (tc, td) 组合，生成 K 个价格情景：
    price_scenario = prediction + ε,  ε ~ N(0, confidence²)

评估指标:
    - 期望收益: mean(profit_k)
    - 收益波动: std(profit_k)
    - 下行风险: P(percentile(profit_k, 10) < 0)
    - 夏普比率: mean(profit_k) / std(profit_k)

决策规则:
    选择夏普比率最高且下行风险 < 20% 的策略。

接口函数:
    monte_carlo_validate(prediction, confidence, candidate_strategies, config) -> dict
    evaluate_strategy_robustness(prediction, confidence, tc, td, n_scenarios) -> dict
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def _simulate_profit_distribution(
    prediction: np.ndarray,
    confidence: np.ndarray,
    tc: int,
    td: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    对指定 (tc, td) 生成 K 个价格情景并计算收益分布。

    Returns
    -------
    np.ndarray [K]
        K 次模拟的收益样本（元）。
    """
    K = n_scenarios
    n = prediction.shape[0]

    # 生成情景矩阵 [K, n]
    noise = rng.normal(loc=0.0, scale=confidence, size=(K, n))
    scenarios = prediction + noise

    # 计算每次情景的收益
    charge_cost = scenarios[:, tc:tc + 8].sum(axis=1) * 1000.0
    discharge_revenue = scenarios[:, td:td + 8].sum(axis=1) * 1000.0
    profits = discharge_revenue - charge_cost

    return profits


def evaluate_strategy_robustness(
    prediction: np.ndarray,
    confidence: np.ndarray,
    tc: int,
    td: int,
    n_scenarios: int = 100,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    """
    对单一 (tc, td) 策略进行 Monte-Carlo 鲁棒性评估。

    Parameters
    ----------
    prediction : np.ndarray [96]
        点预测电价。
    confidence : np.ndarray [96]
        预测标准差。
    tc : int
        充电开始索引。
    td : int
        放电开始索引。
    n_scenarios : int
        模拟情景数 K，默认 100。
    random_seed : int | None
        随机种子，用于结果可复现。

    Returns
    -------
    dict
        {
            'expected_profit': float,      // 平均收益
            'profit_std': float,           // 收益标准差
            'profit_p10': float,           // 10% 分位数收益
            'downside_risk': float,        // P(profit < 0)
            'sharpe_ratio': float,         // mean / std
            'is_robust': bool,             // downside_risk < 0.2
        }
    """
    rng = np.random.default_rng(random_seed)
    profits = _simulate_profit_distribution(
        prediction, confidence, tc, td, n_scenarios, rng
    )

    expected_profit = float(profits.mean())
    profit_std = float(profits.std(ddof=1))
    profit_p10 = float(np.percentile(profits, 10))
    downside_risk = float((profits < 0).sum() / n_scenarios)
    sharpe_ratio = (
        expected_profit / profit_std if profit_std > 1e-9 else np.inf
    )

    return {
        "expected_profit": expected_profit,
        "profit_std": profit_std,
        "profit_p10": profit_p10,
        "downside_risk": downside_risk,
        "sharpe_ratio": sharpe_ratio,
        "is_robust": downside_risk < 0.2,
    }


def monte_carlo_validate(
    prediction: np.ndarray,
    confidence: np.ndarray,
    candidate_strategies: List[Tuple[int, int]] | None = None,
    n_scenarios: int = 100,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    """
    对候选策略集合进行 Monte-Carlo 验证，返回最优鲁棒策略。

    Parameters
    ----------
    prediction : np.ndarray [96]
        点预测电价。
    confidence : np.ndarray [96]
        预测标准差。
    candidate_strategies : List[Tuple[int, int]] | None
        候选 (tc, td) 列表。若 None，则遍历所有合法组合。
    n_scenarios : int
        模拟情景数，默认 100。
    random_seed : int | None
        随机种子。

    Returns
    -------
    dict
        {
            'best_tc': int,
            'best_td': int,
            'best_sharpe': float,
            'results': List[dict],  // 每个候选的详细评估
        }
    """
    rng = np.random.default_rng(random_seed)

    if candidate_strategies is None:
        candidate_strategies = []
        for tc in range(0, 81):
            for td in range(tc + 8, 89):
                candidate_strategies.append((tc, td))

    results = []
    best_sharpe = -np.inf
    best_tc, best_td = -1, -1

    for tc, td in candidate_strategies:
        profits = _simulate_profit_distribution(
            prediction, confidence, tc, td, n_scenarios, rng
        )

        expected_profit = float(profits.mean())
        profit_std = float(profits.std(ddof=1))
        downside_risk = float((profits < 0).sum() / n_scenarios)
        sharpe_ratio = (
            expected_profit / profit_std if profit_std > 1e-9 else np.inf
        )

        entry = {
            "tc": tc,
            "td": td,
            "expected_profit": expected_profit,
            "profit_std": profit_std,
            "downside_risk": downside_risk,
            "sharpe_ratio": sharpe_ratio,
            "is_robust": downside_risk < 0.2,
        }
        results.append(entry)

        # 决策规则：夏普比率最高且下行风险 < 20%
        if downside_risk < 0.2 and sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_tc = tc
            best_td = td

    return {
        "best_tc": best_tc,
        "best_td": best_td,
        "best_sharpe": float(best_sharpe),
        "results": results,
    }
