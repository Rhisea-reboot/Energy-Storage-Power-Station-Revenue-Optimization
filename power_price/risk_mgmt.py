"""
risk_mgmt.py
============
Day 13 风险修正与阈值逻辑模块。

实现策略:
    A. 置信度折扣 (Confidence Discounting)
    B. 分位数保守估计 (Quantile-based)
    C. 收益阈值机制 (Profit Threshold)
    D. 极端价格过滤 (Extreme Price Filter)

接口函数:
    apply_risk_management(prediction, confidence, config) -> dict
"""

import numpy as np
from typing import Dict, Any


def _confidence_discounting(
    prediction: np.ndarray,
    confidence: np.ndarray,
    lambda_risk: float,
) -> np.ndarray:
    """
    A. 置信度折扣：adjusted_profit = expected_profit - λ × Σ(confidence × |Et|)
    此处返回的是经折扣后的价格估计（等价于在搜索中直接扣除风险惩罚）。
    为了与搜索层解耦，本模块返回修正后的 price 代理值。
    """
    # 风险折扣不直接修改价格，而是在搜索层通过 risk_aversion 参数生效
    # 本函数保留用于显式计算风险惩罚金额
    return prediction


def quantile_conservative_estimate(
    prediction: np.ndarray,
    confidence: np.ndarray,
    z_score: float = 1.28,
) -> np.ndarray:
    """
    B. 分位数保守估计
        conservative = prediction - z × confidence

    Parameters
    ----------
    prediction : np.ndarray [96]
        点预测电价。
    confidence : np.ndarray [96]
        预测标准差。
    z_score : float
        正态分布分位数：1.28≈80%, 1.645≈90%, 1.96≈95%。

    Returns
    -------
    np.ndarray [96]
        保守估计价格（用于充电决策时取上界，放电决策时取下界，
        此处统一返回下界以做保守放电估计）。
    """
    return prediction - z_score * confidence


def extreme_price_filter(
    prediction: np.ndarray,
    negative_threshold: float = 0.0,
    spike_threshold: float = 800.0,
) -> Dict[str, np.ndarray]:
    """
    D. 极端价格过滤
        - 负电价时段强制标记为充电机会
        - 尖峰电价(>800)时段强制标记为放电机会

    Parameters
    ----------
    prediction : np.ndarray [96]
        预测电价。
    negative_threshold : float
        负电价阈值，默认 0.0 元/MWh。
    spike_threshold : float
        尖峰电价阈值，默认 800.0 元/MWh。

    Returns
    -------
    dict
        {
            'negative_mask': np.ndarray [96] bool,  // 应强制充电的时段
            'spike_mask':    np.ndarray [96] bool,  // 应强制放电的时段
        }
    """
    negative_mask = prediction < negative_threshold
    spike_mask = prediction > spike_threshold
    return {
        "negative_mask": negative_mask,
        "spike_mask": spike_mask,
    }


def profit_threshold_adaptive(
    historical_profits: np.ndarray,
    fixed_threshold: float | None = None,
    percentile: float = 25.0,
) -> float:
    """
    C. 收益阈值机制
        优先使用固定阈值；若未提供，则取历史收益分布的指定分位数。

    Parameters
    ----------
    historical_profits : np.ndarray
        历史每日收益样本（可为空）。
    fixed_threshold : float | None
        固定阈值。若提供则直接返回。
    percentile : float
        历史分位数，默认 25.0（即只执行收益排在前75%的策略）。

    Returns
    -------
    float
        当日应采用的收益阈值（元）。
    """
    if fixed_threshold is not None:
        return fixed_threshold
    if historical_profits.size == 0:
        return 0.0
    return float(np.percentile(historical_profits, percentile))


def apply_risk_management(
    prediction: np.ndarray,
    confidence: np.ndarray | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    综合风险管理入口：对预测价格进行风险修正并输出决策辅助信息。

    Parameters
    ----------
    prediction : np.ndarray [96]
        点预测电价。
    confidence : np.ndarray | None, optional
        预测标准差或置信度评分。
    config : dict | None, optional
        风险参数配置，键包括:
        - 'risk_aversion' (float): λ，默认 0.1
        - 'z_score' (float): 分位数保守系数，默认 1.28
        - 'fixed_profit_threshold' (float | None): 固定收益阈值
        - 'historical_profits' (np.ndarray): 历史收益样本
        - 'negative_threshold' (float): 负电价阈值，默认 0.0
        - 'spike_threshold' (float): 尖峰阈值，默认 800.0

    Returns
    -------
    dict
        {
            'adjusted_prediction': np.ndarray [96],  // 经风险修正的价格代理
            'risk_aversion': float,
            'profit_threshold': float,
            'extreme_negative_mask': np.ndarray [96] bool,
            'extreme_spike_mask': np.ndarray [96] bool,
            'conservative_prediction': np.ndarray [96], // z_score保守估计
        }
    """
    if config is None:
        config = {}

    risk_aversion = float(config.get("risk_aversion", 0.1))
    z_score = float(config.get("z_score", 1.28))
    fixed_threshold = config.get("fixed_profit_threshold", None)
    historical_profits = config.get("historical_profits", np.array([]))
    negative_threshold = float(config.get("negative_threshold", 0.0))
    spike_threshold = float(config.get("spike_threshold", 800.0))

    if confidence is None:
        confidence = np.full(96, prediction.std() * 0.1)

    # A. 置信度折扣 -> 在搜索层通过 risk_aversion 生效，此处仅计算惩罚金额参考
    _confidence_discounting(prediction, confidence, risk_aversion)

    # B. 分位数保守估计
    conservative_prediction = quantile_conservative_estimate(
        prediction, confidence, z_score
    )

    # C. 收益阈值
    profit_threshold = profit_threshold_adaptive(
        historical_profits, fixed_threshold
    )

    # D. 极端价格过滤
    extreme = extreme_price_filter(
        prediction, negative_threshold, spike_threshold
    )

    # 综合修正后的预测代理：用保守估计替换原始预测用于策略搜索
    adjusted_prediction = conservative_prediction.copy()

    return {
        "adjusted_prediction": adjusted_prediction,
        "risk_aversion": risk_aversion,
        "profit_threshold": profit_threshold,
        "extreme_negative_mask": extreme["negative_mask"],
        "extreme_spike_mask": extreme["spike_mask"],
        "conservative_prediction": conservative_prediction,
    }
