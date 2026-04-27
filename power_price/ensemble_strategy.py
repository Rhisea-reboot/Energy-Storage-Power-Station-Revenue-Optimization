"""
ensemble_strategy.py
====================
Day 14 多模型集成接口 — 支持多模型预测输入并融合为单一策略输入。

集成策略:
    1. 简单平均 (Simple Average)
    2. 方差加权 (Variance Weighting): w_i ∝ 1 / variance_i
    3. 分位数融合 (Quantile Fusion): 取中位数或指定分位数

接口函数:
    ensemble_predictions(predictions, confidences, model_weights, method) -> tuple
"""

import numpy as np
from typing import List, Tuple


def simple_average(
    predictions: List[np.ndarray],
    model_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. 简单平均集成

    Parameters
    ----------
    predictions : List[np.ndarray]
        M 个模型的预测序列，每个形状 (96,)。
    model_weights : np.ndarray | None
        M 个权重，sum=1。若为 None 则等权平均。

    Returns
    -------
    tuple (ensemble_pred, ensemble_conf)
        ensemble_pred : np.ndarray [96]
        ensemble_conf : np.ndarray [96]  (加权方差和)
    """
    M = len(predictions)
    if M == 0:
        raise ValueError("predictions 列表不能为空")

    arr = np.stack(predictions, axis=0)  # [M, 96]

    if model_weights is None:
        model_weights = np.ones(M) / M
    else:
        model_weights = np.asarray(model_weights, dtype=np.float64)
        if not np.isclose(model_weights.sum(), 1.0):
            model_weights = model_weights / model_weights.sum()

    # 加权预测
    ensemble_pred = np.average(arr, axis=0, weights=model_weights)

    # 集成置信度 = 加权方差和（反映模型间分歧）
    mean_sq = np.average(arr ** 2, axis=0, weights=model_weights)
    ensemble_conf = np.sqrt(np.maximum(mean_sq - ensemble_pred ** 2, 0.0))

    return ensemble_pred, ensemble_conf


def variance_weighting(
    predictions: List[np.ndarray],
    confidences: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2. 方差加权集成：低方差模型获得更高权重
        w_i ∝ 1 / mean(confidence_i)

    Parameters
    ----------
    predictions : List[np.ndarray]
        M 个模型预测。
    confidences : List[np.ndarray]
        M 个模型置信度。

    Returns
    -------
    tuple (ensemble_pred, ensemble_conf)
    """
    M = len(predictions)
    if M != len(confidences):
        raise ValueError("predictions 与 confidences 长度不一致")

    variances = np.array([c.mean() for c in confidences])
    # 防止除零
    inv_var = 1.0 / np.maximum(variances, 1e-9)
    weights = inv_var / inv_var.sum()

    return simple_average(predictions, weights)


def quantile_fusion(
    predictions: List[np.ndarray],
    quantile: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3. 分位数融合：取各模型预测的指定分位数（如中位数更稳健）

    Parameters
    ----------
    predictions : List[np.ndarray]
        M 个模型预测。
    quantile : float
        分位数，默认 0.5（中位数）。

    Returns
    -------
    tuple (ensemble_pred, ensemble_conf)
        ensemble_conf 取 IQR / 1.349 作为稳健标准差估计。
    """
    arr = np.stack(predictions, axis=0)  # [M, 96]
    ensemble_pred = np.percentile(arr, quantile * 100, axis=0)

    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    # IQR 近似正态标准差
    ensemble_conf = (q75 - q25) / 1.349

    return ensemble_pred, ensemble_conf


def ensemble_predictions(
    predictions: List[np.ndarray],
    confidences: List[np.ndarray] | None = None,
    model_weights: np.ndarray | None = None,
    method: str = "simple_average",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    多模型预测集成入口。

    Parameters
    ----------
    predictions : List[np.ndarray]
        M 个模型的预测序列，每个 (96,)。
    confidences : List[np.ndarray] | None
        M 个模型的置信度序列。method='variance_weighting' 时必需。
    model_weights : np.ndarray | None
        M 个权重。method='simple_average' 时可选。
    method : str
        集成方法，可选:
        - 'simple_average'    : 简单加权平均（默认）
        - 'variance_weighting': 方差倒数加权
        - 'quantile_fusion'   : 分位数融合（默认中位数）

    Returns
    -------
    tuple (ensemble_pred, ensemble_conf)
        ensemble_pred : np.ndarray [96]
        ensemble_conf : np.ndarray [96]
    """
    method = method.lower().replace("-", "_")

    if method == "simple_average":
        return simple_average(predictions, model_weights)
    elif method == "variance_weighting":
        if confidences is None:
            raise ValueError(
                "variance_weighting 需要提供 confidences"
            )
        return variance_weighting(predictions, confidences)
    elif method == "quantile_fusion":
        return quantile_fusion(predictions)
    else:
        raise ValueError(f"未知的集成方法: {method}")
