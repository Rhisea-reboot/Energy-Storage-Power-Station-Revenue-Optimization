"""
strategy_wrapper.py
===================
Day 12 Python ctypes 封装层，提供调用 C++ 核心搜索的 Python 接口。

同时包含纯 Python 备用实现，以应对 C++ 编译环境不可用的情况
（O(N²) ≈ 6561 次评估在 Python 中完全可接受）。

接口函数:
    optimize_charge_discharge(prices, confidence, risk_aversion, profit_threshold) -> dict
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, Any

# ------------------------------------------------------------------
# 尝试加载编译后的动态库
# ------------------------------------------------------------------
_LIB_PATH = Path(__file__).parent / "strategy_core.so"
_lib = None

if _LIB_PATH.exists():
    try:
        _lib = ctypes.CDLL(str(_LIB_PATH))
    except OSError:
        _lib = None

# ------------------------------------------------------------------
# 声明 C 函数签名
# ------------------------------------------------------------------
if _lib is not None:
    # 基础接口（不带置信度）
    _lib.find_optimal_strategy.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # prices
        ctypes.c_int,                     # n
        ctypes.c_double,                  # risk_aversion
        ctypes.POINTER(ctypes.c_int),     # out_charge_start
        ctypes.POINTER(ctypes.c_int),     # out_discharge_start
        ctypes.POINTER(ctypes.c_double),  # out_profit
    ]
    _lib.find_optimal_strategy.restype = ctypes.c_int  # 0=成功

    # 扩展接口（带置信度）
    _lib.find_optimal_strategy_with_confidence.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # prices
        ctypes.POINTER(ctypes.c_double),  # confidence
        ctypes.c_int,                     # n
        ctypes.c_double,                  # risk_aversion
        ctypes.c_double,                  # profit_threshold
        ctypes.POINTER(ctypes.c_int),     # out_charge_start
        ctypes.POINTER(ctypes.c_int),     # out_discharge_start
        ctypes.POINTER(ctypes.c_double),  # out_profit
    ]
    _lib.find_optimal_strategy_with_confidence.restype = ctypes.c_int


# ------------------------------------------------------------------
# 纯 Python 备用实现（与 C++ 逻辑完全一致）
# ------------------------------------------------------------------
def _pure_python_optimize(
    prices: np.ndarray,
    confidence: np.ndarray | None,
    risk_aversion: float,
    profit_threshold: float,
) -> Dict[str, Any]:
    """纯 Python O(N²) 暴力搜索，作为 C++ 的降级备份。"""
    n = prices.shape[0]
    duration = 8
    max_tc = n - duration * 2      # 80
    max_td = n - duration          # 88

    best_score = -np.inf
    best_tc = -1
    best_td = -1
    best_raw_profit = 0.0

    for tc in range(0, max_tc + 1):
        for td in range(tc + duration, max_td + 1):
            charge_cost = prices[tc:tc + duration].sum() * 1000.0
            discharge_revenue = prices[td:td + duration].sum() * 1000.0
            raw_profit = discharge_revenue - charge_cost

            if confidence is not None:
                risk_penalty = (
                    confidence[tc:tc + duration].sum()
                    + confidence[td:td + duration].sum()
                ) * 1000.0
                score = raw_profit - risk_aversion * risk_penalty
            else:
                score = raw_profit

            if score > best_score:
                best_score = score
                best_tc = tc
                best_td = td
                best_raw_profit = raw_profit

    execute = (best_score > profit_threshold) and (best_tc >= 0)

    return {
        "charge_start": best_tc if execute else None,
        "discharge_start": best_td if execute else None,
        "expected_profit": float(best_raw_profit),
        "adjusted_profit": float(best_score),
        "execute": execute,
    }


# ------------------------------------------------------------------
# Python 公共接口
# ------------------------------------------------------------------
def optimize_charge_discharge(
    prices: np.ndarray,
    confidence: np.ndarray | None = None,
    risk_aversion: float = 0.1,
    profit_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Python 接口：调用 C++ 核心搜索，寻找最优充放电策略。

    Parameters
    ----------
    prices : np.ndarray
        预测电价序列，形状必须为 (96,)，dtype 为浮点型。
    confidence : np.ndarray | None, optional
        预测置信度序列（标准差或不确定性评分），形状 (96,)。
        越高表示不确定性越大。若未提供，默认使用 prices.std() * 0.1。
    risk_aversion : float, optional
        风险厌恶系数 λ，默认 0.1。取值范围建议 0.0 ~ 1.0。
    profit_threshold : float, optional
        最低执行收益阈值（元）。只有当预期收益 > threshold 时才执行交易。
        默认 0.0。

    Returns
    -------
    dict
        {
            'charge_start': int | None,        // 最优充电开始索引
            'discharge_start': int | None,     // 最优放电开始索引
            'power_profile': np.ndarray [96],  // 功率计划 (+1000/0/-1000)
            'expected_profit': float,          // 原始预期收益（元）
            'adjusted_profit': float,          // 风险调整后收益（元）
            'execute': bool,                   // 是否执行交易
        }
    """
    # ---- 输入验证 -------------------------------------------------
    if prices.shape != (96,):
        raise ValueError(f"prices must be shape (96,), got {prices.shape}")
    if not np.issubdtype(prices.dtype, np.floating):
        prices = prices.astype(np.float64)

    if confidence is None:
        confidence = np.full(96, prices.std() * 0.1, dtype=np.float64)
    else:
        if confidence.shape != (96,):
            raise ValueError(
                f"confidence must be shape (96,), got {confidence.shape}"
            )
        confidence = confidence.astype(np.float64)

    # ---- 尝试调用 C++ 库 -------------------------------------------
    if _lib is not None:
        c_start = ctypes.c_int(0)
        d_start = ctypes.c_int(0)
        profit = ctypes.c_double(0.0)

        if confidence is not None and _lib.find_optimal_strategy_with_confidence:
            status = _lib.find_optimal_strategy_with_confidence(
                prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                confidence.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                96,
                risk_aversion,
                profit_threshold,
                ctypes.byref(c_start),
                ctypes.byref(d_start),
                ctypes.byref(profit),
            )
        else:
            status = _lib.find_optimal_strategy(
                prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                96,
                risk_aversion,
                ctypes.byref(c_start),
                ctypes.byref(d_start),
                ctypes.byref(profit),
            )

        if status == 0:
            power = np.zeros(96, dtype=np.int32)
            execute = profit.value > profit_threshold

            if execute and c_start.value >= 0 and d_start.value >= 0:
                power[c_start.value : c_start.value + 8] = -1000
                power[d_start.value : d_start.value + 8] = +1000

            return {
                "charge_start": c_start.value if execute else None,
                "discharge_start": d_start.value if execute else None,
                "power_profile": power,
                "expected_profit": float(profit.value),
                "adjusted_profit": float(profit.value),
                "execute": execute,
            }
        # C++ 返回错误时，降级到 Python 实现

    # ---- 纯 Python 备用实现 ----------------------------------------
    result = _pure_python_optimize(
        prices, confidence, risk_aversion, profit_threshold
    )

    power = np.zeros(96, dtype=np.int32)
    if result["execute"]:
        cs = result["charge_start"]
        ds = result["discharge_start"]
        if cs is not None and ds is not None:
            power[cs : cs + 8] = -1000
            power[ds : ds + 8] = +1000

    result["power_profile"] = power
    return result
