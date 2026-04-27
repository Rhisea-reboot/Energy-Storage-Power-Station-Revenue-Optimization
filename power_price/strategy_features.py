"""
strategy_features.py
===================
Day 5 物理特征辅助模块 — 为储能策略注入强相关物理特征。

本模块由成员C（策略算法师）提供，供成员A集成到 align_pipeline.py 中使用。

接口函数:
    add_strategy_features(df: pd.DataFrame) -> pd.DataFrame

新增特征列:
    - deep_peak_risk          : 净负荷深度调峰风险标记 (0/1)
    - expected_price_volatility: 竞价空间未来2小时滚动标准差
    - is_extreme_renewable    : 极端渗透率标记 (0/1)
    - midday_solar_risk       : 午间光伏风险标记 (0/1)
    - evening_ramp_stress     : 晚高峰爬坡压力标记 (0/1)
    - tie_line_margin         : 联络线裕度 (MW)
"""

import pandas as pd
import numpy as np


def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算与储能充放电策略强相关的物理特征，并注入到主数据表中。

    Parameters
    ----------
    df : pd.DataFrame
        对齐后的15分钟分辨率数据表，必须包含以下列：
        - load_forecast
        - renewable_forecast
        - hydro_forecast
        - wind_actual
        - solar_actual
        - price
        - bidding_space_forecast
        - net_load_forecast

    Returns
    -------
    pd.DataFrame
        注入6个策略特征列后的数据表。
    """
    required_cols = [
        "load_forecast",
        "renewable_forecast",
        "hydro_forecast",
        "wind_actual",
        "solar_actual",
        "price",
        "bidding_space_forecast",
        "net_load_forecast",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"add_strategy_features: 缺失必需列 {missing}")

    df = df.copy()

    # ------------------------------------------------------------------
    # 5.1 净负荷深度调峰风险标记
    #     高风电 + 低负荷 + 低温(供暖期) 组合条件
    #     供暖期近似：11月15日 ~ 次年3月15日
    # ------------------------------------------------------------------
    is_heating_season = False
    if isinstance(df.index, pd.DatetimeIndex):
        month = df.index.month
        day = df.index.day
        is_heating_season = (
            ((month == 11) & (day >= 15))
            | (month.isin([12, 1, 2]))
            | ((month == 3) & (day <= 15))
        )
    else:
        is_heating_season = pd.Series(False, index=df.index)

    high_wind = df["wind_actual"] > df["wind_actual"].quantile(0.75)
    low_load = df["load_forecast"] < df["load_forecast"].quantile(0.25)
    df["deep_peak_risk"] = (
        (high_wind & low_load & is_heating_season)
    ).astype(int)

    # ------------------------------------------------------------------
    # 5.2 竞价空间波动率代理
    #     未来2小时滚动标准差（8个15分钟点）
    #     注意：使用当前及过去数据，避免未来信息泄露
    # ------------------------------------------------------------------
    df["expected_price_volatility"] = (
        df["bidding_space_forecast"]
        .rolling(window=8, min_periods=1)
        .std()
        .fillna(0.0)
    )

    # ------------------------------------------------------------------
    # 5.3 极端渗透率标记
    #     renewable_penetration = renewable_forecast / load_forecast
    #     若 > 0.6 标记为极端
    # ------------------------------------------------------------------
    renewable_penetration = (
        df["renewable_forecast"] / df["load_forecast"].replace(0, np.nan)
    ).fillna(0.0)
    df["is_extreme_renewable"] = (renewable_penetration > 0.6).astype(int)

    # ------------------------------------------------------------------
    # 5.4 分时段供需结构标记
    #     午间光伏风险: 10:00-14:00 且 solar_actual > q75
    #     晚高峰爬坡压力: 18:00-22:00 且 net_load_forecast 变化率大
    # ------------------------------------------------------------------
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        midday = (hour >= 10) & (hour < 14)
        evening = (hour >= 18) & (hour < 22)
    else:
        midday = pd.Series(False, index=df.index)
        evening = pd.Series(False, index=df.index)

    high_solar = df["solar_actual"] > df["solar_actual"].quantile(0.75)
    df["midday_solar_risk"] = (midday & high_solar).astype(int)

    # 晚高峰净负荷爬坡压力 = 未来2小时净负荷增幅
    net_load_ramp = (
        df["net_load_forecast"]
        .diff(periods=8)
        .fillna(0.0)
    )
    df["evening_ramp_stress"] = (
        evening & (net_load_ramp > net_load_ramp.quantile(0.75))
    ).astype(int)

    # ------------------------------------------------------------------
    # 5.5 联络线裕度计算
    #     假设联络线总容量为 8000 MW（蒙西常见外送通道规模）
 #     裕度 = 容量 - |实际功率绝对值|，这里用 net_load_forecast 代理
    # ------------------------------------------------------------------
    TIE_LINE_CAPACITY_MW = 8000.0
    df["tie_line_margin"] = (
        TIE_LINE_CAPACITY_MW - df["net_load_forecast"].abs()
    ).clip(lower=0.0)

    return df
