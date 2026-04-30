"""
将 test_in_feature_ori.csv（原始预测特征）转换为模型可用的完整特征 DataFrame。

核心逻辑:
    1. 读取 test CSV，映射中文列名为英文
    2. 用 forecast 值填充缺失的 actual 值（测试期实际值未知）
    3. 从 weather_features.csv 获取同期气象数据并重采样到15分钟
    4. 拼接 aligned_15min_full.csv 的历史尾部（确保滚动窗口足够）
    5. 重新计算派生特征与时间特征
    6. 返回完整特征 DataFrame（含 test 期间）
"""

import sys
from pathlib import Path

# 允许导入根目录的 Dataaligning
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from Dataaligning import PowerDataAligner


_BASE_DIR = Path(__file__).parent  # power_price/


def prepare_test_features(
    test_csv: str | None = None,
    aligned_csv: str | None = None,
    weather_csv: str | None = None,
    price_col: str = "A",
) -> pd.DataFrame:
    """
    从原始 test 特征构造完整模型输入特征。

    Returns
    -------
    pd.DataFrame
        索引为北京时间时间戳，包含 test 期间（以及足够历史）的完整特征。
        调用方可直接传给 inference.PricePredictor.predict_batch()。
    """
    if test_csv is None:
        test_csv = str(_BASE_DIR / "test_data" / "test_in_feature_ori.csv")
    if aligned_csv is None:
        aligned_csv = str(_BASE_DIR / "data" / "aligned_15min_full.csv")
    if weather_csv is None:
        weather_csv = str(_BASE_DIR / "data" / "weather_features.csv")

    print("=" * 60)
    print("开始构造 Test 期间完整特征")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 读取并解析 test CSV
    # ------------------------------------------------------------------
    print(f"[1/5] 加载原始 test 特征: {test_csv}")
    test_df = pd.read_csv(test_csv, parse_dates=["times"])
    test_df.set_index("times", inplace=True)
    test_df.index = pd.to_datetime(test_df.index).tz_localize(None)
    print(f"  Test 期间: {test_df.index.min()} ~ {test_df.index.max()}, {len(test_df)} 行")

    # 中文 -> 英文列名映射
    rename_map = {
        "系统负荷预测值": "load_forecast",
        "风光总加预测值": "renewable_forecast",
        "联络线预测值": "tie_line_forecast",
        "风电预测值": "wind_forecast",
        "光伏预测值": "solar_forecast",
        "水电预测值": "hydro_forecast",
        "非市场化机组预测值": "non_market_forecast",
    }
    test_df.rename(columns=rename_map, inplace=True)

    # 用 forecast 填充 actual（测试期最佳估计）
    for prefix in ["load", "renewable", "wind", "solar", "hydro", "non_market", "tie_line"]:
        test_df[f"{prefix}_actual"] = test_df[f"{prefix}_forecast"]

    # 价格列为目标变量，测试期缺失
    test_df[price_col] = np.nan

    # ------------------------------------------------------------------
    # 2. 加载气象数据并重采样到 15 分钟
    # ------------------------------------------------------------------
    print(f"[2/5] 加载气象数据: {weather_csv}")
    weather_df = pd.read_csv(weather_csv, parse_dates=["timestamp"], index_col="timestamp")
    weather_df.index = pd.to_datetime(weather_df.index).tz_localize(None)

    aligner = PowerDataAligner(price_col=price_col)
    weather_15m = aligner.resample_to_15min(weather_df, "weather")
    # 去除时区，确保与 test_df 一致
    if weather_15m.index.tz is not None:
        weather_15m.index = weather_15m.index.tz_localize(None)

    # 合并 test 与气象（按最近邻，容忍15分钟）
    merged_test = pd.merge_asof(
        test_df.sort_index(),
        weather_15m.sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )
    print(f"  合并气象后维度: {merged_test.shape}")

    # ------------------------------------------------------------------
    # 3. 加载历史 aligned 数据作为上下文
    # ------------------------------------------------------------------
    print(f"[3/5] 加载历史对齐数据: {aligned_csv}")
    aligned_df = pd.read_csv(aligned_csv, parse_dates=["times"], index_col="times")
    aligned_df.index = pd.to_datetime(aligned_df.index).tz_localize(None)

    # 只取 test 开始之前的数据，避免重叠
    test_start = merged_test.index.min()
    hist_cutoff = test_start - pd.Timedelta(minutes=15)
    aligned_hist = aligned_df[aligned_df.index <= hist_cutoff].copy()
    print(f"  历史数据: {aligned_hist.index.min()} ~ {aligned_hist.index.max()}, {len(aligned_hist)} 行")

    # ------------------------------------------------------------------
    # 4. 拼接并重新计算派生特征
    # ------------------------------------------------------------------
    print("[4/5] 拼接历史 + Test 并重新计算派生特征")
    # 确保列对齐（concat 会自动处理）
    combined = pd.concat([aligned_hist, merged_test], sort=False)
    combined.sort_index(inplace=True)

    # 关键：测试期间无法获取真实电价，A 列必须保持 NaN，
    # 使得 calculate_power_system_features 中的 curtailment_flag 被正确置 0
    test_mask = combined.index >= test_start
    combined.loc[test_mask, price_col] = np.nan

    # 重新计算电力系统特征与时间特征
    combined = aligner.calculate_power_system_features(combined)
    combined = aligner.add_cyclical_time_features(combined)

    # 添加策略特征与滞后特征（与训练时特征工程保持一致）
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from features import add_strategy_features, add_lag_features
    combined = add_strategy_features(combined, target_col=price_col)
    combined = add_lag_features(combined, target_col=price_col)

    # 清理极端值与缺失值（A 列除外，保持 NaN）
    combined = combined.replace([np.inf, -np.inf], np.nan)
    fill_cols = [c for c in combined.columns if c != price_col]
    combined[fill_cols] = combined[fill_cols].ffill().bfill().fillna(0)
    # 历史期间的 A 保留原值，test 期间保持 NaN

    print(f"  拼接后总维度: {combined.shape}")

    # ------------------------------------------------------------------
    # 5. 校验 test 期间是否包含模型所需全部特征
    # ------------------------------------------------------------------
    print("[5/5] 特征完整性校验")
    mask_path = Path(__file__).parent / "data" / "feature_mask.json"
    if mask_path.exists():
        import json
        with open(mask_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)
        missing = [c for c in selected_features if c not in combined.columns]
        if missing:
            print(f"  [警告] 缺失特征 ({len(missing)} 个): {missing[:10]}...")
        else:
            print(f"  全部 {len(selected_features)} 个特征均已就绪")
    else:
        print(f"  未找到 {mask_path}，跳过校验")

    return combined


if __name__ == "__main__":
    df = prepare_test_features()
    # 仅保留 test 期间的部分供查看
    test_start = pd.Timestamp("2026-01-01 00:00")
    test_part = df[df.index >= test_start]
    print(f"\nTest 期间特征预览 (前5行):")
    print(test_part.head())
    print(f"\nTest 期间特征列数: {len(test_part.columns)}")
