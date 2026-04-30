import pandas as pd
import numpy as np
from pathlib import Path
import json

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    # 对仍存在的 NaN 用 0 填充（而非列均值，避免数据泄露）
    df = df.fillna(0)
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=== 添加时间特征 ===")
    slot = df.index.hour * 4 + df.index.minute // 15
    dow = df.index.dayofweek
    month = df.index.month
    dayofyear = df.index.dayofyear

    df['time_sin'] = np.sin(2 * np.pi * slot / 96)
    df['time_cos'] = np.cos(2 * np.pi * slot / 96)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    df['is_weekend'] = (dow >= 5).astype(int)
    df['is_peak_hour'] = ((df.index.hour >= 8) & (df.index.hour <= 22)).astype(int)
    # 年内周期（处理跨年）
    df['year_phase_sin'] = np.sin(2 * np.pi * dayofyear / 365)
    df['year_phase_cos'] = np.cos(2 * np.pi * dayofyear / 365)
    return df

def add_strategy_features(df: pd.DataFrame, target_col: str = 'A') -> pd.DataFrame:
    print("=== 添加策略特征 ===")
    new_cols = {}

    # 净负荷
    if 'load_forecast' in df.columns and 'renewable_forecast' in df.columns:
        if 'net_load_forecast' not in df.columns:
            new_cols['net_load_forecast'] = df['load_forecast'] - df['renewable_forecast']

    # 价格滚动统计（使用目标列 A 而非 price）
    price_col = target_col if target_col in df.columns else 'price'
    if price_col in df.columns:
        # 多时间尺度滚动统计
        for window in [8, 24, 96, 192, 672]:
            for suffix, func in [('rolling_mean', 'mean'), ('rolling_std', 'std'),
                                 ('rolling_max', 'max'), ('rolling_min', 'min')]:
                col_name = f'{price_col}_{suffix}_{window}'
                if col_name not in df.columns:
                    new_cols[col_name] = getattr(df[price_col].rolling(window, min_periods=1), func)()

        # 价格差分（变化率）
        for suffix, func in [('diff_1', lambda s: s.diff(1)),
                             ('diff_96', lambda s: s.diff(96)),
                             ('pct_change_1', lambda s: s.pct_change(1).replace([np.inf, -np.inf], 0))]:
            col_name = f'{price_col}_{suffix}'
            if col_name not in df.columns:
                new_cols[col_name] = func(df[price_col])

        # 日内统计（每天的价格均值、标准差）
        date_idx = pd.Series(df.index.date, index=df.index)
        for suffix, func in [('daily_mean', 'mean'), ('daily_std', 'std')]:
            col_name = f'{price_col}_{suffix}'
            if col_name not in df.columns:
                new_cols[col_name] = df.groupby(date_idx)[price_col].transform(func)
        zscore_name = f'{price_col}_daily_zscore'
        if zscore_name not in df.columns and f'{price_col}_daily_mean' in new_cols and f'{price_col}_daily_std' in new_cols:
            new_cols[zscore_name] = (df[price_col] - new_cols[f'{price_col}_daily_mean']) / (new_cols[f'{price_col}_daily_std'] + 1e-6)

    # 负荷与新能源的滚动统计
    for col in ['load_forecast', 'renewable_forecast', 'wind_forecast', 'solar_forecast']:
        if col in df.columns:
            for window in [8, 96]:
                for suffix, func in [('rolling_mean', 'mean'), ('rolling_std', 'std')]:
                    col_name = f'{col}_{suffix}_{window}'
                    if col_name not in df.columns:
                        new_cols[col_name] = getattr(df[col].rolling(window, min_periods=1), func)()

    # 峰谷比：日内最高负荷 / 最低负荷
    if 'load_forecast' in df.columns and 'load_peak_valley_ratio' not in df.columns:
        date_idx = pd.Series(df.index.date, index=df.index)
        daily_max_load = df.groupby(date_idx)['load_forecast'].transform('max')
        daily_min_load = df.groupby(date_idx)['load_forecast'].transform('min')
        new_cols['load_peak_valley_ratio'] = daily_max_load / (daily_min_load + 1e-6)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str = 'A') -> pd.DataFrame:
    print("=== 添加滞后特征 ===")
    new_cols = {}

    # 目标变量滞后（核心时序特征）
    if target_col in df.columns:
        for lag in [1, 96, 192, 672]:
            col_name = f'{target_col}_lag_{lag}'
            if col_name not in df.columns:
                new_cols[col_name] = df[target_col].shift(lag)

    # 关键外生变量滞后
    lag_cols = [c for c in ['load_forecast', 'renewable_forecast', 'bidding_space_forecast'] if c in df.columns]
    for c in lag_cols:
        for lag in [1, 96, 672]:
            col_name = f'{c}_lag_{lag}'
            if col_name not in df.columns:
                new_cols[col_name] = df[c].shift(lag)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

def select_top_features(df: pd.DataFrame, target_col='A', top_k=50):
    print("=== 特征筛选与持久化 ===")
    num_df = df.select_dtypes(include=[np.number])

    # 目标列存在缺失值时，只用非缺失行计算相关性
    if target_col in num_df.columns and num_df[target_col].isna().any():
        valid_mask = num_df[target_col].notna()
        num_df_valid = num_df[valid_mask]
        corr = num_df_valid.corr()[target_col].abs().sort_values(ascending=False)
    else:
        corr = num_df.corr()[target_col].abs().sort_values(ascending=False)

    # 排除目标列自身，以及 test 期间不可用的 A-衍生特征
    # 测试期不提供历史真实电价，A_lag / A_rolling 等特征在 test 期间无法正确计算
    candidate_cols = [c for c in corr.index
                      if c != target_col and not c.startswith(f"{target_col}_")]

    # 时间周期特征强制保留：Pearson 相关性低但对日内变化至关重要
    temporal_cols = [
        'time_sin', 'time_cos',
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos',
        'year_phase_sin', 'year_phase_cos',
        'is_weekend', 'is_peak_hour',
    ]
    forced = [c for c in temporal_cols if c in candidate_cols]
    for c in forced:
        candidate_cols.remove(c)

    cols = forced + [c for c in candidate_cols[:top_k - len(forced)]]

    print(f"  筛选后 {len(cols)} 维特征（强制保留 {len(forced)} 个时间特征，已排除 {target_col}_* 衍生特征）")

    # 保存掩码，供 Dataset.py 和 Predictor 使用
    mask_path = Path("power_price/data/feature_mask.json")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_text(json.dumps(cols, indent=2))
    return df[cols + [target_col]], cols

def build_feature_pipeline(df: pd.DataFrame, target_col='A'):
    df = add_temporal_features(df)
    df = add_strategy_features(df, target_col=target_col)
    df = add_lag_features(df, target_col=target_col)
    df = sanitize_dataframe(df)
    df, features = select_top_features(df, target_col)
    return df, features
