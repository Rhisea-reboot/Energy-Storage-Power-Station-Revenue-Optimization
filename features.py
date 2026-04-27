import pandas as pd
import numpy as np
from pathlib import Path
import json

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=== 添加时间特征 ===")
    slot = df.index.hour * 4 + df.index.minute // 15
    dow = df.index.dayofweek
    month = df.index.month

    df['time_sin'] = np.sin(2 * np.pi * slot / 96)
    df['time_cos'] = np.cos(2 * np.pi * slot / 96)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    df['is_weekend'] = (dow >= 5).astype(int)
    return df

def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=== 添加策略特征 ===")
    if 'load_forecast' in df.columns and 'renewable_forecast' in df.columns:
        df['net_load_forecast'] = df['load_forecast'] - df['renewable_forecast']
    if 'price' in df.columns:
        df['price_rolling_mean_8'] = df['price'].rolling(8, min_periods=1).mean()
        df['price_rolling_std_8'] = df['price'].rolling(8, min_periods=1).std()
    return df

def add_lag_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    print("=== 添加滞后特征 ===")
    for c in cols:
        if c in df.columns:
            df[f'{c}_lag_1'] = df[c].shift(1)
            df[f'{c}_lag_96'] = df[c].shift(96)
    return df

def select_top_features(df: pd.DataFrame, target_col='A', top_k=50):
    print("=== 特征筛选与持久化 ===")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()[target_col].abs().sort_values(ascending=False)
    # 选出前K个特征（不含目标列）
    cols = [c for c in corr.index if c != target_col][:top_k]
    
    # 保存掩码，供 Dataset.py 和 Predictor 使用
    Path("power_price/data/feature_mask.json").write_text(json.dumps(cols, indent=2))
    return df[cols + [target_col]], cols

def build_feature_pipeline(df: pd.DataFrame, target_col='A'):
    df = add_temporal_features(df)
    df = add_strategy_features(df)
    lag_cols = [c for c in ['price', 'load_forecast'] if c in df.columns]
    df = add_lag_features(df, lag_cols)
    df = sanitize_dataframe(df)
    df, features = select_top_features(df, target_col)
    return df, features