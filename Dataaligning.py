import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict,List,Tuple,Optional
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

@dataclass
class DataAlignmentConfig:
    target_freq: str = '15min'
    target_points: int = 96
    timezone: str = 'Asia/Shanghai'
    price_interpolate: bool = False
    weather_interpolate: bool = True
    boundary_interpolate: bool = False
class PowerDataAligner:
    def __init__(self, config: DataAlignmentConfig = None,
                 tie_line_capacity: Optional[float] = None,
                 price_col: str = 'price'):
        self.config = config or DataAlignmentConfig()
        self.tie_line_capacity = tie_line_capacity
        self.price_col = price_col
        self.boundary_cols = [
            'load_actual',
            'renewable_actual',
            'tie_line_actual',
            'wind_actual',
            'solar_actual',
            'hydro_actual',
            'non_market_actual'
        ]
        self.boundary_forecast_cols = [
            'load_forecast',
            'renewable_forecast',
            'tie_line_forecast',
            'wind_forecast',
            'solar_forecast',
            'hydro_forecast',
            'non_market_forecast'
        ]
        self.weather_cols = [
            't2m_mean',
            't2m_max',
            't2m_min',
            't2m_std',
            'temp_spread',
            'ghi_mean',
            'ghi_max',
            'ghi_min',
            'ghi_std',
            'solar_potential',
            'sp_mean',
            'sp_max',
            'sp_min',
            'sp_std',
            'tcc_mean',
            'tcc_max',
            'tcc_min',
            'tcc_std',
            'wspd_mean',
            'wspd_max',
            'wind_power_potential'
        ]
    def load_raw_data(self,price_path: str,boundary_path: str,weather_path: str)-> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        print("===加载原始数据===")
        price_df = pd.read_csv(price_path,parse_dates = ['times'])
        price_df.set_index('times',inplace = True)
        price_df.index = pd.to_datetime(price_df.index)
        print(f"电价数据：{len(price_df)}行，时间范围：{price_df.index[0]}~{price_df.index[-1]}")
        boundary_df = pd.read_csv(boundary_path,parse_dates = ['times'])
        boundary_df.set_index('times',inplace = True)
        boundary_df.index = pd.to_datetime(boundary_df.index)
        boundary_df.rename(columns={
            '系统负荷实际值': 'load_actual',
            '系统负荷预测值': 'load_forecast',
            '风光总加实际值': 'renewable_actual',
            '风光总加预测值': 'renewable_forecast',
            '联络线实际值': 'tie_line_actual',
            '联络线预测值': 'tie_line_forecast',
            '风电实际值': 'wind_actual',
            '风电预测值': 'wind_forecast',
            '光伏实际值': 'solar_actual',
            '光伏预测值': 'solar_forecast',
            '水电实际值': 'hydro_actual',
            '水电预测值': 'hydro_forecast',
            '非市场化机组实际值': 'non_market_actual',
            '非市场化机组预测值': 'non_market_forecast',
        }, inplace=True)
        print(f"边界数据：{len(boundary_df)}行，时间范围：{boundary_df.index[0]}~{boundary_df.index[-1]}")
        weather_df = pd.read_csv(weather_path,parse_dates = ['timestamp'])
        weather_df.set_index('timestamp',inplace = True)
        weather_df.index = pd.to_datetime(weather_df.index)
        print(f"气象数据：{len(weather_df)}行，时间范围: {weather_df.index[0]}~{weather_df.index[-1]}")
        return price_df,boundary_df,weather_df        
    def unify_timezone(self,df: pd.DataFrame,df_name: str = "数据")-> pd.DataFrame:
        if df.index.tz is None:
            # 无TZ视为UTC（气象数据常见），转北京时间(+8)
            df.index = df.index.tz_localize('UTC').tz_convert(self.config.timezone)
            print(f"{df_name}: 假设原始为UTC，已转换为北京时间")
        else:
            df.index = df.index.tz_convert(self.config.timezone)
            print(f"{df_name}: 时区已转换至北京时间")
        return df
    def detect_frequency(self,df: pd.DataFrame,name: str) -> str:
        """
        检测数据实际分辨率
        """
        if len(df) < 2:
            return "unknown"
        median_diff = df.index.to_series().diff().median()
        minutes = median_diff.total_seconds() / 60
        if minutes == 15:
            return "15min"
        elif minutes == 60:
            return "1h"
        elif minutes == 1440:
            return "1d"
        else:
            return f"{int(minutes)}min"
    def resample_to_15min(self,df: pd.DataFrame,data_type: str,method: str = 'linear') -> pd.DataFrame:
        """
        将数据重采样到15分钟分辨率
        策略：
        - 电价: 前向填充（严禁插值，保持原始跳变）
        - 边界条件: 前向填充（实际值严禁插值）
        - 气象数据: 线性插值（物理连续变量允许插值）
        """
        original_freq = self.detect_frequency(df, data_type)
        print(f"{data_type} 原始分辨率: {original_freq}")
            
        if original_freq == "15min":
            print(f"{data_type}: 已为15分钟分辨率，跳过重采样")
            return df
            
        # 创建完整的15分钟时间索引
        full_index = pd.date_range(
            start=df.index.min().normalize(),  # 当日0点
            end=df.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),  # 次日23:45
            freq='15min',
            tz=self.config.timezone
        )
            
        # 重新索引（引入NaN）
        df_resampled = df.reindex(full_index)
            
        # 根据数据类型选择填充策略
        if data_type == "price":
            # 电价严禁插值，使用ffill（前向填充）然后bfill（后向填充）处理边界
            df_resampled = df_resampled.ffill().bfill()
            print(f"{data_type}: 使用前向填充（保持价格跳变特性）")
                
        elif data_type.startswith("boundary"):
            # 边界条件实际值严禁插值，使用ffill
            # 预测值可少量插值，但优先ffill
            df_resampled = df_resampled.ffill().bfill()
            print(f"{data_type}: 使用前向填充（保持实际值真实性）")
                
        elif data_type == "weather":
            # 气象数据允许线性插值（物理连续）
            df_resampled = df_resampled.interpolate(method='linear', limit_direction='both')
            # 剩余缺失值用均值填充
            df_resampled.fillna(df_resampled.mean(), inplace=True)
            print(f"{data_type}: 使用线性插值+均值填充")
            
        print(f"{data_type}: 重采样后 {len(df_resampled)} 行")
        return df_resampled
    def calculate_power_system_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        计算电力系统关键物理特征（严格依照比赛特征量）
        """
        print("=== 计算电力系统衍生特征 ===")
            
        # 1. 风光总加（如果原始数据没有，则计算）
        if 'renewable_actual' not in df.columns and 'wind_actual' in df.columns:
            df['renewable_actual'] = df['wind_actual'] + df['solar_actual']
            df['renewable_forecast'] = df['wind_forecast'] + df['solar_forecast']
            print("✓ 计算风光总加")
            
        # 2. 竞价空间（市场清算空间）= 负荷 - 风光 - 水电 - 非市场化
        # 这是电价的第一预测因子！
        df['bidding_space_actual'] = (
            df['load_actual'] 
            - df['renewable_actual'] 
            - df['hydro_actual'] 
            - df['non_market_actual']
        )
        df['bidding_space_forecast'] = (
            df['load_forecast'] 
            - df['renewable_forecast'] 
            - df['hydro_forecast'] 
            - df['non_market_forecast']
        )
        print("✓ 计算竞价空间（核心特征）")
            
        # 3. 净负荷（扣除新能源后的需求）
        df['net_load_actual'] = df['load_actual'] - df['renewable_actual']
        df['net_load_forecast'] = df['load_forecast'] - df['renewable_forecast']
        print("✓ 计算净负荷")
            
        # 4. 新能源渗透率
        df['renewable_penetration'] = df['renewable_actual'] / df['load_actual']
        df['renewable_penetration'] = df['renewable_penetration'].replace([np.inf, -np.inf], 0)
        print("✓ 计算新能源渗透率")
            
        # 5. 供需缺口（预测偏差）
        total_supply_forecast = (
            df['renewable_forecast'] + 
            df['hydro_forecast'] + 
            df['tie_line_forecast'] + 
            df['non_market_forecast']
        )
        df['supply_demand_gap'] = df['load_forecast'] - total_supply_forecast
        print("✓ 计算供需缺口")
            
        # 6. 联络线净受电比例
        df['tie_line_ratio'] = df['tie_line_actual'] / df['load_actual']
        df['tie_line_ratio'] = df['tie_line_ratio'].replace([np.inf, -np.inf], 0)
        print("✓ 计算联络线比例")
            
        # 7. 预测误差（系统性偏差特征）
        df['load_forecast_error'] = df['load_actual'] - df['load_forecast']
        df['wind_forecast_error'] = df['wind_actual'] - df['wind_forecast']
        df['solar_forecast_error'] = df['solar_actual'] - df['solar_forecast']
        df['renewable_forecast_error'] = df['renewable_actual'] - df['renewable_forecast']
        print("✓ 计算预测误差")
        
        # ================== 精细化特征（蒙西市场高 ROI） ==================
        
        # 8. 非线性边际效应
        df['penetration_sq'] = df['renewable_penetration'] ** 2
        df['is_extreme_renewable'] = (df['renewable_penetration'] > 0.6).astype(int)
        print("✓ 非线性边际效应 (penetration_sq, is_extreme_renewable)")
        
        # 9. 分时段供需结构
        hour = pd.Series(df.index.hour, index=df.index)
        df['midday_solar_risk'] = 0.0
        mask_midday = hour.between(10, 14)
        df.loc[mask_midday, 'midday_solar_risk'] = (
            df.loc[mask_midday, 'solar_actual'] / df.loc[mask_midday, 'load_actual']
        ).replace([np.inf, -np.inf], 0)
        
        df['evening_ramp_stress'] = 0.0
        mask_evening = hour.between(18, 22)
        df.loc[mask_evening, 'evening_ramp_stress'] = (
            df.loc[mask_evening, 'net_load_actual'] / df.loc[mask_evening, 'load_actual']
        ).replace([np.inf, -np.inf], 0)
        print("✓ 分时段供需结构 (midday_solar_risk, evening_ramp_stress)")
        
        # 10. 储能策略相关特征
        # 竞价空间短期波动 = 未来2小时的价格波动预期代理变量
        df['expected_price_volatility'] = df['bidding_space_forecast'].rolling(window=8, min_periods=1).std()
        # 深度调峰风险：高风电 + 低负荷 + 低温（供暖期）
        t2m_safe = df['t2m_mean'].fillna(df['t2m_mean'].mean())
        df['deep_peak_risk'] = (
            (df['wind_actual'] > df['wind_actual'].quantile(0.8)) &
            (df['load_actual'] < df['load_actual'].quantile(0.3)) &
            (t2m_safe < 5)
        ).astype(int)
        print("✓ 储能策略特征 (expected_price_volatility, deep_peak_risk)")
        
        # 11. 联络线精细特征
        capacity = self.tie_line_capacity
        if capacity is None:
            capacity = df['tie_line_actual'].abs().quantile(0.95) * 1.1
        df['tie_line_margin'] = capacity - df['tie_line_actual'].abs()
        df['is_high_export'] = (df['tie_line_actual'].abs() > capacity * 0.9).astype(int)
        print(f"✓ 联络线精细特征 (capacity={capacity:.0f} MW)")
        
        # 12. 蒙西市场特异性
        # 风光互补指数（滚动7日 = 672点）
        df['wind_solar_correlation_7d'] = (
            df['wind_actual'].rolling(window=672, min_periods=96)
            .corr(df['solar_actual'])
        ).fillna(0)
        # 即时风光比作为互补指数的局部代理
        df['wind_solar_instant_ratio'] = df['wind_actual'] / (df['solar_actual'] + 1e-6)
        
        # 弃风弃光隐性成本标记
        if self.price_col in df.columns:
            df['curtailment_flag'] = (
                (df['renewable_actual'] < df['renewable_forecast'] * 0.8) &
                (df[self.price_col] < 50)
            ).astype(int)
        else:
            df['curtailment_flag'] = 0
            print(f"  警告: 未找到价格列 '{self.price_col}'，curtailment_flag 置为 0")
        
        # 供热期刚性约束（11月-3月，凌晨0-6点）
        df['heating_season_rigid'] = (
            df.index.month.isin([11, 12, 1, 2, 3]) & hour.between(0, 6)
        ).astype(int)
        print("✓ 蒙西特异性特征 (wind_solar_corr, curtailment, heating_rigid)")
        
        return df
    def add_cyclical_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加周期性时间特征（Day 4 内容提前集成）
        """
        # 15分钟粒度的一天中的位置 (0-95)
        time_of_day = df.index.hour * 4 + df.index.minute // 15
        day_of_week = df.index.dayofweek
        month = df.index.month
            
        # 正余弦编码（保持周期性连续性）
        df['time_sin'] = np.sin(2 * np.pi * time_of_day / 96)
        df['time_cos'] = np.cos(2 * np.pi * time_of_day / 96)
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
            
        # 标记特征
        df['is_weekend'] = (day_of_week >= 5).astype(int)
        df['is_peak_hour'] = ((df.index.hour >= 8) & (df.index.hour <= 22)).astype(int)
            
        return df
    def validate_15min_integrity(self,df: pd.DataFrame) -> bool:
        """
        验证15分钟分辨率完整性（检测节点）
        检查：
        1. 每日是否恰好96个点
        2. 时间间隔是否严格15分钟
        3. 是否存在缺失值
        """
        print("=== 15分钟分辨率完整性检测 ===")
            
        # 按日期分组检查
        df['date'] = df.index.date
        daily_counts = df.groupby('date').size()
            
        invalid_days = daily_counts[daily_counts != 96]
        if len(invalid_days) > 0:
            print(f"发现 {len(invalid_days)} 天数据不完整（非96点）:")
            print(invalid_days.head())
            return False
            
        # 检查时间间隔
        time_diffs = df.index.to_series().diff().dropna()
        non_15min = time_diffs[time_diffs != pd.Timedelta(minutes=15)]
        if len(non_15min) > 0:
            print(f"发现 {len(non_15min)} 个非15分钟间隔")
            return False
            
        # 检查缺失值（关键特征）
        essential_cols = [self.price_col, 'load_actual', 'load_forecast', 'bidding_space_forecast']
        missing = df[essential_cols].isnull().sum()
        if missing.any():
            print(f"发现缺失值:\n{missing[missing > 0]}")
            return False
            
        print("15分钟分辨率检测通过：每日96点，无缺失，间隔正确")
        return True
    def align_all_data(self,price_path: str,boundary_path: str,weather_path: str,output_path: str) -> pd.DataFrame:
        """
        主对齐流程（Day 2 完整管线）
        """
        print("=" * 60)
        print("开始 Day 2: 多源数据对齐 (15分钟分辨率)")
        print("=" * 60)
            
        # 1. 加载数据
        price_df, boundary_df, weather_df = self.load_raw_data(
            price_path, boundary_path, weather_path
        )
            
        # 2. 统一时区（转为北京时间）
        price_df = self.unify_timezone(price_df, "电价")
        boundary_df = self.unify_timezone(boundary_df, "边界条件")
        weather_df = self.unify_timezone(weather_df, "气象")
            
        # 3. 重采样到15分钟（核心对齐步骤）
        print("\n=== 重采样到15分钟分辨率 ===")
        price_15m = self.resample_to_15min(price_df, "price")
        boundary_15m = self.resample_to_15min(boundary_df, "boundary")
        weather_15m = self.resample_to_15min(weather_df, "weather")
            
        # 4. 合并所有数据（以时间戳为键）
        print("\n=== 合并多源数据 ===")
        # 先合并边界条件和电价
        merged = pd.merge_asof(
            price_15m.sort_index(),
            boundary_15m.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('1min')  # 允许1分钟误差
        )
            
        # 再合并气象数据
        merged = pd.merge_asof(
            merged.sort_index(),
            weather_15m.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('15min')
        )
            
        print(f"合并后数据维度: {merged.shape}")
            
        # 5. 计算电力系统衍生特征
        merged = self.calculate_power_system_features(merged)
            
        # 6. 添加时间特征
        merged = self.add_cyclical_time_features(merged)
            
        # 7. 数据质量检测
        is_valid = self.validate_15min_integrity(merged)
        if not is_valid:
            print("警告：数据完整性检测未通过，请检查原始数据")
            
        # 8. 保存对齐后的数据
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path)
            
        print(f"\nDay 2 完成！对齐数据已保存至: {output_path}")
        print(f"特征维度: {len(merged.columns)}")
        print(f"时间跨度: {merged.index[0]} 至 {merged.index[-1]}")
        print(f"总样本数: {len(merged)} (应能被96整除: {len(merged) % 96 == 0})")
            
        # 输出特征列表（供Day 3使用）
        print(f"\n特征列分类:")
        print(f"- 电价相关: {[c for c in merged.columns if 'price' in c]}")
        print(f"- 边界条件实际值: {[c for c in merged.columns if 'actual' in c]}")
        print(f"- 边界条件预测值: {[c for c in merged.columns if 'forecast' in c]}")
        print(f"- 气象特征: {[c for c in merged.columns if any(w in c for w in ['t2m', 'ghi', 'sp', 'tcc', 'wspd'])]}")
        derived_keywords = ['bidding', 'net_load', 'penetration', 'gap', 'error', 'sin', 'cos',
                            'solar_risk', 'ramp_stress', 'volatility', 'peak_risk',
                            'margin', 'export', 'correlation', 'curtailment', 'heating']
        print(f"- 衍生特征: {[c for c in merged.columns if any(x in c for x in derived_keywords)]}")
        return merged
if __name__ == "__main__":
    aligner = PowerDataAligner(price_col='A')
    
    # 执行完整对齐流程
    df_aligned = aligner.align_all_data(
        price_path="power_price/data/mengxi_node_price_selected.csv",      # 历史电价
        boundary_path="power_price/data/mengxi_boundary_anon_filtered.csv",  # 边界条件（负荷、风光等）
        weather_path="power_price/data/weather_features.csv",      # Day1输出的气象特征（6变量）
        output_path="power_price/data/aligned_15min_full.csv"      # Day2输出
    )
    
    # 验证每日96点
    print("\n=== 验证示例（首日数据） ===")
    first_day = df_aligned.iloc[:96]
    print(f"首日时间点: {first_day.index[0]} ~ {first_day.index[-1]}")
    print(f"首日数据点数: {len(first_day)}")
    print(f"竞价空间范围: {first_day['bidding_space_forecast'].min():.2f} ~ {first_day['bidding_space_forecast'].max():.2f}")