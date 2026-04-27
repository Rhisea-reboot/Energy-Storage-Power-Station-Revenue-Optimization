import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import joblib
warnings.filterwarnings('ignore')

class PowerPriceDataset(Dataset):
    """
    电力现货电价预测数据集 (Day 3 核心实现)
    
    结构：
    - 输入：过去 lookback_window 个时间步 (默认672个点 = 7天×96点)
    - 输出：未来 forecast_horizon 个时间步 (默认96个点 = 1天)
    - 分辨率：15分钟
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'price',
                 lookback_window: int = 672,      # 过去7天 (7×96)
                 forecast_horizon: int = 96,       # 未来1天 (1×96)
                 scaler: Optional[StandardScaler] = None,
                 mode: str = 'train'):
        """
        Args:
            df: 对齐后的DataFrame (Day 2输出)
            feature_cols: 特征列名列表
            target_col: 目标列名（电价）
            lookback_window: 历史窗口长度（15分钟点数）
            forecast_horizon: 预测 horizon（15分钟点数）
            scaler: 预训练的StandardScaler（验证集传入，训练集None）
            mode: 'train' 或 'val'
        """
        self.df = df.sort_index()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        
        # 提取特征矩阵和目标向量
        self.features = df[feature_cols].values  # (N, F)
        self.targets = df[target_col].values     # (N,)
        
        # 标准化处理（关键：防止数据泄露）
        if mode == 'train' and scaler is None:
            # 训练集：拟合标准化器
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            print(f"[{mode}] StandardScaler fitted on training data")
        elif scaler is not None:
            # 验证集/测试集：使用训练集的scaler
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
            print(f"[{mode}] Using pre-fitted StandardScaler")
        else:
            raise ValueError("Validation/Test set must provide a pre-fitted scaler")
        
        # 构建样本索引（滑动窗口）
        # 每个样本需要: [t-lookback, t] 预测 [t+1, t+forecast]
        self.indices = self._build_indices()
        
        print(f"[{mode}] Dataset initialized:")
        print(f"  总时间步: {len(df)}, 特征维度: {len(feature_cols)}")
        print(f"  有效样本数: {len(self.indices)}")
        print(f"  输入窗口: {lookback_window}点 ({lookback_window/96:.1f}天)")
        print(f"  输出 horizon: {forecast_horizon}点 ({forecast_horizon/96:.1f}天)")
    
    def _build_indices(self) -> List[int]:
        """
        构建有效样本的起始索引
        确保每个样本都有完整的历史和未来数据
        """
        valid_indices = []
        total_len = len(self.df)
        
        # 滑动窗口步长可以调整（这里用1步长=15分钟，数据量大时可用更大步长）
        stride = 1  # 每15分钟一个样本
        
        for i in range(0, total_len - self.lookback_window - self.forecast_horizon + 1, stride):
            # 检查时间连续性（不能有缺失）
            start_time = self.df.index[i]
            end_history = self.df.index[i + self.lookback_window - 1]
            end_forecast = self.df.index[i + self.lookback_window + self.forecast_horizon - 1]
            
            # 验证时间间隔是否正确（15分钟）
            expected_delta = pd.Timedelta(minutes=15)
            actual_delta_hist = end_history - start_time
            actual_delta_forecast = end_forecast - end_history
            
            if (actual_delta_hist == expected_delta * (self.lookback_window - 1) and 
                actual_delta_forecast == expected_delta * self.forecast_horizon):
                valid_indices.append(i)
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            x: (lookback_window, n_features) 历史特征
            y: (forecast_horizon,) 未来电价
        """
        start_idx = self.indices[idx]
        
        # 历史窗口 [start, start+lookback)
        x = self.features[start_idx : start_idx + self.lookback_window]
        
        # 预测目标 [start+lookback, start+lookback+forecast)
        target_start = start_idx + self.lookback_window
        y = self.targets[target_start : target_start + self.forecast_horizon]
        
        # 转为Tensor
        x = torch.FloatTensor(x)   # (672, F)
        y = torch.FloatTensor(y)   # (96,)
        
        return x, y
    
    def get_scaler(self) -> StandardScaler:
        """返回标准化器，供验证集使用"""
        return self.scaler


class PowerPriceDataModule:
    """
    数据模块：处理训练/验证分割、DataLoader创建
    严格按时间序列分割（非随机），防止数据泄露
    """
    
    def __init__(self, 
                 csv_path: str,
                 train_end_date: str,      # 训练集截止日期，如 '2024-10-31'
                 val_start_date: str,      # 验证集开始日期，如 '2024-11-01'
                 feature_cols: Optional[List[str]] = None,
                 target_col: str = 'price',
                 lookback_window: int = 672,
                 forecast_horizon: int = 96,
                 batch_size: int = 32):
        
        self.csv_path = Path(csv_path).expanduser()
        self.train_end_date = pd.Timestamp(train_end_date)
        self.val_start_date = pd.Timestamp(val_start_date)
        self.target_col = target_col
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        
        # 加载数据
        self.df = pd.read_csv(self.csv_path, parse_dates=['times'], index_col='times')
        self.df.index = self.df.index.tz_localize(None)  # 去掉时区，便于与训练/验证日期比较
        print(f"数据加载完成: {len(self.df)} 行, {self.df.index[0]} 到 {self.df.index[-1]}")
        
        # 自动识别特征列（如果未指定）
        if feature_cols is None:
            # 排除目标列、日期列、辅助列
            exclude_cols = [target_col, 'date', 'is_weekend', 'is_peak_hour']  # 保留时间编码特征
            self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        else:
            self.feature_cols = feature_cols
        
        print(f"使用特征列 ({len(self.feature_cols)} 个): {self.feature_cols[:5]}...")
        
        # 分割数据集（时间序列分割！严禁随机打乱）
        self.train_df = self.df[self.df.index <= self.train_end_date]
        self.val_df = self.df[self.df.index >= self.val_start_date]
        
        # 检查分割点是否连续
        gap = self.val_df.index.min() - self.train_df.index.max()
        print(f"\n数据分割:")
        print(f"  训练集: {self.train_df.index[0]} 到 {self.train_df.index[-1]} ({len(self.train_df)} 点)")
        print(f"  验证集: {self.val_df.index[0]} 到 {self.val_df.index[-1]} ({len(self.val_df)} 点)")
        print(f"  间隔: {gap}")
        
        # 初始化数据集（训练集先fit scaler）
        self.train_dataset = PowerPriceDataset(
            self.train_df, 
            self.feature_cols, 
            self.target_col,
            lookback_window, 
            forecast_horizon,
            scaler=None,  # 训练集创建新的scaler
            mode='train'
        )
        
        # 验证集使用训练集的scaler
        self.val_dataset = PowerPriceDataset(
            self.val_df,
            self.feature_cols,
            self.target_col,
            lookback_window,
            forecast_horizon,
            scaler=self.train_dataset.get_scaler(),  # 关键：防止数据泄露
            mode='val'
        )
    
    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 时间序列数据不打乱！
            num_workers=0,  # 调试时设为0，生产环境可增大
            pin_memory=True
        )
    
    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_feature_dim(self) -> int:
        return len(self.feature_cols)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        'csv_path': 'power_price/data/aligned_15min_full.csv',  # Day 2 输出
        'train_end_date': '2025-10-31',  # 前10个月训练
        'val_start_date': '2025-11-01',  # 后2个月验证
        'target_col': 'A',                # 蒙西节点电价列名
        'lookback_window': 672,   # 7天历史 (7×96)
        'forecast_horizon': 96,   # 1天预测 (1×96)
        'batch_size': 32
    }
    
    print("="*60)
    print("Day 3: PyTorch Dataset 构建")
    print("="*60)
    
    # 创建数据模块
    data_module = PowerPriceDataModule(**CONFIG)
    
    # 获取DataLoader
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    
    # 测试一个batch
    print("\n测试训练集第一个batch:")
    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch shape: x={x.shape}, y={y.shape}")  # x: (32, 672, F), y: (32, 96)
        print(f"输入特征维度: {x.shape[-1]}")
        print(f"输入时间范围: 过去{x.shape[1]/96:.1f}天")
        print(f"输出时间范围: 未来{y.shape[1]/96:.1f}天")
        break
    
    # 统计信息
    print(f"\n数据加载器就绪:")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"特征维度: {data_module.get_feature_dim()}")
    
    # 保存scaler供后续使用（重要！）
    scaler_path = Path('power_price/data/scaler.pkl').expanduser()
    joblib.dump(data_module.train_dataset.get_scaler(), scaler_path)
    print(f"\nStandardScaler已保存至: {scaler_path}")