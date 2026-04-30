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
    """Per-timestep 数据集：每个时间点独立预测电价"""

    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'price',
                 scaler: Optional[StandardScaler] = None,
                 mode: str = 'train'):

        self.df = df.sort_index()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.mode = mode

        # 提取特征矩阵和目标向量
        self.features = df[feature_cols].values  # (N, F)
        self.targets = df[target_col].values     # (N,)

        # 标准化
        if mode == 'train' and scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            print(f"[{mode}] StandardScaler fitted on training data")
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
            print(f"[{mode}] Using pre-fitted StandardScaler")
        else:
            raise ValueError("Validation/Test set must provide a pre-fitted scaler")

        # 过滤有效目标值的行
        valid_mask = ~np.isnan(self.targets)
        self.valid_indices = np.where(valid_mask)[0]

        print(f"[{mode}] Dataset initialized:")
        print(f"  总时间步: {len(df)}, 特征维度: {len(feature_cols)}")
        print(f"  有效样本数: {len(self.valid_indices)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row_idx = self.valid_indices[idx]
        x = torch.FloatTensor(self.features[row_idx])   # (F,)
        y = torch.FloatTensor([self.targets[row_idx]])  # (1,)
        return x, y

    def get_scaler(self) -> StandardScaler:
        return self.scaler


class PowerPriceDataModule:
    """数据模块：按时间序列分割训练/验证"""

    def __init__(self,
                 csv_path: str,
                 train_end_date: str,
                 val_start_date: str,
                 feature_cols: Optional[List[str]] = None,
                 target_col: str = 'price',
                 batch_size: int = 256):

        self.csv_path = Path(csv_path).expanduser()
        self.train_end_date = pd.Timestamp(train_end_date)
        self.val_start_date = pd.Timestamp(val_start_date)
        self.target_col = target_col
        self.batch_size = batch_size

        self.df = pd.read_csv(self.csv_path, parse_dates=['times'], index_col='times')
        self.df.index = self.df.index.tz_localize(None)
        print(f"数据加载完成: {len(self.df)} 行, {self.df.index[0]} 到 {self.df.index[-1]}")

        if feature_cols is None:
            exclude_cols = [target_col, 'date', 'is_weekend', 'is_peak_hour']
            self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        else:
            self.feature_cols = feature_cols

        print(f"使用特征列 ({len(self.feature_cols)} 个): {self.feature_cols[:5]}...")

        # 时序分割
        self.train_df = self.df[self.df.index <= self.train_end_date]
        self.val_df = self.df[self.df.index >= self.val_start_date]

        gap = self.val_df.index.min() - self.train_df.index.max()
        print(f"\n数据分割:")
        print(f"  训练集: {self.train_df.index[0]} 到 {self.train_df.index[-1]} ({len(self.train_df)} 点)")
        print(f"  验证集: {self.val_df.index[0]} 到 {self.val_df.index[-1]} ({len(self.val_df)} 点)")
        print(f"  间隔: {gap}")

        self.train_dataset = PowerPriceDataset(
            self.train_df, self.feature_cols, self.target_col,
            scaler=None, mode='train'
        )

        self.val_dataset = PowerPriceDataset(
            self.val_df, self.feature_cols, self.target_col,
            scaler=self.train_dataset.get_scaler(), mode='val'
        )

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0
        )

    def get_feature_dim(self) -> int:
        return len(self.feature_cols)


if __name__ == "__main__":
    CONFIG = {
        'csv_path': 'power_price/data/aligned_15min_processed.csv',
        'train_end_date': '2025-10-31',
        'val_start_date': '2025-11-01',
        'target_col': 'A',
        'batch_size': 256
    }

    print("="*60)
    print("Per-timestep Dataset 构建")
    print("="*60)

    data_module = PowerPriceDataModule(**CONFIG)

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch: x={x.shape}, y={y.shape}")
        break

    print(f"\n训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    scaler_path = Path('power_price/data/scaler.pkl').expanduser()
    joblib.dump(data_module.train_dataset.get_scaler(), scaler_path)
    print(f"Scaler saved to {scaler_path}")
