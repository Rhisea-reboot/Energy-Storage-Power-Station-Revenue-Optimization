import torch
import torch.nn as nn

class PriceModel(nn.Module):
    """Per-timestep MLP: 每个时间点独立预测电价，通过时间特征捕获日内规律"""
    def __init__(self, feature_dim, hidden=256, dropout=0.1):
        super().__init__()
        print(f"=== MLP模型: 输入特征维度={feature_dim}, hidden={hidden} ===")

        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden // 2)
        self.fc4 = nn.Linear(hidden // 2, hidden // 4)
        self.head = nn.Linear(hidden // 4, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, F] — 每个时间点的特征
        返回: price [B], conf [B] (conf 暂为占位)
        """
        h = self.dropout(torch.relu(self.fc1(x)))
        h = self.dropout(torch.relu(self.fc2(h)))
        h = self.dropout(torch.relu(self.fc3(h)))
        h = self.dropout(torch.relu(self.fc4(h)))
        price = self.head(h).squeeze(-1)
        conf = torch.zeros_like(price)
        return price, conf


if __name__ == "__main__":
    model = PriceModel(feature_dim=50)
    x = torch.randn(32, 50)
    p, c = model(x)
    print(f"输入: {x.shape}, 输出: price={p.shape}, conf={c.shape}")
