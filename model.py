import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用 1D 卷积捕捉时间序列的局部特征
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x)) # 移除此处的 ReLU，在残差相加后再激活
        x += residual
        return F.relu(x)

class PriceModel(nn.Module):
    def __init__(self, feature_dim, hidden=128, layers=4, horizon=96):
        """
        feature_dim: 输入特征数，由 feature_mask.json 动态决定
        hidden: 隐藏层维度
        layers: 残差块数量
        horizon: 预测步长，默认 96 (24小时)
        """
        super().__init__()
        print(f"=== 初始化模型: 输入特征维度={feature_dim} ===")

        # 1. 线性投影：将动态的特征维度映射到统一的 hidden 空间
        self.in_proj = nn.Linear(feature_dim, hidden)

        # 2. 卷积层：用于在时间维度上提取特征
        self.conv_in = nn.Conv1d(hidden, hidden, 3, padding=1)

        # 3. 深度残差骨干网络
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden) for _ in range(layers)
        ])

        # 4. 全局特征聚合
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 5. 回归头部
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # 双头输出：电价预测 + 置信度（预测标准差）
        self.head_price = nn.Linear(hidden, horizon)
        self.head_conf = nn.Linear(hidden, horizon)

    def forward(self, x):
        """
        输入 x 形状: [Batch, 672, Feature_Dim]
        """
        # Step 1: 维度对齐 [B, 672, F] -> [B, 672, H]
        x = self.in_proj(x)
        
        # Step 2: 转换为卷积格式 [B, H, 672]
        x = x.permute(0, 2, 1)

        # Step 3: 通过卷积网络
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)

        # Step 4: 空间池化并进入 MLP [B, H]
        x = self.pool(x).squeeze(-1)
        x = self.mlp(x)

        # Step 5: 分支输出
        # 电价直接线性输出
        price = self.head_price(x)
        
        # 置信度输出必须为正值，使用 softplus 激活并加一个小量防止除以0
        # 对应训练中的 Loss: 让 conf 逼近预测残差 |pred - target|
        conf = F.softplus(self.head_conf(x)) + 1e-6

        return price, conf

# ==========================================
# 调试用：验证维度是否闭合
# ==========================================
if __name__ == "__main__":
    # 如需使用真实 feature_dim，读取 feature_mask.json:
    # import json
    # with open("power_price/data/feature_mask.json", "r") as f:
    #     feature_cols = json.load(f)
    # test_feature_dim = len(feature_cols)

    test_feature_dim = 50
    model = PriceModel(feature_dim=test_feature_dim)

    # 模拟输入：Batch=8, 时间窗口=672, 特征=50
    test_input = torch.randn(8, 672, test_feature_dim)
    p, c = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"电价预测形状: {p.shape}") # 应为 [8, 96]
    print(f"置信度预测形状: {c.shape}") # 应为 [8, 96]