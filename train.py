import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import joblib
from pathlib import Path
from model import PriceModel
from Dataset import PowerPriceDataModule

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, conf = model(x)
        
        # 核心改动：联合 Loss。price 用 Huber，conf 拟合预测残差
        price_loss = F.huber_loss(pred, y)
        error = torch.abs(pred.detach() - y)
        conf_loss = F.mse_loss(conf, error) 
        
        loss = price_loss + 0.2 * conf_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从 JSON 加载特征列表，确保维度一致
    with open("power_price/data/feature_mask.json", "r") as f:
        feature_cols = json.load(f)

    data = PowerPriceDataModule(
        csv_path="power_price/data/aligned_15min_full.csv",
        train_end_date="2025-10-31",
        val_start_date="2025-11-01",
        feature_cols=feature_cols, # 注入掩码
        target_col='A',
        batch_size=32
    )

    model = PriceModel(feature_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    save_path = Path("power_price/models")
    save_path.mkdir(parents=True, exist_ok=True)

    best_loss = 1e9
    for epoch in range(2):
        train_loss = train_one_epoch(model, data.get_train_loader(), optimizer, device)
        # 验证逻辑简化为 price MSE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in data.get_val_loader():
                p, _ = model(x.to(device))
                val_loss += F.mse_loss(p, y.to(device)).item()
        val_loss /= len(data.get_val_loader())

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path / "best_model.pt")
            print(">>> Model Saved")

    # 保存与 feature_mask 维度一致的 scaler，供推理使用
    scaler_path = Path("power_price/data/scaler.pkl")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.train_dataset.get_scaler(), scaler_path)
    print(f">>> Scaler Saved ({len(feature_cols)} features) to {scaler_path}")

if __name__ == "__main__":
    main()