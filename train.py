import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import joblib
from pathlib import Path
from model import PriceModel
from Dataset import PowerPriceDataModule
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).squeeze(-1)
        pred, _ = model(x)
        loss = F.huber_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y_dev = x.to(device), y.to(device).squeeze(-1)
            pred, _ = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_dev.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return rmse, mae, preds, targets


def train_model(
    csv_path: str = "power_price/data/aligned_15min_processed.csv",
    model_save_path: str = "power_price/models/best_model.pt",
    scaler_save_path: str = "power_price/data/scaler.pkl",
    feature_mask_path: str = "power_price/data/feature_mask.json",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    train_end_date: str = "2025-10-31",
    val_start_date: str = "2025-11-01",
    target_col: str = 'A',
    patience: int = 15,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    else:
        device = torch.device("cpu")
        print(f"[Train] WARNING: CUDA not available, using CPU | Epochs: {epochs}")

    with open(feature_mask_path, "r") as f:
        feature_cols = json.load(f)

    data = PowerPriceDataModule(
        csv_path=csv_path,
        train_end_date=train_end_date,
        val_start_date=val_start_date,
        feature_cols=feature_cols,
        target_col=target_col,
        batch_size=batch_size,
    )

    model = PriceModel(feature_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, data.get_train_loader(), optimizer, device)
        val_rmse, val_mae, _, _ = validate(model, data.get_val_loader(), device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs}: TrainLoss={train_loss:.4f}, ValRMSE={val_rmse:.4f}, ValMAE={val_mae:.4f}, LR={current_lr:.6f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Model Saved (ValRMSE={best_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  >>> Early stopping at epoch {epoch+1}")
                break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    scaler_path = Path(scaler_save_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.train_dataset.get_scaler(), scaler_path)
    print(f">>> Scaler Saved ({len(feature_cols)} features) to {scaler_path}")

    return model, data


def main():
    train_model(epochs=150, batch_size=512)


if __name__ == "__main__":
    main()
