#!/usr/bin/env python3
"""
run_pipeline.py
================
电力现货电价预测与储能策略优化 — 端到端流水线入口

本脚本串联 Achieve.md (Day1-3)、B-Plan.md (Day4-11)、C_Plan.md (Day12-15)
三个阶段计划的实现模块，提供从原始数据到最终策略提交的全流程运行能力。

各阶段也可独立运行，详见 README.md。
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 路径常量
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "power_price" / "data"
MODEL_DIR = ROOT / "power_price" / "models"

WEATHER_RAW = DATA_DIR / "weather_raw"
WEATHER_CSV = DATA_DIR / "weather_features.csv"
PRICE_CSV = DATA_DIR / "mengxi_node_price_selected.csv"
BOUNDARY_CSV = DATA_DIR / "mengxi_boundary_anon_filtered.csv"
ALIGNED_CSV = DATA_DIR / "aligned_15min_full.csv"
SCALER_PATH = DATA_DIR / "scaler.pkl"
FEATURE_MASK = DATA_DIR / "feature_mask.json"
MODEL_WEIGHTS = MODEL_DIR / "best_model.pt"
OUTPUT_CSV = ROOT / "power_price" / "output" / "output.csv"

TARGET_COL = "A"  # 蒙西节点电价列名
LOOKBACK = 672    # 7天历史
HORIZON = 96      # 1天预测


# ==================================================================
# Stage 1: 数据预处理 (Day 1-3) — 对应 Achieve.md
# ==================================================================
def stage1_data_preprocessing(skip_if_exists: bool = True):
    """
    Day 1: 气象数据降维与特征提取 (Test_Weather_processor.py)
    Day 2: 多源数据对齐 (Dataaligning.py)
    Day 3: Dataset 构建验证 (Dataset.py)
    """
    print("\n" + "=" * 60)
    print("Stage 1: 数据预处理 (Day 1-3)")
    print("=" * 60)

    # Day 1: 气象处理
    if skip_if_exists and WEATHER_CSV.exists():
        print(f"[Day 1] 气象特征已存在，跳过: {WEATHER_CSV}")
    else:
        print("[Day 1] 运行气象数据降维...")
        from Test_Weather_processor import WeatherForecastProcessor
        processor = WeatherForecastProcessor()
        processor.process_directory(
            input_dir=str(WEATHER_RAW),
            output_path=str(WEATHER_CSV),
            freq='1h'
        )

    # Day 2: 多源对齐
    if skip_if_exists and ALIGNED_CSV.exists():
        print(f"[Day 2] 对齐数据已存在，跳过: {ALIGNED_CSV}")
    else:
        print("[Day 2] 运行多源数据对齐...")
        from Dataaligning import PowerDataAligner, DataAlignmentConfig
        aligner = PowerDataAligner(
            config=DataAlignmentConfig(),
            price_col=TARGET_COL
        )
        aligner.align_all_data(
            price_path=str(PRICE_CSV),
            boundary_path=str(BOUNDARY_CSV),
            weather_path=str(WEATHER_CSV),
            output_path=str(ALIGNED_CSV)
        )

    # Day 3: 验证 Dataset 可加载
    print("[Day 3] 验证 PyTorch Dataset 构建...")
    from Dataset import PowerPriceDataModule
    # 临时加载确认格式正确
    dm = PowerPriceDataModule(
        csv_path=str(ALIGNED_CSV),
        train_end_date="2025-10-31",
        val_start_date="2025-11-01",
        target_col=TARGET_COL,
        lookback_window=LOOKBACK,
        forecast_horizon=HORIZON,
        batch_size=4
    )
    x, y = next(iter(dm.get_train_loader()))
    print(f"  ✓ Dataset 验证通过: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"  ✓ 特征维度: {dm.get_feature_dim()}")


# ==================================================================
# Stage 2: 特征工程 (Day 4-7) — 对应 B-Plan.md 第一阶段
# ==================================================================
def stage2_feature_engineering(force: bool = False):
    """
    在 aligned_15min_full.csv 上运行特征工程管线，生成 power_price/data/feature_mask.json。
    features.py 提供时间特征、策略特征、滞后特征及 Top-K 筛选。
    """
    print("\n" + "=" * 60)
    print("Stage 2: 特征工程 (Day 4-7)")
    print("=" * 60)

    if not force and FEATURE_MASK.exists():
        print(f"特征掩码已存在，跳过: {FEATURE_MASK}")
        with open(FEATURE_MASK, "r") as f:
            cols = json.load(f)
        print(f"  现有特征数: {len(cols)}")
        return cols

    print("运行特征工程管线...")
    from features import build_feature_pipeline

    df = pd.read_csv(ALIGNED_CSV, parse_dates=['times'], index_col='times')
    df_processed, selected_features = build_feature_pipeline(df, target_col=TARGET_COL)

    # 保存处理后的宽表（可选，供调试查看）
    processed_path = DATA_DIR / "aligned_15min_processed.csv"
    df_processed.to_csv(processed_path)
    print(f"  ✓ 特征工程完成，选定 {len(selected_features)} 维特征")
    print(f"  ✓ 处理后的数据保存至: {processed_path}")
    print(f"  ✓ 特征掩码保存至: {FEATURE_MASK}")
    return selected_features


# ==================================================================
# Stage 3: 模型训练与推理 (Day 8-11) — 对应 B-Plan.md 第二阶段
# ==================================================================
def stage3_train_model(epochs: int = 10, batch_size: int = 32):
    """
    Day 8-11: ResNet-MLP 训练 (train.py) → 保存 best_model.pt
    """
    print("\n" + "=" * 60)
    print("Stage 3: 模型训练 (Day 8-11)")
    print("=" * 60)

    if not FEATURE_MASK.exists():
        raise FileNotFoundError(f"请先运行 Stage 2 生成 {FEATURE_MASK}")

    with open(FEATURE_MASK, "r") as f:
        feature_cols = json.load(f)

    print(f"加载模型: feature_dim={len(feature_cols)}, epochs={epochs}")
    from model import PriceModel
    from Dataset import PowerPriceDataModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PowerPriceDataModule(
        csv_path=str(ALIGNED_CSV),
        train_end_date="2025-10-31",
        val_start_date="2025-11-01",
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        lookback_window=LOOKBACK,
        forecast_horizon=HORIZON,
        batch_size=batch_size
    )
    model = PriceModel(feature_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import torch.nn.functional as F
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in data.get_train_loader():
            x, y = x.to(device), y.to(device)
            pred, conf = model(x)
            price_loss = F.huber_loss(pred, y)
            error = torch.abs(pred.detach() - y)
            conf_loss = F.mse_loss(conf, error)
            loss = price_loss + 0.2 * conf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(data.get_train_loader())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in data.get_val_loader():
                p, _ = model(x.to(device))
                val_loss += F.mse_loss(p, y.to(device)).item()
        val_loss /= len(data.get_val_loader())

        print(f"  Epoch {epoch+1:02d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            print("    >>> Model Saved")

    # 保存 scaler
    import joblib
    joblib.dump(data.train_dataset.get_scaler(), SCALER_PATH)
    print(f"  ✓ Scaler 保存至: {SCALER_PATH}")
    print(f"  ✓ 最佳模型保存至: {MODEL_WEIGHTS}")


def stage3_inference_single_day(df_input: pd.DataFrame) -> tuple:
    """
    使用训练好的模型对单天进行推理。
    返回: (prediction[96], confidence[96])
    """
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"模型权重不存在: {MODEL_WEIGHTS}，请先运行训练")

    from inference import PricePredictor
    predictor = PricePredictor(
        model_path=str(MODEL_WEIGHTS),
        scaler_path=str(SCALER_PATH),
        mask_path=str(FEATURE_MASK)
    )
    pred, conf = predictor.predict(df_input)
    return pred, conf


# ==================================================================
# Stage 4: 策略生成 (Day 12-15) — 对应 C_Plan.md
# ==================================================================
def stage4_generate_strategy(prediction: np.ndarray, confidence: np.ndarray | None = None,
                             timestamp_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    Day 12-15: 调用 power_price/main.py 中的策略管线生成单日充放电计划。
    """
    print("\n" + "=" * 60)
    print("Stage 4: 策略生成 (Day 12-15)")
    print("=" * 60)

    sys.path.insert(0, str(ROOT / "power_price"))
    from main import run_single_day

    df_result = run_single_day(
        prediction=prediction,
        confidence=confidence,
        timestamp_index=timestamp_index
    )
    print(f"  ✓ 策略生成完成，输出维度: {df_result.shape}")
    print(f"  ✓ 充电时段: {df_result[df_result['power'] == -1000]['times'].tolist()[:2]} ...")
    print(f"  ✓ 放电时段: {df_result[df_result['power'] == 1000]['times'].tolist()[:2]} ...")
    return df_result


def stage4_generate_strategy_batch(
    predictions_list: list,
    confidences_list: list | None = None,
    timestamps_list: list | None = None,
) -> pd.DataFrame:
    """
    Day 12-15: 批量生成 D 天的充放电策略，输出 96×D 行提交大表。
    """
    print("\n" + "=" * 60)
    print("Stage 4: 批量策略生成 (Day 12-15)")
    print("=" * 60)

    sys.path.insert(0, str(ROOT / "power_price"))
    from main import run_batch

    df_result = run_batch(
        predictions_list=predictions_list,
        confidences_list=confidences_list,
        timestamps_list=timestamps_list,
    )
    D = len(predictions_list)
    print(f"  ✓ 批量策略生成完成，输出维度: {df_result.shape} ({D} 天)")
    for i in range(min(D, 3)):
        day_df = df_result.iloc[i * 96 : (i + 1) * 96]
        charge_times = day_df[day_df["power"] == -1000]["times"].tolist()[:2]
        discharge_times = day_df[day_df["power"] == 1000]["times"].tolist()[:2]
        if charge_times or discharge_times:
            print(f"    Day {i+1}: 充电 {charge_times}... 放电 {discharge_times}...")
    if D > 3:
        print(f"    ... 共 {D} 天")
    return df_result


# ==================================================================
# 命令行入口
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description="电力现货电价预测与储能策略优化 — 端到端流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整流水线（数据处理 → 特征工程 → 训练 → 策略）
  python run_pipeline.py --all --epochs 10

  # 仅运行数据预处理
  python run_pipeline.py --stage1

  # 仅运行特征工程
  python run_pipeline.py --stage2

  # 仅运行模型训练
  python run_pipeline.py --stage3 --epochs 20

  # 基于已有模型生成单日策略（演示模式，使用随机价格）
  python run_pipeline.py --stage4-demo
        """
    )
    parser.add_argument("--all", action="store_true", help="运行完整流水线")
    parser.add_argument("--stage1", action="store_true", help="阶段1: 数据预处理")
    parser.add_argument("--stage2", action="store_true", help="阶段2: 特征工程")
    parser.add_argument("--stage3", action="store_true", help="阶段3: 模型训练")
    parser.add_argument("--stage4-demo", action="store_true",
                        help="阶段4: 策略生成演示（使用模拟预测数据）")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--force", action="store_true", help="强制重新运行（跳过缓存）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV), help="策略输出CSV路径")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="原始 test 特征 CSV 路径（如 power_price/test_data/test_in_feature_ori.csv）。提供时直接走端到端推理+策略")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 端到端捷径：从原始特征 CSV 直接出策略
    if args.input_csv:
        print("\n" + "=" * 60)
        print("端到端模式：从原始特征 CSV 生成策略")
        print("=" * 60)
        sys.path.insert(0, str(ROOT / "power_price"))
        from main import run_batch, load_config
        from prepare_test_features import prepare_test_features
        from inference import PricePredictor

        df_features = prepare_test_features(test_csv=args.input_csv)
        test_start = df_features.index[df_features.index >= "2026-01-01"].min()
        test_end = df_features.index.max()
        target_dates = pd.date_range(start=test_start.date(), end=test_end.date(), freq="D")
        print(f"目标预测日期: {len(target_dates)} 天")

        predictor = PricePredictor(
            model_path=str(MODEL_WEIGHTS),
            scaler_path=str(SCALER_PATH),
            mask_path=str(FEATURE_MASK)
        )
        # 滚动预测：测试期无法获取前一日真实电价，target_col='A' 确保预测值回写
        preds, confs, tss = predictor.predict_batch(
            df_features, target_dates, skip_insufficient=True, target_col="A"
        )
        df_strategy = run_batch(preds, confs, tss, config=load_config())
        df_strategy.to_csv(args.output, index=False, encoding="utf-8")
        print(f"\n✅ 策略结果已保存: {args.output}  ({len(df_strategy)} 行)")
        return

    run_all = args.all

    # Stage 1
    if run_all or args.stage1:
        stage1_data_preprocessing(skip_if_exists=not args.force)

    # Stage 2
    if run_all or args.stage2:
        stage2_feature_engineering(force=args.force)

    # Stage 3
    if run_all or args.stage3:
        stage3_train_model(epochs=args.epochs, batch_size=args.batch_size)

    # Stage 4 Demo / All
    if run_all or args.stage4_demo:
        D = 59  # 与 output_demo.csv 保持一致（59 天）

        predictions_list = []
        confidences_list = []
        timestamps_list = []

        if MODEL_WEIGHTS.exists():
            print("\n使用已有模型进行批量推理...")
            from inference import PricePredictor
            predictor = PricePredictor(
                model_path=str(MODEL_WEIGHTS),
                scaler_path=str(SCALER_PATH),
                mask_path=str(FEATURE_MASK)
            )
            df_input = pd.read_csv(ALIGNED_CSV, parse_dates=['times'], index_col='times')
            target_dates = pd.date_range(start="2026-01-01", periods=D, freq='D')
            predictions_list, confidences_list, timestamps_list = predictor.predict_batch(
                df_input, target_dates
            )
            actual_D = len(predictions_list)
            if actual_D < D:
                print(f"[警告] 真实推理仅返回 {actual_D} 天，用模拟数据补足至 {D} 天")
                np.random.seed(42)
                for i in range(actual_D, D):
                    pred = np.random.randn(96) * 100 + 300
                    pred[10:18] = 50
                    pred[50:58] = 800
                    predictions_list.append(pred)
                    confidences_list.append(np.full(96, 30.0))
                    timestamps_list.append(
                        pd.date_range(
                            start=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                            periods=96, freq="15min"
                        )
                    )
        else:
            print("\n警告: 模型权重不存在，将使用模拟预测数据进行策略演示...")
            np.random.seed(42)
            for i in range(D):
                pred = np.random.randn(96) * 100 + 300
                pred[10:18] = 50
                pred[50:58] = 800
                predictions_list.append(pred)
                confidences_list.append(np.full(96, 30.0))
                timestamps_list.append(
                    pd.date_range(
                        start=pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(days=i),
                        periods=96, freq="15min"
                    )
                )

        df_strategy = stage4_generate_strategy_batch(
            predictions_list, confidences_list, timestamps_list
        )
        df_strategy.to_csv(args.output, index=False, encoding="utf-8")
        print(f"\n✅ 策略结果已保存: {args.output}  ({len(df_strategy)} 行, {D} 天)")


if __name__ == "__main__":
    main()
