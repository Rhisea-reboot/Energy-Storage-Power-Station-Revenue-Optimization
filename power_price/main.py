"""
main.py
=======
Day 15 端到端主入口 — 从预测输入到提交 CSV 的全流程管线。

端到端流程:
    1. 加载配置 (paths, risk_params)
    2. 获取 prediction[96] + confidence[96]（调用成员B的 inference 模块）
    3. 多模型集成（若提供多模型输入）
    4. 风险管理修正
    5. 调用策略模块获取最优 (tc, td)
    6. 生成 power[96] 序列
    7. 格式化为提交 CSV

命令行用法:
    python main.py --prediction data/pred.npy --confidence data/conf.npy --output output/output.csv
    python main.py --batch --prediction-dir data/preds/ --output output/output.csv

接口函数:
    run_single_day(prediction, confidence, timestamp_index, config) -> pd.DataFrame
    run_batch(predictions_list, confidences_list, timestamps_list, config) -> pd.DataFrame
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:
    yaml = None

# 内部模块导入
from strategy_wrapper import optimize_charge_discharge
from risk_mgmt import apply_risk_management
from ensemble_strategy import ensemble_predictions
from monte_carlo_validator import monte_carlo_validate

# 端到端推理支持（从原始特征 CSV 直接出结果）
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference import PricePredictor
from prepare_test_features import prepare_test_features


def _default_config() -> Dict[str, Any]:
    """默认配置（当 risk_config.yaml 不存在时回退使用）。"""
    return {
        "risk_aversion": 0.1,
        "z_score": 1.28,
        "fixed_profit_threshold": None,
        "negative_threshold": 0.0,
        "spike_threshold": 800.0,
        "n_monte_carlo_scenarios": 100,
        "use_monte_carlo": False,
        "ensemble_method": "simple_average",
        "power_mw": 1000,
        "duration_slots": 8,
    }


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """加载 YAML 配置，若失败则返回默认配置。"""
    if config_path is None:
        candidate = Path(__file__).parent / "risk_config.yaml"
        if candidate.exists():
            config_path = str(candidate)

    if yaml is not None and config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg:
            return cfg
    return _default_config()


def generate_power_profile(
    charge_start: int | None,
    discharge_start: int | None,
    power_mw: int = 1000,
    duration_slots: int = 8,
) -> np.ndarray:
    """
    根据最优 (tc, td) 生成 96 点功率计划。

    Returns
    -------
    np.ndarray [96] int32
        充电时段=-power_mw, 放电时段=+power_mw, 其余=0。
    """
    power = np.zeros(96, dtype=np.int32)
    if charge_start is not None and discharge_start is not None:
        power[charge_start : charge_start + duration_slots] = -power_mw
        power[discharge_start : discharge_start + duration_slots] = power_mw
    return power


def run_single_day(
    prediction: np.ndarray,
    confidence: np.ndarray | None = None,
    timestamp_index: pd.DatetimeIndex | None = None,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    单日端到端策略生成。

    Parameters
    ----------
    prediction : np.ndarray [96]
        次日 00:00-23:45 的电价预测值（15分钟分辨率）。
    confidence : np.ndarray | None, optional
        预测置信度 [96]。
    timestamp_index : pd.DatetimeIndex | None, optional
        北京时间时间戳，freq='15min', tz='Asia/Shanghai'。
        若 None，则自动生成次日 00:00 开始的索引。
    config : dict | None, optional
        运行配置，默认加载 risk_config.yaml。

    Returns
    -------
    pd.DataFrame
        提交格式表格，列: times, 实时价格(预测), power
    """
    if config is None:
        config = load_config()

    # 若未提供 confidence，使用默认
    if confidence is None:
        confidence = np.full(96, prediction.std() * 0.1)

    # 1) 风险管理修正
    risk_result = apply_risk_management(prediction, confidence, config)
    adjusted_prediction = risk_result["adjusted_prediction"]
    profit_threshold = risk_result["profit_threshold"]
    risk_aversion = risk_result["risk_aversion"]

    # 2) 策略搜索（使用风险修正后的预测）
    strategy = optimize_charge_discharge(
        prices=adjusted_prediction,
        confidence=confidence,
        risk_aversion=risk_aversion,
        profit_threshold=profit_threshold,
    )

    # 3) 可选：Monte-Carlo 鲁棒性验证
    if config.get("use_monte_carlo", False) and strategy["execute"]:
        mc_result = monte_carlo_validate(
            prediction=prediction,
            confidence=confidence,
            candidate_strategies=[
                (strategy["charge_start"], strategy["discharge_start"])
            ],
            n_scenarios=config.get("n_monte_carlo_scenarios", 100),
        )
        if mc_result["best_tc"] < 0:
            # 鲁棒性不通过，放弃交易
            strategy["execute"] = False
            strategy["charge_start"] = None
            strategy["discharge_start"] = None

    # 4) 生成功率计划
    power = generate_power_profile(
        strategy["charge_start"],
        strategy["discharge_start"],
        power_mw=config.get("power_mw", 1000),
        duration_slots=config.get("duration_slots", 8),
    )

    # 5) 构造时间索引
    if timestamp_index is None:
        timestamp_index = pd.date_range(
            start="2024-01-01 00:00",
            periods=96,
            freq="15min",
        )
    # 去除时区信息，确保 CSV 输出格式为 YYYY-MM-DD HH:MM:SS（与提交示例一致）
    if timestamp_index.tz is not None:
        timestamp_index = pd.to_datetime(timestamp_index.strftime('%Y-%m-%d %H:%M:%S'))

    # 6) 组装 DataFrame
    df = pd.DataFrame(
        {
            "times": timestamp_index,
            "实时价格": prediction.astype(float),
            "power": power.astype(float),
        }
    )
    return df


def run_batch(
    predictions_list: List[np.ndarray],
    confidences_list: List[np.ndarray] | None = None,
    timestamps_list: List[pd.DatetimeIndex] | None = None,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    批量处理 D 天，生成 96×D 行提交大表。

    Parameters
    ----------
    predictions_list : List[np.ndarray]
        D 天的预测序列，每个 (96,)。
    confidences_list : List[np.ndarray] | None
        D 天的置信度序列。
    timestamps_list : List[pd.DatetimeIndex] | None
        D 天的时间索引列表。
    config : dict | None
        运行配置。

    Returns
    -------
    pd.DataFrame
        垂直拼接后的提交表格，维度 (96*D, 3)。
    """
    D = len(predictions_list)
    if confidences_list is None:
        confidences_list = [None] * D
    if timestamps_list is None:
        timestamps_list = [None] * D

    frames: List[pd.DataFrame] = []
    for i in range(D):
        df = run_single_day(
            prediction=predictions_list[i],
            confidence=confidences_list[i],
            timestamp_index=timestamps_list[i],
            config=config,
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="储能充放电策略端到端生成器"
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="单日预测 .npy 文件路径，形状 (96,)",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        help="批量预测目录，内含 D 个 .npy 文件",
    )
    parser.add_argument(
        "--confidence",
        type=str,
        default=None,
        help="单日置信度 .npy 文件路径，形状 (96,)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/output.csv",
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="风险参数 YAML 配置文件路径",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="启用批量模式",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="原始 test 特征 CSV 路径（如 test_in_feature_ori.csv）。提供此参数时，自动完成特征工程→模型预测→策略生成",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- 端到端模式：从原始特征 CSV 直接生成策略 --------------------------
    if args.input_csv:
        print("=" * 60)
        print("端到端模式：从原始特征 CSV 生成策略")
        print("=" * 60)

        # 1. 特征工程
        df_features = prepare_test_features(test_csv=args.input_csv)

        # 2. 确定需要预测的目标日期（test 期间每一天）
        test_start = df_features.index[df_features.index >= "2026-01-01"].min()
        test_end = df_features.index.max()
        target_dates = pd.date_range(start=test_start.date(), end=test_end.date(), freq="D")
        print(f"目标预测日期: {len(target_dates)} 天 ({target_dates[0].date()} ~ {target_dates[-1].date()})")

        # 3. 模型预测
        base_dir = Path(__file__).parent
        predictor = PricePredictor(
            model_path=str(base_dir / "models" / "best_model.pt"),
            scaler_path=str(base_dir / "data" / "scaler.pkl"),
            mask_path=str(base_dir / "data" / "feature_mask.json"),
        )
        # 使用滚动预测：测试期无法获取前一日真实电价，
        # 若模型输入依赖历史目标变量，则必须用预测值替代（target_col='A'）
        predictions_list, confidences_list, timestamps_list = predictor.predict_batch(
            df_input=df_features,
            target_dates=target_dates,
            skip_insufficient=True,
            target_col="A",
        )

        if not predictions_list:
            print("错误: 未生成任何预测结果")
            return 1

        print(f"成功生成 {len(predictions_list)} 天预测")

        # 4. 策略生成
        df = run_batch(
            predictions_list=predictions_list,
            confidences_list=confidences_list,
            timestamps_list=timestamps_list,
            config=config,
        )

        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"结果已保存: {args.output}  (形状 {df.shape})")
        return 0

    # ---- 单日/多日模式 -------------------------------------------------
    if not args.batch and args.prediction:
        pred = np.load(args.prediction)
        conf = None
        if args.confidence:
            conf = np.load(args.confidence)

        # 自动检测维度：支持 (96,), (D,96) 或 (D*96,)
        if pred.ndim == 1 and pred.shape[0] == 96:
            df = run_single_day(pred, conf, config=config)
        elif pred.ndim == 2 and pred.shape[1] == 96:
            D = pred.shape[0]
            predictions_list = [pred[i] for i in range(D)]
            confidences_list = [conf[i] if conf is not None else None for i in range(D)] if conf is not None else None
            timestamps_list = [
                pd.date_range(
                    start=pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(days=i),
                    periods=96, freq="15min"
                )
                for i in range(D)
            ]
            df = run_batch(predictions_list, confidences_list, timestamps_list, config=config)
        elif pred.ndim == 1 and pred.shape[0] % 96 == 0:
            D = pred.shape[0] // 96
            pred_2d = pred.reshape(D, 96)
            predictions_list = [pred_2d[i] for i in range(D)]
            confidences_list = None
            if conf is not None and conf.shape[0] % 96 == 0:
                conf_2d = conf.reshape(D, 96)
                confidences_list = [conf_2d[i] for i in range(D)]
            timestamps_list = [
                pd.date_range(
                    start=pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(days=i),
                    periods=96, freq="15min"
                )
                for i in range(D)
            ]
            df = run_batch(predictions_list, confidences_list, timestamps_list, config=config)
        else:
            print(f"错误: prediction 形状 {pred.shape} 不符合要求，应为 (96,)、(D,96) 或 (D*96,)")
            return 1

        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"结果已保存: {args.output}  (形状 {df.shape})")
        return 0

    # ---- 批量模式 -------------------------------------------------
    if args.batch and args.prediction_dir:
        pred_dir = Path(args.prediction_dir)
        pred_files = sorted(pred_dir.glob("*.npy"))
        if not pred_files:
            print(f"错误: 目录 {pred_dir} 中未找到 .npy 文件")
            return 1

        predictions_list = [np.load(f) for f in pred_files]
        D = len(predictions_list)
        # 自动生成连续日期时间戳，从 2026-01-01 开始（与 output_demo.csv 一致）
        timestamps_list = [
            pd.date_range(
                start=pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(days=i),
                periods=96,
                freq="15min",
            )
            for i in range(D)
        ]
        df = run_batch(
            predictions_list,
            config=config,
            timestamps_list=timestamps_list,
        )
        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"批量结果已保存: {args.output}  (形状 {df.shape}, {D} 天)")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
