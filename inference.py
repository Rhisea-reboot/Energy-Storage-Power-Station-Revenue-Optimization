import torch
import numpy as np
import joblib
import json
import pandas as pd
from model import PriceModel

class PricePredictor:
    def __init__(self, 
                 model_path="power_price/models/best_model.pt", 
                 scaler_path="power_price/data/scaler.pkl",
                 mask_path="power_price/data/feature_mask.json"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(scaler_path)
        
        with open(mask_path, "r") as f:
            self.selected_features = json.load(f)

        self.model = PriceModel(feature_dim=len(self.selected_features))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def predict(self, df_input: pd.DataFrame):
        """
        输入包含所有原始特征的 DataFrame，内部自动完成筛选、排序和标准化
        """
        # 1. 严格按训练时的特征顺序提取
        x_raw = df_input[self.selected_features].values[-672:]
        
        # 2. 标准化 (Scaler 期望 2D 输入)
        x_scaled = self.scaler.transform(x_raw)
        
        # 3. 转换为 Tensor [1, 672, F]
        x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, conf = self.model(x_tensor)

        return pred.cpu().numpy().flatten(), conf.cpu().numpy().flatten()

    def predict_batch(self, df_input: pd.DataFrame, target_dates: list, skip_insufficient: bool = True, target_col: str | None = None):
        """
        对 target_dates 中的每一天进行批量预测。
        每一天取该日 00:00 之前 672 个时间点的数据作为输入（7天×96点）。

        **滚动预测支持**：若提供 target_col（如 'A'），每预测一天后会把预测值
        写回 df_input 的对应位置。这符合"测试期无法获取前一日真实电价"的约束：
        当模型输入中若包含历史目标变量滞后特征时，必须用预测值替代真实值。

        Parameters
        ----------
        df_input : pd.DataFrame
            包含完整特征的 DataFrame，索引为时间戳。
        target_dates : list-like
            目标日期列表（如 pd.date_range('2026-01-01', periods=59, freq='D')）。
        skip_insufficient : bool
            若某日前历史数据不足 672 点，True 则跳过并警告，False 则抛异常。
        target_col : str | None
            目标变量列名（如 'A'）。提供时启用滚动预测，将每日预测结果写回 df_input。

        Returns
        -------
        predictions_list : List[np.ndarray]
        confidences_list : List[np.ndarray]
        timestamps_list : List[pd.DatetimeIndex]
        """
        predictions_list = []
        confidences_list = []
        timestamps_list = []

        df_input = df_input.copy()
        # 统一为无时区，便于日期对齐
        if df_input.index.tz is not None:
            df_input.index = df_input.index.tz_localize(None)

        for target_date in target_dates:
            target_start = pd.Timestamp(target_date)

            # 找到 target_start 在索引中的精确位置
            if target_start not in df_input.index:
                # 找同一天的第一个可用时刻
                same_day = df_input.index[df_input.index.date == target_start.date()]
                if len(same_day) == 0:
                    msg = f"目标日期 {target_date.date()} 在输入数据中无对应时间点，跳过"
                    if skip_insufficient:
                        print(f"[警告] {msg}")
                        continue
                    raise ValueError(msg)
                target_start = same_day[0]

            end_pos = df_input.index.get_loc(target_start)
            start_pos = end_pos - 672

            if start_pos < 0:
                msg = (
                    f"目标日期 {target_date.date()} 前只有 {end_pos} 个历史点，"
                    f"需要 672 个（7天×96点）"
                )
                if skip_insufficient:
                    print(f"[警告] {msg}，跳过")
                    continue
                raise ValueError(msg)

            df_lookback = df_input.iloc[start_pos:end_pos]
            pred, conf = self.predict(df_lookback)

            # 滚动预测：将预测值写回 df_input，供下一天使用
            if target_col is not None and target_col in df_input.columns:
                pred_end = end_pos + 96
                if pred_end <= len(df_input):
                    df_input.iloc[end_pos:pred_end, df_input.columns.get_loc(target_col)] = pred
                else:
                    # 边界保护：若最后一批不足96点，只写回有效长度
                    valid_len = len(df_input) - end_pos
                    df_input.iloc[end_pos:end_pos + valid_len, df_input.columns.get_loc(target_col)] = pred[:valid_len]

            ts = pd.date_range(start=target_start, periods=96, freq="15min")
            predictions_list.append(pred)
            confidences_list.append(conf)
            timestamps_list.append(ts)

        return predictions_list, confidences_list, timestamps_list

if __name__ == "__main__":
    # 模拟角色 C 的调用
    engine = PricePredictor()
    df_recent = pd.read_csv("power_price/data/aligned_15min_full.csv", parse_dates=['times'], index_col='times')
    price_pred, confidence = engine.predict(df_recent)
    print(f"预测未来24小时最高电价: {np.max(price_pred):.2f}")
    
    # 保存为 .npy 供 main.py 调用
    np.save("power_price/data/pred.npy", price_pred)
    np.save("power_price/data/conf.npy", confidence)
    print("Saved: power_price/data/pred.npy, power_price/data/conf.npy")