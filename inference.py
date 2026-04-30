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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[Predictor] GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[Predictor] WARNING: CUDA not available, using CPU")

        self.scaler = joblib.load(scaler_path)

        with open(mask_path, "r") as f:
            self.selected_features = json.load(f)

        self.model = PriceModel(feature_dim=len(self.selected_features))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def _predict_rows(self, features_2d):
        """对二维特征数组逐行预测，返回预测值数组"""
        x_scaled = self.scaler.transform(features_2d)
        # 分批处理以避免 OOM
        batch_size = 2048
        all_preds = []
        for i in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[i:i+batch_size]).float().to(self.device)
            with torch.no_grad():
                pred, _ = self.model(batch)
            all_preds.append(pred.cpu().numpy())
        return np.concatenate(all_preds)

    def predict(self, df_input: pd.DataFrame):
        """预测最后96个时间点（调试用）"""
        features = df_input[self.selected_features].values[-96:]
        preds = self._predict_rows(features)
        conf = np.zeros_like(preds)
        return preds, conf

    def predict_batch(self, df_input: pd.DataFrame, target_dates: list,
                      skip_insufficient: bool = True, target_col: str | None = None):
        """
        对 target_dates 中每一天的96个时间点进行预测。
        """
        predictions_list = []
        confidences_list = []
        timestamps_list = []

        df_input = df_input.copy()
        if df_input.index.tz is not None:
            df_input.index = df_input.index.tz_localize(None)

        for target_date in target_dates:
            target_start = pd.Timestamp(target_date)

            # 找到目标日期在索引中的位置
            same_day = df_input.index[df_input.index.date == target_start.date()]
            if len(same_day) == 0:
                msg = f"目标日期 {target_date.date()} 在输入数据中无对应时间点，跳过"
                if skip_insufficient:
                    print(f"[警告] {msg}")
                    continue
                raise ValueError(msg)

            day_mask = df_input.index.date == target_start.date()
            day_features = df_input.loc[day_mask, self.selected_features].values

            if len(day_features) < 96:
                msg = f"目标日期 {target_date.date()} 仅 {len(day_features)} 点，不足96，跳过"
                if skip_insufficient:
                    print(f"[警告] {msg}")
                    continue
                raise ValueError(msg)

            # 取前96个点
            day_features = day_features[:96]
            pred = self._predict_rows(day_features)

            # 滚动预测：将预测值写回 A 列
            if target_col is not None and target_col in df_input.columns:
                day_idx = df_input.index[day_mask][:96]
                df_input.loc[day_idx, target_col] = pred

            ts = pd.date_range(start=same_day[0], periods=96, freq="15min")
            predictions_list.append(pred)
            confidences_list.append(np.zeros(96))
            timestamps_list.append(ts)

        return predictions_list, confidences_list, timestamps_list


if __name__ == "__main__":
    engine = PricePredictor()
    df_recent = pd.read_csv("power_price/data/aligned_15min_processed.csv",
                            parse_dates=['times'], index_col='times')
    price_pred, confidence = engine.predict(df_recent)
    print(f"预测未来24小时最高电价: {np.max(price_pred):.2f}")

    np.save("power_price/data/pred.npy", price_pred)
    np.save("power_price/data/conf.npy", confidence)
    print("Saved: power_price/data/pred.npy, power_price/data/conf.npy")
