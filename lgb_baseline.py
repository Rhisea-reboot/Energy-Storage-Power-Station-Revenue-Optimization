"""
LightGBM 基线：根据边界条件预测节点电价 A
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 路径配置 ====================
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
train_feature_path = os.path.join(ROOT, 'power_price/data/mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(ROOT, 'power_price/data/mengxi_node_price_selected.csv')
output_price_path = os.path.join(ROOT, 'power_price/output/lgb_baseline_price.csv')
output_power_path = os.path.join(ROOT, 'power_price/output/lgb_baseline_output.csv')

# 测试集天数（与 output_demo.csv 对齐）
TEST_DAYS = 59

# 边界条件特征列（与测试集对齐，仅使用预测值列）
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

# ==================== 1. 数据准备 ====================
df_feat = pd.read_csv(train_feature_path)
df_label = pd.read_csv(train_label_path)

# 按 times 内连接对齐
df_all = pd.merge(df_feat, df_label, on='times', how='inner')
df_all['times'] = pd.to_datetime(df_all['times'])
df_all = df_all.sort_values('times').reset_index(drop=True)

# 划分训练集和测试集：最后 TEST_DAYS 天（5664 行）作为测试集
test_rows = TEST_DAYS * 96
df_test = df_all.tail(test_rows).copy().reset_index(drop=True)
df_train = df_all.head(len(df_all) - test_rows).copy().reset_index(drop=True)


# 添加时间特征
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


df_train = add_time_features(df_train)
all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']

X = df_train[all_features].values
y = df_train[target_col].values

# 按时间顺序划分，最后20%做验证
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# ==================== 2. 模型训练 ====================
train_set = lgb.Dataset(X_train, label=y_train, feature_name=all_features)
val_set = lgb.Dataset(X_val, label=y_val, feature_name=all_features, reference=train_set)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'verbose': -1,
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# 验证集评估
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
print(f'\n验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}')

# ==================== 3. 测试集推理 ====================
df_test = add_time_features(df_test)

X_test = df_test[all_features].values
y_test_pred = model.predict(X_test)

# 保存预测电价（每天 96 点，共 TEST_DAYS 天）
df_price = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
df_price.to_csv(output_price_path, index=False)
print(f'推理结果已保存: {output_price_path}, shape={df_price.shape}')


# ==================== 4. 充放电策略生成 ====================
def generate_strategy(price_csv, save_path="output_profit_15min.csv", test_days=TEST_DAYS, target_col='A'):
    """
    根据预测的实时价格确定充放电策略。
    约束：0<=tc<=80, tc+8<=td<=88, 每次连续8个时间点, 功率±1000。
    若最大收益<=0则当天不操作。
    """
    df_price = pd.read_csv(price_csv, parse_dates=['times'])
    df_price = df_price.sort_values('times').reset_index(drop=True)

    results = []
    for day_idx in range(test_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96
        day_df = df_price.iloc[start_idx:end_idx].copy()
        prices = day_df[target_col].values

        best_profit = 0.0
        best_tc = None
        best_td = None

        for tc in range(0, 81):
            for td in range(tc + 8, 89):
                charge_cost = prices[tc:tc + 8].sum() * 1000.0
                discharge_revenue = prices[td:td + 8].sum() * 1000.0
                profit = discharge_revenue - charge_cost
                if profit > best_profit:
                    best_profit = profit
                    best_tc = tc
                    best_td = td

        power = np.zeros(96, dtype=np.float64)
        if best_tc is not None and best_td is not None:
            power[best_tc:best_tc + 8] = -1000.0
            power[best_td:best_td + 8] = 1000.0

        day_out = pd.DataFrame({
            'times': day_df['times'].values,
            '实时价格': 0.0,
            'power': power,
        })
        results.append(day_out)

    df_out = pd.concat(results, ignore_index=True)
    # 确保时间格式为 YYYY-MM-DD HH:MM:SS（无时区后缀）
    df_out['times'] = pd.to_datetime(df_out['times']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f'策略结果已保存: {save_path}, shape={df_out.shape}')


generate_strategy(output_price_path, output_power_path)
