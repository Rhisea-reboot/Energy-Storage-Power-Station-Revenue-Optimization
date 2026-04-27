import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class WeatherForecastProcessor:

    def __init__(self):
        self.channel_map = {
            'ghi': 0,
            'sp': 1,
            't2m': 2,
            'tcc': 3,
            'tp': 4,
            'u100': 5,
            'v100': 6 
        }
        self.key_vars = ['t2m','ghi','sp','tcc','u100','v100']
    def extract_features(self,file_path):
        ds = xr.open_dataset(file_path)
        data = ds['data']
        base_time = pd.to_datetime(ds.time.values[0])
        results = []
        for lead_idx in range(len(ds.lead_time)):
            lead_hour = int(ds.lead_time.values[lead_idx])
            actual_time = base_time + pd.Timedelta(hours = lead_hour)
            row = {'timestamp': actual_time}
            for var_name in self.key_vars:
                ch_idx = self.channel_map[var_name]
                field = data[0,lead_idx,ch_idx,:,:].values
                row[f'{var_name}_mean'] = np.nanmean(field)
                row[f'{var_name}_max'] = np.nanmax(field)
                row[f'{var_name}_min'] = np.nanmin(field)
                row[f'{var_name}_std'] = np.nanstd(field)
                if var_name == 'u100' and 'v100' in self.key_vars:
                    v_field = data[0,lead_idx,self.channel_map['v100'],:,:].values
                    wind_speed = np.sqrt(field**2 + v_field**2)
                    row['wspd_mean'] = np.nanmean(wind_speed)
                    row['wspd_max'] = np.nanmax(wind_speed)
            results.append(row)
        df = pd.DataFrame(results).set_index('timestamp')
        ds.close()
        return df
    def add_derived_features(self,df):
        df['t2m_mean'] = df['t2m_mean'] - 273.15
        df['t2m_max'] = df['t2m_max'] - 273.15
        df['t2m_min'] = df['t2m_min'] - 273.15
        df['temp_spread'] = df['t2m_max'] - df['t2m_min']
        df['solar_potential'] = df['ghi_mean'] * (1 - df['tcc_mean'])
        df['wind_power_potential'] = df['wspd_mean'] ** 3
        return df
    def process_directory(self,input_dir,output_path,freq = '1h'):
        input_path = Path(input_dir).expanduser()
        output_path = Path(output_path).expanduser()
        nc_files = sorted(list(input_path.glob("*.nc")))
        print(f"发现{len(nc_files)}个预报文件")
        all_data = []
        for i, file in enumerate(nc_files):
            if i%10 == 0:
                print(f"处理：{i+1}/{len(nc_files)}-{file.name}")
            try:
                df = self.extract_features(file)
                all_data.append(df)
            except Exception as e:
                print(f"错误 {file.name}: {e}")
                continue
        if not all_data:
            raise ValueError(f"在 {input_path} 中没有找到可处理的 .nc 文件，请检查路径是否正确。")
        # 合并并排序所有数据
        combined = pd.concat(all_data).sort_index()
        # 添加衍生特征
        combined = self.add_derived_features(combined)
        # 去重并重新索引
        combined = combined[~combined.index.duplicated(keep = 'first')]
        # 插值到指定频率
        if freq and freq != '1h':
            combined = combined.resample(freq).interpolate(method = 'linear')
        combined.to_csv(output_path)
        print(f"\n处理完成!")
        print(f"时间范围：{combined.index[0]} 到 {combined.index[-1]}")
        print(f"总时长：{len(combined)} 小时")
        print(f"特征维度：{len(combined.columns)}维")
        print(f"文件保存至:{output_path}")
        return combined
    
# main

if __name__ == "__main__":
    processor = WeatherForecastProcessor()
    df = processor.process_directory(input_dir = "power_price/data/weather_raw",output_path = "power_price/data/weather_features.csv",freq='1h')
    print("\n预览处理后的数据:")
    print(df.head())
    print("\n特征列表：")
    print(df.columns.tolist())
