import xarray as xr

file_path = "power_price/data/weather_raw/20250101.nc"

ds = xr.open_dataset(file_path)
print(ds)
print("\n变量列表：",list(ds.data_vars))



