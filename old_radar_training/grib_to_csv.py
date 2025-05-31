import xarray as xr
import pandas as pd

# Open the GRIB2 file using xarray and cfgrib
ds = xr.open_dataset('cropped.grib2', engine='cfgrib')

# Convert to a DataFrame
df = ds.to_dataframe().reset_index()

# Export to CSV
df.to_csv('output.csv', index=False)

print(f"Data has been exported to output.csv")