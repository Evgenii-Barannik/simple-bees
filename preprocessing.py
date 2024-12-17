import pandas as pd
import numpy as np
import xarray as xr
import os

CSV_INPUT_FOLDER = "example_data"

def parse_spectrum(spectrum_string):
    list_of_strings = str(spectrum_string).strip('[]').split(';')
    list_of_floats = [float(s) for s in list_of_strings]
    return list_of_floats

if not os.path.exists(CSV_INPUT_FOLDER):
    raise Exception(f"Folder '{CSV_INPUT_FOLDER}' does not exist")

csv_pathnames = [os.path.join(CSV_INPUT_FOLDER, name) for name in os.listdir(CSV_INPUT_FOLDER) if name.endswith('.csv')]
if not csv_pathnames:
    raise Exception(f"Put .csv files in '{CSV_INPUT_FOLDER}', it is empty now.")
print("Reading .csv files: ", csv_pathnames)

parsed_rows = []
for pathname in csv_pathnames:
    df_with_raw_spectrum = pd.read_csv(pathname)
    for _, row in df_with_raw_spectrum.iterrows():
        unix_epoch = pd.Timestamp(str(row['Date']))
        spectrum = parse_spectrum(row['Spectrum'])
        parsed_rows.append({
            "unix_epoch": unix_epoch,
            "sensor": row['Sensor'],
            "sensor_pack": row['Sensor_pack'],
            "base": row['Base'],
            "temperature": row['Temperature'],
            "humidity": row['Humidity'],
            "spectrum": spectrum
        })

main_dataframe = pd.DataFrame(parsed_rows)

unique_sensors = main_dataframe['sensor'].unique()
unique_unix_epochs = main_dataframe['unix_epoch'].unique()
num_epochs = len(unique_unix_epochs)
num_sensors = len(unique_sensors)

# Create arrays with NaN for missing data
sensorpack_data = np.full((num_epochs, num_sensors), np.nan) 
base_data = np.full((num_epochs, num_sensors), np.nan)
temperature_data = np.full((num_epochs, num_sensors), np.nan) 
humidity_data = np.full((num_epochs, num_sensors), np.nan) 
max_channels = max(len(spectrum) for spectrum in main_dataframe['spectrum'] if spectrum is not None)
spectrum_array = np.full((num_epochs, num_sensors, max_channels), np.nan)

# Populate arrays 
for _, row in main_dataframe.iterrows():
    epoch_idx = list(unique_unix_epochs).index(row['unix_epoch'])
    sensor_idx = list(unique_sensors).index(row['sensor'])

    sensorpack_data[epoch_idx, sensor_idx] = row['sensor_pack']
    base_data[epoch_idx, sensor_idx] = row['base']
    temperature_data[epoch_idx, sensor_idx] = row['temperature']
    humidity_data[epoch_idx, sensor_idx] = row['humidity']
    spectrum_array[epoch_idx, sensor_idx, :len(range(max_channels))] = row['spectrum']

dataset = xr.Dataset(
    {
        "sensor_pack": (['unix_epoch', 'sensor'], sensorpack_data),
        "base": (['unix_epoch', 'sensor'], base_data),
        "temperature": (['unix_epoch', 'sensor'], temperature_data),
        "humidity": (['unix_epoch', 'sensor'], humidity_data),
        "spectrum": (['unix_epoch', 'sensor', 'channel'], spectrum_array)
    },
    coords={
        "unix_epoch": unique_unix_epochs,
        "sensor": unique_sensors,
        "channel": np.arange(max_channels)
    }
)

print("Dataset contents:")
print(dataset)

print("\nSlice of a spectrum for specific (timestamp, sensor):")
print(dataset.sel(unix_epoch=unique_unix_epochs[0], sensor=20)['spectrum'].values[0:20])
