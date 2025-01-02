import pandas as pd
import numpy as np
import xarray as xr
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
# My CICD
def show_image(plotname):
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {plotname}')

def download_csv_files():
    utc_tz = ZoneInfo('UTC')
    today = datetime.today().astimezone(utc_tz)
    date_to = today.strftime('%Y-%m-%d') 
    date_from = (today - timedelta(days=5)).strftime('%Y-%m-%d') 
    sensors = [20, 21, 46]
    base_url = "http://apiologia.zymologia.fi/export/"

    for sensor in sensors:
        url = f"{base_url}?sensor={sensor}&date_from={date_from}&date_to={date_to}"
        response = requests.get(url)
        
        if response.status_code == 200:
            output_dir = "weekly_data"
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            filename = f"{output_dir}/sensor_{sensor}_data_{date_from}_to_{date_to}.csv"
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Data for sensor {sensor} downloaded and saved as {filename}")
        else:
            print(f"Failed to download data for sensor {sensor}. Status code: {response.status_code}")

def parse_spectrum(spectrum_string):
    list_of_strings = str(spectrum_string).strip('[]').split(';')
    list_of_floats = [float(s) for s in list_of_strings]
    return list_of_floats

def load_dataset(csv_input_folder: str, timestamps_as_floats: bool):
    if not os.path.exists(csv_input_folder):
        raise Exception(f"Folder '{csv_input_folder}' does not exist")
    csv_pathnames = [os.path.join(csv_input_folder, name) for name in os.listdir(csv_input_folder) if name.endswith('.csv')]
    if not csv_pathnames:
        raise Exception(f"Put .csv files in '{csv_input_folder}', it is empty now.")
    print("Reading .csv files: ", csv_pathnames)

    parsed_rows = []
    for pathname in csv_pathnames:
        df_with_raw_spectrum = pd.read_csv(pathname)
        for _, row in df_with_raw_spectrum.iterrows():
            if timestamps_as_floats:
                ts = pd.to_datetime(row["Date"]).timestamp()
            else:
                ts = pd.to_datetime(row["Date"])
            spectrum = parse_spectrum(row['Spectrum'])
            parsed_rows.append({
                "timestamp": ts,
                "sensor": row['Sensor'],
                "sensor_pack": row['Sensor_pack'],
                "base": row['Base'],
                "temperature": row['Temperature'],
                "humidity": row['Humidity'],
                "spectrum": spectrum
            })

    main_dataframe = pd.DataFrame(parsed_rows)

    unique_sensors = main_dataframe['sensor'].unique()
    unique_timestamps = main_dataframe['timestamp'].unique()
    num_epochs = len(unique_timestamps)
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
        epoch_idx = list(unique_timestamps).index(row['timestamp'])
        sensor_idx = list(unique_sensors).index(row['sensor'])

        sensorpack_data[epoch_idx, sensor_idx] = row['sensor_pack']
        base_data[epoch_idx, sensor_idx] = row['base']
        temperature_data[epoch_idx, sensor_idx] = row['temperature']
        humidity_data[epoch_idx, sensor_idx] = row['humidity']
        spectrum_array[epoch_idx, sensor_idx, :len(range(max_channels))] = row['spectrum']

    dataset = xr.Dataset(
        {
            "sensor_pack": (['timestamp', 'sensor'], sensorpack_data),
            "base": (['timestamp', 'sensor'], base_data),
            "temperature": (['timestamp', 'sensor'], temperature_data),
            "humidity": (['timestamp', 'sensor'], humidity_data),
            "spectrum": (['timestamp', 'sensor', 'channel'], spectrum_array)
        },
        coords={
            "timestamp": unique_timestamps,
            "sensor": unique_sensors,
            "channel": np.arange(max_channels)
        }
    )
    return dataset

if __name__ == "__main__":
    dataset = load_dataset("example_data", False)
    print("Memory usage: {:.3f} MB".format(dataset.nbytes / (1024**2)))

    filtered_dataset = dataset['timestamp'].where(
            ~dataset.sel(sensor=21)['base'].isnull(),
            drop=True
    ).head()
    print (filtered_dataset, "\n")

    ## Saving dataframe (requires epochs_as_floats==True)
    # dataset.to_netcdf("dataframe.nc")
