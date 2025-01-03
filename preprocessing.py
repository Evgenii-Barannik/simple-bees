import pandas as pd
import numpy as np
import xarray as xr
import os
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# My CICD
def show_image(plotname):
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {plotname}')

def download_files_if_needed(sensors, datetime_start, datetime_end, output_dir):
    # Database stores datetimes as UTC, so we transform datetimes to UTC before using GET request
    assert datetime_start <= datetime_end
    utc_tz = ZoneInfo('UTC')
    date_start = datetime_start.astimezone(utc_tz).strftime('%Y-%m-%d') 
    date_end = datetime_end.astimezone(utc_tz).strftime('%Y-%m-%d') 

    downloaded_files = []
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    base_url = "http://apiologia.zymologia.fi/export/"

    for sensor in sensors:
        pathname = f"{output_dir}/sensor_{sensor}_data_{date_start}_to_{date_end}.csv"

        if not os.path.isfile(pathname):
            url = f"{base_url}?sensor={sensor}&date_from={date_start}&date_to={date_end}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(pathname, 'wb') as file:
                    file.write(response.content)
                print(f"Data for sensor {sensor} downloaded and saved as {pathname}")
            else:
                print(f"Failed to download data for sensor {sensor}. Status code: {response.status_code}")

        else:
            print(f"Data for sensor {sensor} already exists at {pathname}")

        downloaded_files.append(pathname)           
    return downloaded_files

def get_max_spectrum_len(files_to_load):
    max_spectrum_len = 0

    for pathname in files_to_load:
        df_with_raw_spectra = pd.read_csv(pathname)
        assert len(df_with_raw_spectra) != 0
        for _, row in df_with_raw_spectra.iterrows():
            list_of_strings = str(row['Spectrum']).strip('[]').split(';')
            spectrum_len = len([float(s) for s in list_of_strings])
            if spectrum_len > max_spectrum_len:
                max_spectrum_len = spectrum_len
    assert max_spectrum_len != 0
    return max_spectrum_len

def parse_and_pad_spectrum(spectrum_string, required_length):
    list_of_strings = str(spectrum_string).strip('[]').split(';')
    list_of_floats = [float(s) for s in list_of_strings]

    assert len(list_of_floats) <= required_length
    if len(list_of_floats) < required_length:
        list_of_floats.extend ([0.0]*(required_length - len(list_of_floats)))
    assert len(list_of_floats) == required_length 
    return list_of_floats

def load_dataset(files_to_load):
    if files_to_load == []:
        raise Exception(f"List of files to load is empty: '{files_to_load}'")
    
    max_spectrum_len = get_max_spectrum_len(files_to_load)
    parsed_rows = []
    for pathname in files_to_load:
        df_with_raw_spectrum = pd.read_csv(pathname)

        for _, row in df_with_raw_spectrum.iterrows():
            datetime = pd.to_datetime(row["Date"])
            spectrum = parse_and_pad_spectrum(row['Spectrum'], max_spectrum_len)
            parsed_rows.append({
                "timestamp": datetime,
                "sensor": row['Sensor'],
                "sensor_pack": row['Sensor_pack'],
                "base": row['Base'],
                "temperature": row['Temperature'],
                "humidity": row['Humidity'],
                "spectrum": spectrum
            })

    main_dataframe = pd.DataFrame(parsed_rows)
    unique_sensors = main_dataframe['sensor'].unique()
    unique_datetimes = main_dataframe['timestamp'].unique()
    num_sensors = len(unique_sensors)
    num_datetimes = len(unique_datetimes)

    # Create arrays with NaN for missing data
    sensorpack_data = np.full((num_datetimes, num_sensors), np.nan) 
    base_data = np.full((num_datetimes, num_sensors), np.nan)
    temperature_data = np.full((num_datetimes, num_sensors), np.nan) 
    humidity_data = np.full((num_datetimes, num_sensors), np.nan) 
    max_channels = max(len(spectrum) for spectrum in main_dataframe['spectrum'] if spectrum is not None)
    spectrum_array = np.full((num_datetimes, num_sensors, max_channels), np.nan)

    # Populate arrays 
    for _, row in main_dataframe.iterrows():
        datetime_idx = list(unique_datetimes).index(row['timestamp'])
        sensor_idx = list(unique_sensors).index(row['sensor'])
        sensorpack_data[datetime_idx, sensor_idx] = row['sensor_pack']
        base_data[datetime_idx, sensor_idx] = row['base']
        temperature_data[datetime_idx, sensor_idx] = row['temperature']
        humidity_data[datetime_idx, sensor_idx] = row['humidity']
        spectrum_array[datetime_idx, sensor_idx, :len(range(max_channels))] = row['spectrum']

    dataset = xr.Dataset(
        {
            "sensor_pack": (['timestamp', 'sensor'], sensorpack_data),
            "base": (['timestamp', 'sensor'], base_data),
            "temperature": (['timestamp', 'sensor'], temperature_data),
            "humidity": (['timestamp', 'sensor'], humidity_data),
            "spectrum": (['timestamp', 'sensor', 'channel'], spectrum_array)
        },
        coords={
            "timestamp": unique_datetimes,
            "sensor": unique_sensors,
            "channel": np.arange(max_channels)
        }
    )
    return dataset

if __name__ == "__main__":
    sensors = [20, 109]
    data_dir = "data"
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)

    files = download_files_if_needed(sensors, helsinki_now, helsinki_now, data_dir)
    dataset = load_dataset(files)

    print (dataset, "\n")
    print("Memory usage: {:.3f} MB".format(dataset.nbytes / (1024**2)))
    print ("Spectra shape: ", dataset['spectrum'][0].shape)
