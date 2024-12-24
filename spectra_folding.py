import pandas as pd
import matplotlib.pyplot as plt
import os
import pytz

def filter_and_add_bins(ds, sensor_id, start, end):
    filtered = ds.sel(sensor=sensor_id).where(
            ds.sel(sensor=sensor_id)['base'].notnull() &
            (ds['timestamp'] >= start) &
            (ds['timestamp'] <= end),
            drop=True
    )
    hour_bins = pd.to_datetime(filtered['timestamp'].values).hour
    filtered = filtered.assign_coords(hour_bin=("timestamp", hour_bins))
    return filtered

def plot_mean_spectra(ds, start, end):
    filtered_dataset_20 = filter_and_add_bins(ds, 20, start, end) 
    mean_spectrum_20 = filtered_dataset_20.groupby('hour_bin').mean(dim='timestamp')['spectrum'].mean(dim='hour_bin')
    filtered_dataset_21 = filter_and_add_bins(ds, 21, start, end) 
    mean_spectrum_21 = filtered_dataset_21.groupby('hour_bin').mean(dim='timestamp')['spectrum'].mean(dim='hour_bin')
    filtered_datset_46 = filter_and_add_bins(ds, 46, start, end) 
    mean_spectrum_46 = filtered_datset_46.groupby('hour_bin').mean(dim='timestamp')['spectrum'].mean(dim='hour_bin')

    helsinki_tz = pytz.timezone('Europe/Helsinki')
    start_time = "From: " + str(filtered_dataset_20['timestamp'].values[0].astimezone(helsinki_tz))
    end_time =   "To    : " + str(filtered_dataset_20['timestamp'].values[-1].astimezone(helsinki_tz))

    text = "For sensor 20:\n" + start_time + "\n" + end_time
    _, ax = plt.subplots(figsize=(8, 5)) 
    plt.plot(mean_spectrum_20.values, color='red', linewidth=2, label='Sensor 20, mean')
    plt.plot(mean_spectrum_21.values, color='orange', linewidth=2, label='Sensor 21, mean')
    plt.plot(mean_spectrum_46.values, color='blue', linewidth=2, label='Sensor 46, mean')

    # Put limit value as a tick
    num_of_channels = mean_spectrum_46.values.shape[0] 
    current_ticks = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, num_of_channels]
    ax.set_xticks(current_ticks)

    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.title('Mean spectra')
    plt.legend()
    plt.xlim(0, num_of_channels)
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.64, 0.7, text, ha='left', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    filename = "mean-spectra.png"
    plt.savefig(filename)
    plt.close()

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {filename}')
