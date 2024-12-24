import pandas as pd
import matplotlib.pyplot as plt
import os
import pytz
from datetime import datetime
from preprocessing import download_csv_files, load_dataset

NUM_OF_CHANNELS = 1848

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

    _, ax = plt.subplots(figsize=(8, 5)) 
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    text = ""
    for (sensor, plotting_color) in [(20, "red"), (21, "orange"), (46, "blue")]: 
        filtered_dataset = filter_and_add_bins(ds, sensor, start, end) 
        mean_spectrum = filtered_dataset.groupby('hour_bin').mean(dim='timestamp')['spectrum'].mean(dim='hour_bin')
        assert (NUM_OF_CHANNELS == mean_spectrum.values.shape[0]) 
        start_time = "From: " + str(filtered_dataset['timestamp'].values[0].astimezone(helsinki_tz))
        end_time =   "To: " + str(filtered_dataset['timestamp'].values[-1].astimezone(helsinki_tz))
        plt.plot(mean_spectrum.values, color=plotting_color, linewidth=2, label=f'Sensor {sensor}, mean')
        if sensor == 20: # Detailed info for sensor 20 only
            text = "Sensor: " + str(sensor) +"\n" + start_time + "\n" + end_time +  "\n\n"

    current_ticks = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, NUM_OF_CHANNELS]
    ax.set_xticks(current_ticks)

    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.title('Acoustic spectra of beehives')
    plt.legend()
    plt.xlim(0, NUM_OF_CHANNELS)
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.95, 0.7, text, ha='right', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    filename = "mean-spectra.png"
    plt.savefig(filename)
    plt.close()

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {filename}')

if __name__ == "__main__":
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_week_ago = helsinki_now - pd.Timedelta(weeks=1)

    # With an assumption that csv files are already dowloaded:
    # download_csv_files() 
    ds = load_dataset(csv_input_folder="weekly_data", timestamps_as_floats=False)
    plot_mean_spectra(ds, helsinki_week_ago, helsinki_now)
