import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo

from preprocessing import * 

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

def plot_mean_spectra(ds, sensors, start, end, plotname):
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    text = ""
    spectra_len = ds['spectrum'][0].shape[1]
    cmap = mpl.colormaps['coolwarm']
    colors = cmap(np.linspace(1, 0, len(sensors)))
    for (i, sensor) in enumerate(sensors):
        filtered_dataset = filter_and_add_bins(ds, sensor, start, end) 
        mean_spectrum = filtered_dataset.groupby('hour_bin').mean(dim='timestamp')['spectrum'].mean(dim='hour_bin')
        start_time = "From: " + str(filtered_dataset['timestamp'].values[0].astimezone(helsinki_tz))
        end_time =   "To: " + str(filtered_dataset['timestamp'].values[-1].astimezone(helsinki_tz))
        plt.plot(mean_spectrum.values, linewidth=2, color=colors[i], label=f'Sensor {sensor}, averaged')
        if sensor == 20: # Detailed info for sensor 20 only
            text = "For sensor " + str(sensor) +" averaged\n" + start_time + "\n" + end_time +  "\n\n"

    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.title('Acoustic spectra of beehives')
    plt.legend()
    plt.xlim(0, spectra_len)
    plt.ylim(0, None)
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.95, 0.7, text, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.savefig(plotname)
    plt.close()
    return plotname 

if __name__ == "__main__":
    sensors = [20, 21, 46, 109]
    data_dir = "data"
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_days_ago = helsinki_now - pd.Timedelta(days=3)

    files = download_files_if_needed(sensors, helsinki_days_ago, helsinki_now, data_dir) 
    dataset = load_dataset(files)
    mean_spectra_days = plot_mean_spectra(dataset, sensors, helsinki_days_ago, helsinki_now, "mean-spectra-days.png")
    show_image(mean_spectra_days)
