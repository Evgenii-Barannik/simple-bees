import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from preprocessing import get_dataset

def get_folded_spectra(ds, sensor_id, start, end):
    filtered = ds.sel(sensor=sensor_id).where(
            ds.sel(sensor=sensor_id)['base'].notnull() &
            (ds['timestamp'] >= start) &
            (ds['timestamp'] <= end),
            drop=True
            )
    hour_bins = pd.to_datetime(filtered['timestamp'].values).hour
    filtered = filtered.assign_coords(hour_bin=("timestamp", hour_bins))
    folded_spectra = filtered.groupby('hour_bin').mean(dim='timestamp')['spectrum']
    return folded_spectra

def create_gif():
    start = pd.Timestamp('2024-08-10 00:00:00', tz="UTC+03:00")
    end = pd.Timestamp('2024-08-19 01:00:00', tz="UTC+03:00")
    ds = get_dataset(csv_input_folder="full_data", timestamps_as_floats=False)

    folded_spectra_20 = get_folded_spectra(ds, 20, start, end)
    mean_20 = folded_spectra_20.mean(dim='hour_bin')
    folded_spectra_21 = get_folded_spectra(ds, 21, start, end)
    mean_21 = folded_spectra_21.mean(dim='hour_bin')

    start_time = "Start time: " + str(start)
    end_time =   "End time:   " + str(end)
    text = start_time + "\n" + end_time

    frames = []
    for hour in folded_spectra_20['hour_bin'].values:
        plt.figure(figsize=(8, 5))
        plt.plot(folded_spectra_20.sel(hour_bin=hour).values, color='blue', linewidth=1, label=f'Sensor 20, time bin {hour}:00')
        plt.plot(folded_spectra_21.sel(hour_bin=hour).values, color='orange', linewidth=1, label=f'Sensor 21, time bin {hour}:00')
        plt.xlabel('Channel')
        plt.ylabel('Amplitude')
        plt.title(f'Spectra for different times of the day (phase folded)')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 1860)
        plt.tight_layout()
        plt.figtext(0.65, 0.8, text, ha='left', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        filename = f"temp_frame_{hour}.png"
        plt.savefig(filename)
        plt.close()

        frames.append(imageio.imread(filename))
        os.remove(filename)

    plt.figure(figsize=(8, 5))
    plt.plot(mean_20.values, color='blue', linewidth=2, label='Sensor 20, mean')
    plt.plot(mean_21.values, color='orange', linewidth=2, label='Sensor 21, mean')
    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.title('Mean spectra')
    plt.legend()
    plt.xlim(0, 1860)
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.65, 0.8, text, ha='left', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    final_filename = "temp_final_frame.png"
    plt.savefig(final_filename)
    plt.close()

    frames.append(imageio.imread(final_filename))
    os.remove(final_filename)
    gif_filename = "spectra_comparison.gif"
    imageio.mimsave(gif_filename, frames, duration=1)  # duration=1 second per frame

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {gif_filename}')

if __name__ == "__main__":
    create_gif()
