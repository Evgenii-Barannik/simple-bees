import pandas as pd

from spectra_correlation import plot_correlations
from spectra_folding import plot_mean_spectra
from preprocessing import download_csv_files, load_dataset

if __name__ == "__main__":
    download_csv_files() 
    ds = load_dataset(csv_input_folder="weekly_data", timestamps_as_floats=False)
    now =  pd.to_datetime('now').replace(microsecond=0).tz_localize('Europe/Helsinki')
    one_week_ago = now - pd.Timedelta(weeks=1)
    plot_mean_spectra(ds, one_week_ago, now);
    for sensor in [20, 21, 46]:
        plot_correlations(ds, one_week_ago, now, sensor)
