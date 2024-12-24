import pandas as pd
import pytz
from datetime import datetime

from spectra_correlation import plot_correlations
from spectra_folding import plot_mean_spectra
from preprocessing import download_csv_files, load_dataset

if __name__ == "__main__":
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_week_ago = helsinki_now - pd.Timedelta(weeks=1)

    download_csv_files() 
    ds = load_dataset(csv_input_folder="weekly_data", timestamps_as_floats=False)
    print(helsinki_now)
    plot_mean_spectra(ds, helsinki_week_ago, helsinki_now);
    for sensor in [20, 21, 46]:
        plot_correlations(ds, helsinki_week_ago, helsinki_now, sensor)

    with open('output.txt', 'w') as f:
        print(f"Code executed at {helsinki_now}!", file=f)
