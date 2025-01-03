import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime

from spectra_correlation import *
from spectra_folding import * 
from preprocessing import *

if __name__ == "__main__":
    sensors = [20, 21, 46, 109]
    data_dir = "data"
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_days_ago = helsinki_now - pd.Timedelta(days=4)
    helsinki_hours_ago = helsinki_now - pd.Timedelta(hours=4)

    files = download_files_if_needed(sensors, helsinki_days_ago, helsinki_now, data_dir) 
    dataset = load_dataset(files)
    for sensor in sensors:
        plot_continuous_correlations(dataset, sensor, helsinki_days_ago, helsinki_now)

    mean_spectra_days = plot_mean_spectra(dataset, sensors, helsinki_days_ago, helsinki_now, "mean-spectra-days.png");
    mean_spectra_hours = plot_mean_spectra(dataset, sensors, helsinki_hours_ago, helsinki_now, "mean-spectra-hours.png");

    # My CICD
    firefox_app_path = "/Applications/MySoftware/Firefox.app"
    if os.path.exists(firefox_app_path):
        os.system(f'open -g -a {firefox_app_path} index.html')

    print(f"Code executed at {helsinki_now}.")
