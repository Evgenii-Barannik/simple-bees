import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime

from spectra_correlation import *
from spectra_folding import * 
from preprocessing import *

if __name__ == "__main__":
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_before = helsinki_now - pd.Timedelta(days=3)

    download_csv_files() 
    ds = load_dataset(csv_input_folder="weekly_data", timestamps_as_floats=False)
    for sensor in [20, 21, 46]:
        plot_continuous_correlations(ds, helsinki_before, helsinki_now, sensor)
    plotname = plot_mean_spectra(ds, helsinki_before, helsinki_now);

    # My CICD
    firefox_app_path = "/Applications/MySoftware/Firefox.app"
    if os.path.exists(firefox_app_path):
        os.system(f'open -g -a {firefox_app_path} index.html')

    print(f"Code executed at {helsinki_now}.")
