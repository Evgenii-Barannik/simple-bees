from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import pytz
import pandas as pd
import pytz
from datetime import datetime

from preprocessing import download_csv_files, load_dataset

def calculate_pearson_distance(spectra):
    distances = pdist(spectra, metric='correlation')
    distance_matrix = squareform(distances)
    return distance_matrix

def calculate_cosine_distance(spectra):
    distances = pdist(spectra, metric='cosine')
    distance_matrix = squareform(distances)
    return distance_matrix

def calculate_angular_distance(spectra):
    inner_products = spectra @ spectra.T
    norms = np.linalg.norm(spectra, axis=1)
    norm_matrix = np.outer(norms, norms)
    cosine_similarity = (inner_products / norm_matrix)
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    angular_distance = np.arccos(cosine_similarity)/np.pi
    return angular_distance

def calculate_euclidean_distance(spectra):
    distances = pdist(spectra, metric='euclidean')
    distance_matrix = squareform(distances)
    return distance_matrix

def plot_correlations(ds, start, end, chosen_sensor):
    filtered_dataset = ds.sel(sensor=chosen_sensor).where(
            ds.sel(sensor=chosen_sensor)['base'].notnull() & # Get existing (timestamp, sensor) pairs
            (ds['timestamp'] >= start) &
            (ds['timestamp'] <= end),
            drop=True
    )
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    spectra = filtered_dataset['spectrum'].values
    print(filtered_dataset, "\n")

    # Compute distances
    pearson = calculate_pearson_distance(spectra)
    cosine = calculate_cosine_distance(spectra)
    angular = calculate_angular_distance(spectra)
    euclidean = calculate_euclidean_distance(spectra)

    # Create a figure with 4 subplots arranged in a square
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colormap = 'coolwarm'

    axes[0, 0].set_title('Pearson distance')
    sns.heatmap(pearson,ax=axes[0, 0], cmap=colormap, annot=False)

    axes[0, 1].set_title('Cosine distance')
    sns.heatmap(cosine, ax=axes[0, 1], cmap=colormap, annot=False)

    axes[1, 0].set_title('Angular distance')
    sns.heatmap(angular, ax=axes[1, 0], cmap=colormap, annot=False)

    axes[1, 1].set_title('Euclidean distance')
    sns.heatmap(euclidean, ax=axes[1, 1], cmap=colormap, annot=False)

    start_time = "From: " + str(filtered_dataset['timestamp'].values[0].astimezone(helsinki_tz))
    end_time =   "To: " + str(filtered_dataset['timestamp'].values[-1].astimezone(helsinki_tz))
    text = "Sensor: " + str(chosen_sensor) +"\n" + start_time + "\n" + end_time
    fig.text(0.6, 0.935, text, ha='right', fontsize=14)
    plt.tight_layout(pad=4)  

    # Save plot
    plotname = f"distances-matrix-sensor-{chosen_sensor}.png"
    plt.savefig(plotname)
    print(f"Plot {plotname} was created!")

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {plotname}')

if __name__ == "__main__":
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_week_ago = helsinki_now - pd.Timedelta(weeks=1)

    # With an assumption that csv files are already dowloaded:
    # download_csv_files() 
    ds = load_dataset(csv_input_folder="weekly_data", timestamps_as_floats=False)
    plot_correlations(ds, helsinki_week_ago, helsinki_now, 20) # Testing for sensor 20
