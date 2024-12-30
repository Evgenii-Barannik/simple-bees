from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import pytz
import pandas as pd
import pytz
from datetime import datetime
from matplotlib.ticker import FuncFormatter

from preprocessing import load_dataset, download_csv_files

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

def time_formatter(x, _):
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    dt = datetime.fromtimestamp(x, tz=helsinki_tz)
    return dt.strftime('%H:%M %d-%m')

def plot_continuous_correlations(ds, start, end, chosen_sensor):
    filtered_dataset = ds.sel(sensor=chosen_sensor).where(
            ds.sel(sensor=chosen_sensor)['base'].notnull() & # Get existing (timestamp, sensor) pairs
            (ds['timestamp'] >= start) &
            (ds['timestamp'] <= end),
            drop=True
    )
    print(filtered_dataset)
    
    microseconds_timestamps = filtered_dataset['timestamp'].values
    timestamp_converter = lambda t: t.timestamp()
    vfunc = np.vectorize(timestamp_converter)
    microseconds_timestamps = vfunc(microseconds_timestamps)

    voronoi_edges = (microseconds_timestamps[:-1] + microseconds_timestamps[1:]) / 2
    left_edge  = microseconds_timestamps[0] - (microseconds_timestamps[1] - microseconds_timestamps[0]) / 2
    right_edge = microseconds_timestamps[-1] + (microseconds_timestamps[-1] - microseconds_timestamps[-2]) / 2
    voronoi_edges_with_limits = np.concatenate([
        [left_edge],
        voronoi_edges,
        [right_edge]
    ])

    spectra = filtered_dataset['spectrum'].values
    x_edges = voronoi_edges_with_limits
    y_edges = voronoi_edges_with_limits

    # # Create a figure with 4 subplots arranged in a square
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colormap = "coolwarm"

    # # Compute distances, will be used to color rectangles
    pearson_matrix = calculate_pearson_distance(spectra)
    cosine_matrix = calculate_cosine_distance(spectra)
    angular_matrix = calculate_angular_distance(spectra)
    euclidean_matrix = calculate_euclidean_distance(spectra)

    im1 = axes[0, 0].pcolormesh(x_edges, y_edges, pearson_matrix, cmap=colormap, shading="auto")
    axes[0, 0].set_title('Pearson distance')
    plt.colorbar(im1, ax=axes[0, 0]) 

    im2 = axes[0, 1].pcolormesh(x_edges, y_edges, cosine_matrix, cmap=colormap, shading="auto")
    axes[0, 1].set_title('Cosine distance')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].pcolormesh(x_edges, y_edges, angular_matrix, cmap=colormap, shading="auto")
    axes[1, 0].set_title('Angular distance')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(x_edges, y_edges, euclidean_matrix, cmap=colormap, shading="auto")
    axes[1, 1].set_title('Euclidean distance')
    plt.colorbar(im4, ax=axes[1, 1]) 

    formatter = FuncFormatter(time_formatter)
    for ax in axes.flat:
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, ha='right')
        for timestamp in microseconds_timestamps: # Black dots for timestamps
            ax.scatter(timestamp, timestamp, color='black', s=1, label="Timestamps")

    helsinki_tz = pytz.timezone('Europe/Helsinki')
    start_time = "From: " + str(filtered_dataset['timestamp'].values[0].astimezone(helsinki_tz))
    end_time =   "To: " + str(filtered_dataset['timestamp'].values[-1].astimezone(helsinki_tz))
    text = "Distance measures for signal from sensor " + str(chosen_sensor) +"\n" + start_time + "\n" + end_time
    fig.text(0.6, 0.935, text, ha='right', fontsize=14)
    plt.tight_layout(pad=4)  
    fig.subplots_adjust(top=0.9)

    # Save plot
    plotname = f"distance-measures-sensor-{chosen_sensor}.png"
    plt.savefig(plotname)
    print(f"Plot {plotname} was created!")

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {plotname}')

def plot_correlations(ds, start, end, chosen_sensor):
    filtered_dataset = ds.sel(sensor=chosen_sensor).where(
            ds.sel(sensor=chosen_sensor)['base'].notnull() & # Get existing (timestamp, sensor) pairs
            (ds['timestamp'] >= start) &
            (ds['timestamp'] <= end),
            drop=True
    )
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    spectra = filtered_dataset['spectrum'].values

    # Compute distances
    pearson = calculate_pearson_distance(spectra)
    cosine = calculate_cosine_distance(spectra)
    angular = calculate_angular_distance(spectra)
    euclidean = calculate_euclidean_distance(spectra)

    # Create a figure with 4 subplots arranged in a square
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colormap = 'coolwarm'

    axes[0, 0].set_title('Pearson distance')
    sns.heatmap(pearson,ax=axes[0, 0], cmap=colormap, annot=False, square=True)

    axes[0, 1].set_title('Cosine distance')
    sns.heatmap(cosine, ax=axes[0, 1], cmap=colormap, annot=False, square=True)

    axes[1, 0].set_title('Angular distance')
    sns.heatmap(angular, ax=axes[1, 0], cmap=colormap, annot=False, square=True)

    axes[1, 1].set_title('Euclidean distance')
    sns.heatmap(euclidean, ax=axes[1, 1], cmap=colormap, annot=False, square=True)

    start_time = "From: " + str(filtered_dataset['timestamp'].values[0].astimezone(helsinki_tz))
    end_time =   "To: " + str(filtered_dataset['timestamp'].values[-1].astimezone(helsinki_tz))
    text = "Distance matrices for signal from sensor " + str(chosen_sensor) +"\n" + start_time + "\n" + end_time
    fig.text(0.6, 0.935, text, ha='right', fontsize=14)
    plt.tight_layout(pad=4)  

    # Save plot
    plotname = f"distance-matrices-sensor-{chosen_sensor}.png"
    plt.savefig(plotname)
    print(f"Plot {plotname} was created!")

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {plotname}')

if __name__ == "__main__":
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_before = helsinki_now - pd.Timedelta(days=2)

    # With an assumption that csv files are already dowloaded:
    # download_csv_files() 
    ds = load_dataset(csv_input_folder="weekly_data", timestamps_as_floats=False)
    plot_continuous_correlations(ds, helsinki_before, helsinki_now, 20) # Testing for sensor 20
    plot_correlations(ds, helsinki_before, helsinki_now, 20) # Testing for sensor 20

