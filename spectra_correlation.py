from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from datetime import datetime
import pytz

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
    start_time = "From: " + str(filtered_dataset['timestamp'].values[0].astimezone(helsinki_tz))
    end_time =   "To     : " + str(filtered_dataset['timestamp'].values[-1].astimezone(helsinki_tz))

    print(ds, "\n")
    print(filtered_dataset, "\n")
    print(start_time)
    print(end_time)

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

    text = start_time + "\n" + end_time  + "\nSensor: " + str(chosen_sensor)
    fig.text(0.5, 0.935, text, ha='center', fontsize=14)
    plt.tight_layout(pad=4)  

    # Save plot
    plotname = f"distances-matrix-sensor-{chosen_sensor}.png"
    plt.savefig(plotname)
    print(f"Plot {plotname} was created!")

    # My CICD
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {plotname}')
