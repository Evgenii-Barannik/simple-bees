from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from zoneinfo import ZoneInfo
import copy

from preprocessing import *

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

def format_time_to_helsinki(x, _):
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    dt = datetime.fromtimestamp(x, tz=helsinki_tz) # Timestamps are converted to Helsinki timezone
    return dt.strftime('%H:%M %d')

def get_ticks_between(start, end):
    assert start < end
    midnight_before_start = start.replace(hour=0, minute=0, second=0, microsecond=0) 
    midnight_after_end = end.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
    ticks = [copy.deepcopy(midnight_before_start)]
    moving = midnight_before_start
    while (moving < midnight_after_end):
        moving = moving + pd.Timedelta(hours=12)
        ticks.append(moving)
    return ticks

def plot_continuous_correlations(ds, start, end, chosen_sensor):
    assert start < end

    filtered_dataset = ds.sel(sensor=chosen_sensor).where(
            ds.sel(sensor=chosen_sensor)['base'].notnull() &
            (ds['timestamp'] >= start) &
            (ds['timestamp'] <= end),
            drop=True
    )
    print("Filtered dataset:\n", filtered_dataset)

    utc_datetimes = filtered_dataset['timestamp'].values
    assert str(utc_datetimes[0].tzinfo) == "UTC"
    unix_epochs = np.array([t.timestamp() for t in utc_datetimes]) # Unix/Posix epochs (counted from UTC)
    
    # Find edges for 1D Voronoi tesselation
    voronoi_edges = (unix_epochs[:-1] + unix_epochs[1:]) / 2
    leftmost_edge  = unix_epochs[0] - (unix_epochs[1] - unix_epochs[0]) / 2
    rightmost_edge = unix_epochs[-1] + (unix_epochs[-1] - unix_epochs[-2]) / 2
    voronoi_edges_extended = np.concatenate([
        [leftmost_edge],
        voronoi_edges,
        [rightmost_edge]
    ])
    x_edges = voronoi_edges_extended
    y_edges = voronoi_edges_extended

    # Representations of first and last timestamp in Helsinki timezone
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    start_for_printing = "From: " + str(utc_datetimes[0].astimezone(helsinki_tz))
    end_for_printing =   "To: "   + str(utc_datetimes[-1].astimezone(helsinki_tz))

    # # Compute distances, will be used to color Voronoi cells
    spectra = filtered_dataset['spectrum'].values
    pearson_matrix = calculate_pearson_distance(spectra)
    cosine_matrix = calculate_cosine_distance(spectra)
    angular_matrix = calculate_angular_distance(spectra)
    euclidean_matrix = calculate_euclidean_distance(spectra)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colormap = "coolwarm"
    im1 = axes[0, 0].pcolormesh(x_edges, y_edges, pearson_matrix, cmap=colormap, shading="auto", vmin=0, vmax=1)
    axes[0, 0].set_title('Pearson distance')
    plt.colorbar(im1, ax=axes[0, 0]) 

    im2 = axes[0, 1].pcolormesh(x_edges, y_edges, euclidean_matrix, cmap=colormap, shading="auto")
    axes[0, 1].set_title('Euclidean distance')
    plt.colorbar(im2, ax=axes[0, 1]) 

    im3 = axes[1, 0].pcolormesh(x_edges, y_edges, angular_matrix, cmap=colormap, shading="auto", vmin=0, vmax=1)
    axes[1, 0].set_title('Angular distance')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(x_edges, y_edges, cosine_matrix, cmap=colormap, shading="auto", vmin=0, vmax=1)
    axes[1, 1].set_title('Cosine distance')
    plt.colorbar(im4, ax=axes[1, 1])

    # Data is ploted using unix epochs
    # Ticks are set using unix epochs
    # Tick labels show datetimes in Helsinki timezone
    datetimes_for_ticks = get_ticks_between(start, end) # Datetimes in UTC or Helsinki time (depends on input)
    epochs_for_ticks = [x.timestamp() for x in datetimes_for_ticks] # Unich epochs

    # Settings for Axes
    formatter = FuncFormatter(format_time_to_helsinki)
    for ax in axes.flat:
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_xticks(epochs_for_ticks, labels=datetimes_for_ticks)
        ax.set_yticks(epochs_for_ticks, labels=datetimes_for_ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, ha='right')
        for epoch in unix_epochs: # Black dots for timestamps
            ax.scatter(epoch, epoch, color='black', s=1)

    text = "Distance measures for signal from sensor " + str(chosen_sensor) +"\n" + start_for_printing + "\n" + end_for_printing
    fig.text(0.6, 0.935, text, ha='right', fontsize=14)
    plt.tight_layout(pad=4)  
    fig.subplots_adjust(top=0.9)

    # Save plot
    plotname = f"distance-measures-sensor-{chosen_sensor}.png"
    plt.savefig(plotname)
    print(f"Plot {plotname} was created!")
    return plotname

if __name__ == "__main__":
    # Timezone testing
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    utc_tz = ZoneInfo("UTC")
    helsinki_time = datetime(2024, 12, 25, 2, 0, tzinfo=helsinki_tz)
    utc_time = datetime(2024, 12, 25, 0, 0, tzinfo=utc_tz)
    assert helsinki_time.timestamp() == utc_time.timestamp()

    helsinki_start = datetime(2024, 12, 25, 0, 0, tzinfo=helsinki_tz)
    helsinki_end = helsinki_start + pd.Timedelta(days=3)
    
    # # With an assumption that csv files with data already exist:
    ds = load_dataset(csv_input_folder="example_data", timestamps_as_floats=False)
    plotname = plot_continuous_correlations(ds, helsinki_start, helsinki_end, 20)
    show_image(plotname)
