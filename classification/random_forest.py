import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import date
import glob
import rasterio
import rasterio.plot
import data_preprocessing
import matplotlib.pyplot as plt
from utils import load_cmap
import numpy as np


def get_labels(ground_truth_file):
    src = rasterio.open(ground_truth_file)
    return src.read(1)

def get_normalized_stack(dir_path):
    input_files = glob.glob(f"{dir_path}/aligned/*.tiff")
    print(f"get_normalized_stack|input_files: {input_files}")
    return data_preprocessing.get_normalized_stack(input_files)

def get_data_stack(dir_path):
    input_files = glob.glob(f"{dir_path}/aligned/B*m.tiff")
    print(f"get_data_stack|input_files: {input_files}")
    return data_preprocessing.get_data_stack(input_files)

def train_model(labels, data_stack, output_file):
    # Example: Features = bands, Labels = land cover classes
    X = data_stack.reshape(-1, len(data_stack))  # Flattened bands
    y = labels.flatten()  # Corresponding labels
    print(f"train_model|X:{X.shape}")
    print(f"train_model|y:{y.shape}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    normalized_stack = get_normalized_stack(download_dir)
    classified = clf.predict(X).reshape(normalized_stack.shape[:2])


    cmap, legend = load_cmap(file_path = "../config/color_map.json")
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.legend(**legend)
    rasterio.plot.show(classified, cmap=cmap, ax=ax, title='Land Cover Classification')
    plt.show()

    src = rasterio.open(ground_truth_file)
    with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=classified.shape[0],
            width=classified.shape[1],
            count=1,
            dtype='uint8',
            crs=src.crs,
            transform=src.transform,
    ) as dst:
        dst.write(classified, 1)

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"../data/{collection_name}/{today_string}"
    ground_truth_file = "../data/land_cover/crookstown/raster/cropped_raster.tif"
    labels = get_labels(ground_truth_file)
    print(f"shape of labels {labels.shape}")
    data_stack = get_data_stack(download_dir)
    data_stack = np.nan_to_num(data_stack, nan=0)
    print(f"shape of normalized_stack {data_stack.shape}")
    print(f"shape of normalized_stack {data_stack[0].shape}")
    output_file = "../data/land_cover/crookstown/classified_raster.tif"
    train_model(labels, data_stack, output_file)

