import rasterio
import numpy as np
from sklearn.cluster import KMeans
from datetime import date
from data_mosaic import crop_image
from data_resampling import perform_resampling
from utils import get_polygon
import glob
import data_preprocessing
import geopandas as gpd

def get_labels(ground_truth_file):
    src = rasterio.open(ground_truth_file)
    return src.read(1)

def get_normalized_stack(dir_path):
    input_files = glob.glob(f"{dir_path}/aligned/*.tiff")
    print(f"get_normalized_stack|input_files: {input_files}")
    return data_preprocessing.get_normalized_stack(input_files)

def get_data_stack(dir_path):
    input_files = glob.glob(f"{dir_path}/aligned/*.tiff")
    print(f"get_data_stack|input_files: {input_files}")
    return data_preprocessing.get_data_stack(input_files)

def train_model(X, y, num_clusters):
    print(f"train_model|X:{X.shape}")
    print(f"train_model|y:{y.shape}")
    # Create a K-Means model
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    print(f"train_model|X[y != 0]:{X[y != 0]}")
    print(f"train_model|y[y != 0]:{y[y != 0]}")
    # Train the model on the ground truth data
    kmeans.fit(X[y != 0], y[y != 0])

    y_pred = kmeans.predict(X)

    # Reshape the predicted labels to match the image shape
    classified_image = y_pred.reshape((346, 432))

    return classified_image

def get_cluster_count(ground_truth_shape_file):
    gdf = gpd.read_file(ground_truth_shape_file)
    print(f"read_shape_files|columns: {gdf.columns.values}")
    print(f"read_shape_files|gdf head: {gdf.head()}")
    labels = gdf['Code_18'].values
    return len(labels)

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"../data/{collection_name}/{today_string}"
    crop_shape_file = "../data/land_cover/cork2/shape_file/cropped_shapefile.shp"
    ground_truth_file = "../data/land_cover/cork2/resampled_cropped_raster.tif"
    labels = get_labels(ground_truth_file)
    print(f"shape of labels {labels.shape}")
    data_stack = get_normalized_stack(download_dir)
    print(f"shape of data_stack {data_stack.shape}")
    print(f"shape of data_stack {data_stack[0].shape}")
    num_clusters = get_cluster_count(crop_shape_file)
    print(f"num_clusters {num_clusters}")
    classified_image = train_model(data_stack, labels, num_clusters)
    src = rasterio.open(ground_truth_file)
    with rasterio.open('data/land_cover/cork2/classified_image.tif', 'w', **src.meta) as dst:
        dst.write(classified_image.astype(rasterio.uint8), 1)

