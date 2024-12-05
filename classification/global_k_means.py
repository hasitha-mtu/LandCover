from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import rasterio
from datetime import date
import data_preprocessing
import geopandas as gpd
import matplotlib.pyplot as plt
import glob


def global_kmeans(X, max_clusters, max_iter=100, tol=1e-4):
    """
    Global K-Means algorithm implementation.

    Args:
        X (ndarray): Input data of shape (num_samples, num_features).
        max_clusters (int): Maximum number of clusters.
        max_iter (int): Maximum iterations for K-Means refinement.
        tol (float): Tolerance for convergence.

    Returns:
        cluster_labels (ndarray): Cluster assignments for each sample.
        centroids (ndarray): Final cluster centroids.
    """
    n_samples, n_features = X.shape
    centroids = []  # List to store centroids for all clusters

    for k in range(1, max_clusters + 1):
        best_centroid = None
        best_inertia = float('inf')

        # Try every data point as the initial centroid for the new cluster
        for i in range(n_samples):
            candidate_centroids = np.array(centroids + [X[i]])
            cluster_labels, inertia = run_kmeans(X, candidate_centroids, max_iter, tol)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = X[i]

        centroids.append(best_centroid)

    # Final clustering with optimal centroids
    cluster_labels, _ = run_kmeans(X, np.array(centroids), max_iter, tol)
    return cluster_labels, np.array(centroids)

def run_kmeans(X, centroids, max_iter, tol):
    # Check for NaN values
    if np.isnan(X).any():
        raise ValueError("NaN values remain in the input data!")

    """
    Refine centroids using K-Means iterative updates.
    """
    for _ in range(max_iter):
        cluster_labels, _ = pairwise_distances_argmin_min(X, centroids)
        new_centroids = np.array([X[cluster_labels == k].mean(axis=0) for k in range(len(centroids))])

        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    inertia = np.sum((X - centroids[cluster_labels]) ** 2)
    return cluster_labels, inertia

def get_labels(ground_truth_file):
    src = rasterio.open(ground_truth_file)
    return src.read(1)

def get_cluster_count(ground_truth_shape_file):
    gdf = gpd.read_file(ground_truth_shape_file)
    print(f"read_shape_files|columns: {gdf.columns.values}")
    print(f"read_shape_files|gdf head: {gdf.head()}")
    labels = gdf['Code_18'].values
    return len(labels)

def get_normalized_stack(dir_path):
    input_files = glob.glob(f"{dir_path}/aligned/*.tiff")
    print(f"get_normalized_stack|input_files: {input_files}")
    return data_preprocessing.get_normalized_stack(input_files)

def get_data_stack(dir_path):
    input_files = glob.glob(f"{dir_path}/aligned/*.tiff")
    print(f"get_data_stack|input_files: {input_files}")
    return data_preprocessing.get_data_stack(input_files)

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"../data/{collection_name}/{today_string}"
    crop_shape_file = "../data/land_cover/cork2/shape_file/cropped_shapefile.shp"
    ground_truth_file = "../data/land_cover/cork2/resampled_cropped_raster.tif"
    labels = get_labels(ground_truth_file)
    print(f"shape of labels {labels.shape}")
    data_stack = get_data_stack(download_dir)
    # Mask NaN values (if present)
    data_stack = np.nan_to_num(data_stack, nan=0)
    print(f"shape of data_stack {data_stack.shape}")
    print(f"shape of data_stack {data_stack[0].shape}")
    num_clusters = get_cluster_count(crop_shape_file)
    print(f"num_clusters {num_clusters}")

    # Reshape data for clustering
    X = data_stack.reshape(-1, data_stack.shape[2])  # Shape: (num_pixels, num_bands)
    max_clusters = num_clusters  # Adjust this based on the number of land cover types
    cluster_labels, centroids = global_kmeans(X, max_clusters)

    # Reshape cluster labels back to the original image dimensions
    clustered_image = cluster_labels.reshape(data_stack.shape[:2])

    plt.figure(figsize=(10, 8))
    plt.imshow(clustered_image, cmap='tab20')  # Use a categorical colormap
    plt.colorbar(label='Cluster')
    plt.title('Land Cover Classification (Global K-Means)')
    plt.axis('off')
    plt.show()
    # classified_image = train_model(data_stack, labels, num_clusters)
    # src = rasterio.open(ground_truth_file)
    # with rasterio.open('data/land_cover/cork2/classified_image.tif', 'w', **src.meta) as dst:
    #     dst.write(classified_image.astype(rasterio.uint8), 1)
