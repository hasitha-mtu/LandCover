import rasterio
import numpy as np
from sklearn.cluster import KMeans

def supervised_kmeans(image_file, ground_truth_file, num_clusters):
    """
    Performs supervised K-Means clustering on a remote sensing image.

    Args:
        image_file (str): Path to the input image file.
        ground_truth_file (str): Path to the ground truth file.
        num_clusters (int): Number of clusters.

    Returns:
        numpy.ndarray: Classified image.
    """

    # Load the image and ground truth data
    with rasterio.open(image_file) as src:
        image_data = src.read()
        image_shape = image_data.shape

    ground_truth = rasterio.open(ground_truth_file).read(1)

    # Reshape the data into a 2D array
    X = image_data.reshape(-1, image_data.shape[-1])
    y = ground_truth.ravel()

    # Create a K-Means model
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)

    # Train the model on the ground truth data
    kmeans.fit(X[y != 0], y[y != 0])

    # Predict the labels for the entire image
    y_pred = kmeans.predict(X)

    # Reshape the predicted labels to match the image shape
    classified_image = y_pred.reshape(image_shape[:2])

    return classified_image

if __name__ == "__main__":
    # Example usage:
    image_file = 'your_image.tif'
    ground_truth_file = 'ground_truth.tif'
    num_clusters = 5

    classified_image = supervised_kmeans(image_file, ground_truth_file, num_clusters)

    # Write the classified image to a new GeoTIFF file
    with rasterio.open('classified_image.tif', 'w', **src.meta) as dst:
        dst.write(classified_image.astype(rasterio.uint8), 1)