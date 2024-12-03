# We usually use a set of indices (for example:
# normalized difference vegetation index - NDVI,
# normalized difference urban index - NDUI,
# normalized difference water index - NDWI,
# normalized difference drought index - NDDI, etc.)
# along with the corrected bands data to perform the classification

# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/
# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/
#Spectral Indices: Calculate relevant spectral indices like
# NDVI (Normalized Difference Vegetation Index)
# NDVI = (B08 - B04) / (B08 + B04)

# NDWI (Normalized1 Difference Water Index)
# NDWI = (B03 - B08) / (B03 + B08)

# NDDI (Normalized Difference Drought Index)
# NDDI = (NDVI - NDWI) / (NDVI + NDWI)

# NDUI (Normalized Difference Urban Index
# NDUI = (B11 - B03) / (B11 + B03)

# NDBI (Normalized1 Build Up Index)
# NDBI = (B06 - B05) / (B06 + B05)

# SAVI (Soil-Adjusted Vegetation Index) to highlight specific land cover types.
# SAVI = ((B08 - B04) / (B08 + B04 + L)) * (1.0 + L)
# L = 0.428; // L = soil brightness correction factor could range from (0 -1)

#Textural Features: Extract textural information from the image, such as
# homogeneity
# contrast
# correlation, using techniques like Gray-Level Co-occurrence Matrices (GLCM).

import rasterio
import glob
import os
from utils import get_parent_directories
import matplotlib.pyplot as plt
from datetime import date


def create_feature_indexes1(input_path, output_path, resolution):
    print(f"create_feature_indexes|input_path:{input_path}|output_path:{output_path}")
    b03_file = glob.glob(f"{input_path}/B03_{resolution}m.jp2")
    with rasterio.open(b03_file[0]) as b03_band:
        B03 = b03_band.read(1)
    b04_file = glob.glob(f"{input_path}/B04_{resolution}m.jp2")
    with rasterio.open(b04_file[0]) as b04_band:
        B04 = b04_band.read(1)
    b05_file = glob.glob(f"{input_path}/B05_{resolution}m.jp2")
    with rasterio.open(b05_file[0]) as b05_band:
        B05 = b05_band.read(1)
    b06_file = glob.glob(f"{input_path}/B06_{resolution}m.jp2")
    with rasterio.open(b06_file[0]) as b06_band:
        B06 = b06_band.read(1)
    b08_file = glob.glob(f"{input_path}/B08_{resolution}m.jp2")
    with rasterio.open(b08_file[0]) as b08_band:
        B08 = b08_band.read(1)
    b11_file = glob.glob(f"{input_path}/B11_{resolution}m.jp2")
    with rasterio.open(b11_file[0]) as b11_band:
        B11 = b11_band.read(1)

    print(f"B03 shape: {B03.shape}")
    print(f"B04 shape: {B04.shape}")
    print(f"B05 shape: {B05.shape}")
    print(f"B06 shape: {B06.shape}")
    print(f"B08 shape: {B08.shape}")
    print(f"B11 shape: {B11.shape}")

    ndvi = abs((B08 - B04) / (B08 + B04))
    create_feature_index_image(ndvi, "NDVI", b04_band.crs, b04_band.transform, output_path)

    ndwi = abs((B03 - B08) / (B03 + B08))
    create_feature_index_image(ndwi, "NDWI", b03_band.crs, b03_band.transform, output_path)

    ndbi = abs((B06 - B05) / (B06 + B05))
    create_feature_index_image(ndbi, "NDBI", b05_band.crs, b05_band.transform, output_path)

    # ndui = abs((B11 - B03) / (B11 + B03))
    # create_feature_index_image(ndui, "NDUI", b03_band.crs, b03_band.transform, output_path)

    nddi = abs((ndvi - ndwi) / (ndvi + ndwi))
    create_feature_index_image(nddi, "NDDI", b04_band.crs, b04_band.transform, output_path)
    plot_feature_indexes(ndvi, ndwi, ndbi, nddi)

def create_feature_indexes(input_path, output_path, resolution):
    print(f"create_feature_indexes|input_path:{input_path}|output_path:{output_path}")
    b03_file = glob.glob(f"{input_path}/B03_{resolution}m.tiff")
    with rasterio.open(b03_file[0]) as b03_band:
        B03 = b03_band.read(1)
    b04_file = glob.glob(f"{input_path}/B04_{resolution}m.tiff")
    with rasterio.open(b04_file[0]) as b04_band:
        B04 = b04_band.read(1)
    b05_file = glob.glob(f"{input_path}/B05_{resolution}m.tiff")
    with rasterio.open(b05_file[0]) as b05_band:
        B05 = b05_band.read(1)
    b06_file = glob.glob(f"{input_path}/B06_{resolution}m.tiff")
    with rasterio.open(b06_file[0]) as b06_band:
        B06 = b06_band.read(1)
    b08_file = glob.glob(f"{input_path}/B08_{resolution}m.tiff")
    with rasterio.open(b08_file[0]) as b08_band:
        B08 = b08_band.read(1)
    b11_file = glob.glob(f"{input_path}/B11_{resolution}m.tiff")
    with rasterio.open(b11_file[0]) as b11_band:
        B11 = b11_band.read(1)

    print(f"B03 shape: {B03.shape}")
    print(f"B04 shape: {B04.shape}")
    print(f"B05 shape: {B05.shape}")
    print(f"B06 shape: {B06.shape}")
    print(f"B08 shape: {B08.shape}")
    print(f"B11 shape: {B11.shape}")

    ndvi = abs((B08 - B04) / (B08 + B04))
    create_feature_index_image(ndvi, "NDVI", b04_band.crs, b04_band.transform, output_path)

    ndwi = abs((B03 - B08) / (B03 + B08))
    create_feature_index_image(ndwi, "NDWI", b03_band.crs, b03_band.transform, output_path)

    ndbi = abs((B06 - B05) / (B06 + B05))
    create_feature_index_image(ndbi, "NDBI", b05_band.crs, b05_band.transform, output_path)

    # ndui = abs((B11 - B03) / (B11 + B03))
    # create_feature_index_image(ndui, "NDUI", b03_band.crs, b03_band.transform, output_path)

    nddi = abs((ndvi - ndwi) / (ndvi + ndwi))
    create_feature_index_image(nddi, "NDDI", b04_band.crs, b04_band.transform, output_path)
    plot_feature_indexes(ndvi, ndwi, ndbi, nddi)

def create_feature_index_image(data, feature_name, crs, transform, output_path):
    print(f"{feature_name} : {feature_name}")
    with rasterio.open(f"{output_path}/{feature_name}.tiff", 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def perform_feature_extraction1(input_dir, resolution):
    parent_dirs = get_parent_directories(input_dir)
    for parent_dir in parent_dirs:
        reproject_dir = f"{parent_dir}/reprojected"
        feature_dir = f"{parent_dir}/features"
        os.makedirs(feature_dir, exist_ok=True)
        create_feature_indexes(reproject_dir, feature_dir, resolution)


def perform_feature_extraction(input_dir, resolution):
    roi_dir = f"{input_dir}/roi"
    feature_dir = f"{input_dir}/features"
    os.makedirs(feature_dir, exist_ok=True)
    print(f"perform_feature_extraction|roi_dir:{roi_dir}|feature_dir:{feature_dir}")
    create_feature_indexes(roi_dir, feature_dir, resolution)


def plot_feature_indexes(ndvi, ndwi, ndbi, nddi):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Plot each index
    axes[0, 0].imshow(ndvi, cmap='viridis')
    axes[0, 0].set_title('NDVI')

    axes[0, 1].imshow(ndwi, cmap='Blues')
    axes[0, 1].set_title('NDWI')

    axes[1, 0].imshow(ndbi, cmap='YlGn')
    axes[1, 0].set_title('NDBI')

    axes[1, 1].imshow(nddi, cmap='RdYlGn')
    axes[1, 1].set_title('NDDI')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    perform_feature_extraction(download_dir, resolution)