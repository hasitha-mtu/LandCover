from datetime import date
import glob
import os
import geopandas as gpd
import shapefile
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import re
import ntpath
from utils import view_tiff


def get_input_files(dir_path, resolution = 10,
                    selected_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
                    selected_features = ['NDVI', 'NDWI', 'NDBI', 'NDUI', 'NDDI']):
    band_files = get_band_files(dir_path, resolution, selected_bands)
    print(f"get_input_files|band_files:{band_files}")
    feature_files = get_feature_files(dir_path, selected_features)
    print(f"get_input_files|feature_files:{feature_files}")
    input_files = band_files + feature_files
    for input_file in input_files:
        with rasterio.open(input_file) as src:
            image_data = src.read()
            image_shape = image_data.shape
            print(f"get_input_files|input_file:{input_file}")
            print(f"get_input_files|image_shape:{image_shape}")
            # view_tiff(input_file, title=input_file)
    return input_files

def get_band_files(dir_path, resolution, selected_bands):
    print(f"get_band_files|selected_bands:{selected_bands}")
    band_file_path = f"{dir_path}/roi"
    print(f"get_band_files|band_file_path:{band_file_path}")
    file_list = []
    for band in selected_bands:
        band_file = glob.glob(f"{band_file_path}/{band}_{resolution}m.tiff")[0]
        if os.path.isfile(band_file):
            file_list.append(band_file)
    return file_list

def get_feature_files(dir_path, selected_features):
    feature_file_path = f"{dir_path}/features"
    file_list = []
    for feature in selected_features:
        feature_file = glob.glob(f"{feature_file_path}/{feature}.tiff")[0]
        if os.path.isfile(feature_file):
            file_list.append(feature_file)
    return file_list

def read_shape_files(file_path):
    gdf = gpd.read_file(file_path)
    print(f"read_shape_files|columns: {gdf.columns.values}")
    print(f"read_shape_files|gdf head: {gdf.head()}")
    labels = gdf['Code_18'].values
    print(f"read_shape_files|labels:{labels}")

def resample_and_align_images(dir_path, resolution, selected_bands, selected_features, ground_truth_file):
    input_files = get_input_files(dir_path, resolution, selected_bands, selected_features)
    output_dir = f"{dir_path}/aligned"
    os.makedirs(output_dir, exist_ok=True)
    for input_file in input_files:
        _, input_file_name = ntpath.split(input_file)
        output_image = f"{output_dir}/{input_file_name}"
        resample_and_align(input_file, output_image, ground_truth_file)

def view_shape_of_all_files(dir_path, resolution, selected_bands, selected_features, ground_truth_file):
    input_files = get_input_files(dir_path, resolution, selected_bands, selected_features) + [ground_truth_file]
    for input_file in input_files:
        with rasterio.open(input_file) as src:
            image_data = src.read()
            image_shape = image_data.shape
            print(f"get_input_files|input_file:{input_file}")
            print(f"get_input_files|image_shape:{image_shape}")
            # view_tiff(input_file, title=input_file)

def resample_and_align(input_image, output_image, ground_truth):
    print(f"resample_and_align|input_image:{input_image}")
    print(f"resample_and_align|output_image:{output_image}")
    print(f"resample_and_align|ground_truth:{ground_truth}")
    """
    Resamples the input image and aligns it with the ground truth data.

    Args:
        input_image: Path to the input image.
        ground_truth: Path to the ground truth data.
        output_shape: Desired output shape.

    Returns:
        Resampled and aligned input image.
    """

    with rasterio.open(input_image) as src:
        input_data = src.read(1)
        input_transform = src.transform
        input_crs = src.crs

    with rasterio.open(ground_truth) as src:
        # ground_truth_data = src.read(1)
        ground_truth_transform = src.transform
        ground_truth_crs = src.crs
        image_data = src.read()
        image_shape = image_data.shape
        output_shape = image_shape

    # Reproject the input image to match the ground truth's CRS and resolution
    print(f"resample_and_align|output_shape:{output_shape}")
    dst_data, dst_transform = reproject(
        source=input_data,
        destination=np.zeros(output_shape),
        src_transform=input_transform,
        src_crs=input_crs,
        dst_transform=ground_truth_transform,
        dst_crs=ground_truth_crs,
        resampling=Resampling.bilinear
    )
    with rasterio.open(output_image, 'w', **src.meta) as dst:
        dst.write(dst_data)

def get_normalized_stack(file_paths):
    bands_and_features = []
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            bands_and_features.append(src.read(1))
    return np.stack(bands_and_features)

def get_data_stack(file_paths):
    bands_and_features = []
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            bands_and_features.append(src.read(1))
    return np.stack(bands_and_features, axis=-1)

def crop_shape_file(shape_file, geojson_file):
    shapefile = gpd.read_file(shape_file)
    geojson = gpd.read_file(geojson_file)
    cropped_shapefile = gpd.overlay(shapefile, geojson, how='intersection')
    cropped_shapefile.to_file('cropped_shapefile.shp')

# if __name__ == "__main__":
#     input_shape_file = "data/land_cover/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp"
#     input_geojson_file = "config/cork2.geojson"
#     # crop_shape_file(input_shape_file, input_geojson_file)
#     crop_shape_file = "data/land_cover/cork2/shape_file/cropped_shapefile.shp"
#     read_shape_files(crop_shape_file)

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    features = ['NDVI', 'NDWI', 'NDBI', 'NDUI', 'NDDI']
    ground_truth_file = "data/land_cover/cork2/resampled_cropped_raster.tif"
    view_shape_of_all_files(download_dir, resolution, bands, features, ground_truth_file)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
#     features = ['NDVI', 'NDWI', 'NDBI', 'NDUI', 'NDDI']
#     # get_input_files(download_dir, resolution, bands, features)
#     ground_truth_file = "data/land_cover/cork2/resampled_cropped_raster.tif"
#     resample_and_align_images(download_dir, resolution, bands, features, ground_truth_file)
#
#     input_files = glob.glob(f"{download_dir}/aligned/*.tiff")
#     for input_file in input_files:
#         with rasterio.open(input_file) as src:
#             image_data = src.read()
#             image_shape = image_data.shape
#             print(f"get_input_files|input_file:{input_file}")
#             print(f"get_input_files|image_shape:{image_shape}")