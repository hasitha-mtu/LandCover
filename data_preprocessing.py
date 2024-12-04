from datetime import date
import numpy as np
import glob
import os
import rasterio
import geopandas as gpd
import shapefile

from utils import view_tiff


def get_input_files(dir_path, resolution, selected_bands, selected_features):
    band_files = get_band_files(dir_path, resolution, selected_bands)
    feature_files = get_feature_files(dir_path, selected_features)
    input_files = band_files + feature_files + ["data/land_cover/cork2/resampled_cropped_raster.tif"]
    for input_file in input_files:
        with rasterio.open(input_file) as src:
            image_data = src.read()
            image_shape = image_data.shape
            print(f"get_input_files|input_file:{input_file}")
            print(f"get_input_files|image_shape:{image_shape}")
            # view_tiff(input_file, title=input_file)

def get_band_files(dir_path, resolution, selected_bands):
    band_file_path = f"{dir_path}/roi"
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
    with shapefile.Reader(file_path) as shp:
        print(shp)
        print(shp.fields)
        print(shp.records())


if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    features = ['NDVI', 'NDWI', 'NDBI', 'NDUI', 'NDDI']
    get_input_files(download_dir, resolution, bands, features)
    # shape_file = "data/land_cover/CLC18_IE_ITM/CLC18_IE_ITM.shp"
    # read_shape_files(shape_file)
    # shape_file = "data/land_cover/CLC18_IE/CLC18_IE.shp"
    # read_shape_files(shape_file)
