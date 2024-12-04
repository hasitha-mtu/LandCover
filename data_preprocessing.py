from datetime import date
import numpy as np
import glob
import os

def get_input_files(dir_path, selected_bands, selected_features):
    pass

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

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    features = ['NDVI', 'NDWI', 'NDBI', 'NDDI']
