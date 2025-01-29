import glob
import ntpath
import os
from datetime import date

import geopandas as gpd
import numpy as np
import rasterio
# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
from utils import get_polygon_from_shapefile as get_polygon
import utils
from utils import get_data_frame


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
    return np.stack(bands_and_features, axis=-1)

def get_data_stack(file_paths):
    bands_and_features = []
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            bands_and_features.append(src.read(1))
    return np.stack(bands_and_features)

def crop_shape_file(shape_file, geojson_file):
    shapefile = gpd.read_file(shape_file)
    geojson = gpd.read_file(geojson_file)
    cropped_shapefile = gpd.overlay(shapefile, geojson, how='intersection')
    cropped_shapefile.to_file('cropped_shapefile.shp')

def stack_bands_together(input_dir):
    band_rasters = glob.glob(f"{input_dir}/aligned/*.tiff")
    print(band_rasters)
    print(len(band_rasters))
    with rasterio.open(band_rasters[0]) as ds:
        metadata = ds.profile
        metadata.update({"count": len(band_rasters)})
        stacked_dir = f"{input_dir}/stacked"
        os.makedirs(stacked_dir, exist_ok=True)
        out_file = f"{stacked_dir}/input_stack.tiff"
        with rasterio.open(out_file, 'w', **metadata) as dest:
            for i, band_file in enumerate(band_rasters):
                band_data = rasterio.open(band_file)
                dest.write(band_data.read(1),i+1)
        print(f"Stacked file created {out_file}")

def get_features1(input_files):
    image_data_list = []
    for input_file in input_files:
        img_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
        for b in range(img.shape[2]):
            img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
        print(img.shape)
        image_data_list.append(img)
    features = np.array(image_data_list)
    print(f"features shape: {features.shape}")
    return features

def get_features(input_file):
    img_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    features = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(features.shape[2]):
        features[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    print(features.shape)
    return features

def get_input_labels(shapefile_path, ground_truth, _polygon_path):
    gdf = gpd.read_file(shapefile_path)
    print(f'get_input_labels|shapefile shape:{gdf.shape}')
    print(f'get_input_labels|gdf columns:{gdf.columns.values}')
    df = get_data_frame(ground_truth)
    print(f'get_input_labels|ground_truth shape:{df.shape}')
    print(f'get_input_labels|df columns:{df.columns.values}')
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs="EPSG:4326")
    joined_df = gpd.sjoin(gdf_points, gdf, how='left', predicate='within')
    polygon = get_polygon()
    for i, row in joined_df.iterrows():
        point = Point(row['lat'], row['lon'])
        if not polygon.contains(point):
            joined_df.at[i, 'CODE_18'] = 999
    return joined_df[["lat", "lon", "value", "CODE_18"]]


def get_input_labels3(shapefile_path, ground_truth, polygon_path):
    gdf = gpd.read_file(shapefile_path)
    print(f'get_input_labels|shapefile shape:{gdf.shape}')
    print(f'get_input_labels|gdf columns:{gdf.columns.values}')
    df = get_data_frame(ground_truth)
    print(f'get_input_labels|ground_truth shape:{df.shape}')
    print(f'get_input_labels|df columns:{df.columns.values}')
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs="EPSG:4326")
    joined_df = gdf_points.sjoin(gdf, how='left', predicate='contains')
    polygon = utils.get_polygon(path = polygon_path)
    for i, row in joined_df.iterrows():
        point = Point(row['lat'], row['lon'])
        if not polygon.contains(point):
            joined_df.at[i, 'CODE_18'] = 999
    return joined_df[["lat", "lon", "value", "CODE_18"]]

def get_input_labels2(shapefile_path, ground_truth, polygon_path):
    gdf = gpd.read_file(shapefile_path)
    print(f'get_input_labels|shapefile shape:{gdf.shape}')
    print(f'get_input_labels|gdf columns:{gdf.columns.values}')
    df = get_data_frame(ground_truth)
    print(f'get_input_labels|ground_truth shape:{df.shape}')
    print(f'get_input_labels|df columns:{df.columns.values}')
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs="EPSG:4326")
    joined_df = gdf_points.sjoin(gdf, how='left', predicate='contains')
    joined_df.dropna(axis=0)
    return joined_df[["lat", "lon", "value", "CODE_18"]]

def get_input_labels2(shapefile_path, ground_truth, polygon_path):
    gdf = gpd.read_file(shapefile_path)
    print(f'get_input_labels|shapefile shape:{gdf.shape}')
    print(f'get_input_labels|gdf columns:{gdf.columns.values}')
    df = get_data_frame(ground_truth)
    print(f'get_input_labels|ground_truth shape:{df.shape}')
    print(f'get_input_labels|df columns:{df.columns.values}')
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs="EPSG:4326")
    joined_df = gdf_points.sjoin(gdf, how='left', predicate='intersects')
    polygon = utils.get_polygon(path = polygon_path)
    for i, row in joined_df.iterrows():
        point = Point(row['lat'], row['lon'])
        if not polygon.contains(point):
            joined_df.at[i, 'CODE_18'] = 999
    return joined_df[["lat", "lon", "value", "CODE_18"]]

def generate_labels(download_dir, shapefile_path, ground_truth, geojson_path):
    input_labels = get_input_labels(shapefile_path, ground_truth, geojson_path)
    print(f'generate_labels|input_labels shape:{input_labels.shape}')
    print(f'generate_labels|input_labels columns:{input_labels.columns.values}')
    label_path = f"{download_dir}/selected_area_labels.csv"
    input_labels.to_csv(label_path)
    print(f'generate_labels|file {label_path} created')

def get_labels(download_dir, shapefile_path, ground_truth):
    gdf = gpd.read_file(shapefile_path)
    print(f'get_labels|shapefile shape:{gdf.shape}')
    print(f'get_labels|gdf columns:{gdf.columns.values}')
    df = get_data_frame(ground_truth)
    print(f'get_labels|ground_truth shape:{df.shape}')
    print(f'get_labels|df columns:{df.columns.values}')
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs="EPSG:4326")
    joined_df = gpd.sjoin(gdf_points, gdf, how='left', predicate='within')
    joined_df.to_csv(f"{download_dir}/joined_df.csv")
    polygon = get_polygon()
    valid = 0
    invalid = 0
    for i, row in joined_df.iterrows():
        point = Point(row['lat'], row['lon'])
        if not polygon.contains(point):
            invalid+=1
            joined_df.at[i, 'code_2018'] = 999
        else:
            valid+=1
    print(f'get_labels|total:{valid+invalid}|invalid count:{invalid}|valid count:{valid}')
    joined_df.dropna(subset=['code_2018'], inplace=True)
    joined_df[["lat", "lon", "value", "code_2018"]].to_csv(f"{download_dir}/joined_filtered_df.csv")
    label_path = f"{download_dir}/selected_area_labels.csv"
    input_labels = joined_df[["lat", "lon", "value", "code_2018"]]
    input_labels.to_csv(label_path)
    print(f'get_labels|file {label_path} created')

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    shapefile_path = "data/land_cover/urban_atlas/UrbanAtlasBBox.shp"
    ground_truth = "data/land_cover/selected/area_reference.tiff"
    get_labels(download_dir, shapefile_path, ground_truth)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today = date.today()
#     today_string = today.strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     input_files = get_input_files(download_dir)
#     print(f"input_files : {input_files}")
#     get_features(input_files)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today = date.today()
#     today_string = today.strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     input_file = f"{download_dir}/stacked/input_stack.tiff"
#     get_features(input_file)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     input_files = glob.glob(f"{download_dir}/aligned/*.tiff")
#     get_data_frame(input_files)

# if __name__ == "__main__":
#     input_shape_file = "data/land_cover/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp"
#     input_geojson_file = "config/cork2.geojson"
#     # crop_shape_file(input_shape_file, input_geojson_file)
#     crop_shape_file = "data/land_cover/cork2/shape_file/cropped_shapefile.shp"
#     read_shape_files(crop_shape_file)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
#     features = ['NDVI', 'NDWI', 'NDBI', 'NDUI', 'NDDI']
#     ground_truth_file = "data/land_cover/cork2/resampled_cropped_raster.tif"
#     view_shape_of_all_files(download_dir, resolution, bands, features, ground_truth_file)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     stack_bands_together(download_dir)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#
#     shapefile_path = "data/land_cover/cop/CLC18_IE_wgs84/CLC18_IE_wgs84.shp"
#     ground_truth = "data/land_cover/selected/area_reference.tiff"
#     geojson_path = "config/smaller_selected_map.geojson"
#     generate_labels(download_dir, shapefile_path, ground_truth, geojson_path)



# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
#     features = ['NDVI', 'NDWI', 'NDBI', 'NDUI', 'NDDI']
#     # get_input_files(download_dir, resolution, bands, features)
#     # ground_truth_file = "data/land_cover/selected/selected_area_raster.tif"
#     ground_truth_file = "data/land_cover/selected/area_reference.tiff"
#     resample_and_align_images(download_dir, resolution, bands, features, ground_truth_file)
#     stack_bands_together(download_dir)
#     input_files = glob.glob(f"{download_dir}/aligned/*.tiff")
#
#     shapefile_path = "data/land_cover/cop/CLC18_IE_wgs84/CLC18_IE_wgs84.shp"
#     ground_truth = "data/land_cover/selected/area_reference.tiff"
#     # geojson_path = "config/smaller_selected_map.geojson"
#     geojson_path = "config/crookstown.geojson"
#     # generate_labels(download_dir, shapefile_path, ground_truth, geojson_path)
#     get_labels(download_dir, shapefile_path, ground_truth)
#
#     for input_file in input_files:
#         with rasterio.open(input_file) as src:
#             image_data = src.read()
#             image_shape = image_data.shape
#             print(f"get_input_files|input_file:{input_file}")
#             print(f"get_input_files|image_shape:{image_shape}")
