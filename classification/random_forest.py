import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import date, timedelta
import glob
import rasterio
import rasterio.plot
import data_preprocessing
import matplotlib.pyplot as plt

import utils
from utils import load_cmap
import numpy as np
from pyproj import Transformer
import pandas as pd

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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

def train_model1(labels, data_stack, ground_truth_file, output_file):
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

def train_model(label_df, input_df, _output_file):
    X = input_df.to_numpy()
    y = label_df.to_numpy().ravel()
    print(f"train_model|X:{X.shape}")
    print(f"train_model|y:{y.shape}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,  shuffle=True)
    print('All data include {n} classes: {classes}'.format(n=np.unique(y).size,
                                                                    classes=np.unique(y)))
    print('The training data include {n} classes: {classes}'.format(n=np.unique(y_train).size,
                                                                    classes=np.unique(y_train)))
    print('The training data include {n} classes: {classes}'.format(n=np.unique(y_test).size,
                                                                    classes=np.unique(y_test)))
    # Train model
    clf = RandomForestClassifier(n_estimators=1000, random_state=42, oob_score=True, verbose=1)
    clf.fit(X_train, y_train)
    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=clf.oob_score * 100))
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for b, imp in zip(bands, clf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
    df = pd.DataFrame()
    df['truth'] = y_test
    df['predict'] = clf.predict(X_test)
    print(df['predict'])

    # Cross-tabulate predictions
    print(pd.crosstab(df['truth'], df['predict'], margins=True))

    classified = clf.predict(X).reshape((96, 142))


    cmap, legend = load_cmap(file_path = "../config/color_map.json")
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.legend(**legend)
    rasterio.plot.show(classified, cmap=cmap, ax=ax, title='Land Cover Classification')
    plt.show()


def get_input_files(input_dir):
    file_paths = glob.glob(f"{input_dir}/aligned/*.tiff")
    return file_paths

def get_input_dataframe(file_paths, selected, latlon_crs = 'epsg:4326'):
    df_list = []
    for file_path in file_paths:
        with rasterio.open(file_path) as f:
            zz = f.read(1)
            x = np.linspace(f.bounds.left, f.bounds.right, f.shape[1])
            y = np.linspace(f.bounds.bottom, f.bounds.top, f.shape[0])
        xx, yy = np.meshgrid(x, y)
        df = pd.DataFrame({
            'x': xx.flatten(),
            'y': yy.flatten(),
            'value': zz.flatten(),
        })
        transformer = Transformer.from_crs(f.crs, latlon_crs, always_xy=False)
        df['lat'], df['lon'] = transformer.transform(xx=df.x, yy=df.y)
        df.drop(columns=['x', 'y'], inplace=True)
        df = df[['lat', 'lon', 'value']]
        file_name = utils.file_name_from_path(file_path)
        df = df.rename(columns={"value": file_name})
        df_list.append(df)
    result_df = df_list[0]
    for df in df_list[1:]:
        result_df = pd.merge(result_df, df, how="left", on=["lat", "lon"])
    print(result_df.head())
    print(result_df.shape)
    input_df = result_df[selected]
    return input_df

def get_input_labels1(shapefile_path, ground_truth):
    gdf = gpd.read_file(shapefile_path)
    print(f"get_input_labels|gdf : {gdf}")
    df = get_data_frame(ground_truth)
    df['code'] = np.full(df.shape[0] , 999, dtype=int)
    print(f"get_input_labels|df shape : {df.shape}")
    for i, row in df.iterrows():
        point = Point(row['lat'], row['lon'])
        code = point_contains(point, gdf)
        print(f"{i} |code of {point} is : {code}")
        df.at[i, 'code'] = code
    return df

def get_input_labels(shapefile_path, ground_truth):
    gdf = gpd.read_file(shapefile_path)
    print(f"get_input_labels|gdf : {gdf}")
    df = get_data_frame(ground_truth)
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs="EPSG:4326")
    print(f"get_input_labels|gdf_points : {gdf_points}")
    # Perform spatial join
    # joined_df = gpd.sjoin(gdf_points, gdf, how='inner', predicate='within')
    joined_df = gpd.sjoin(gdf_points, gdf, how='left', predicate='within')
    polygon = utils.get_polygon_from_shapefile(file_path = "../data/land_cover/crookstown/wgs84/crookstown.shp")
    for i, row in joined_df.iterrows():
        point = Point(row['lat'], row['lon'])
        if not polygon.contains(point):
            joined_df.at[i, 'CODE_18'] = 999
    return joined_df[["lat", "lon", "value", "CODE_18"]]



def point_contains(point, gdf):
    code = 999
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        if polygon.contains(point):
            code = row['CODE_18']
    return code

def get_data_frame(file_path, latlon_crs = 'epsg:4326'):
    print(f"get_data_frame|file_path : {file_path}")
    with rasterio.open(file_path) as f:
        zz = f.read(1)
        x = np.linspace(f.bounds.left, f.bounds.right, f.shape[1])
        y = np.linspace(f.bounds.bottom, f.bounds.top, f.shape[0])
        xx, yy = np.meshgrid(x, y)
        df = pd.DataFrame({
            'x': xx.flatten(),
            'y': yy.flatten(),
            'value': zz.flatten(),
        })
        transformer = Transformer.from_crs(f.crs, latlon_crs, always_xy=False)
        df['lat'], df['lon'] = transformer.transform(xx=df.x, yy=df.y)
        df.drop(columns=['x', 'y'], inplace=True)
        df = df[['lat', 'lon', 'value']]
        return df

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    today = date.today() - timedelta(days=1)
    today_string = today.strftime("%Y-%m-%d")
    download_dir = f"../data/{collection_name}/{today_string}"
    input_files = get_input_files(download_dir)
    selected_bands = ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_10m", "B12_10m", "NDBI", "NDDI", "NDUI", "NDVI", "NDWI"]
    # selected_bands = ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_10m", "B12_10m"]
    # selected_bands = ["NDBI", "NDDI", "NDUI", "NDVI", "NDWI"]
    input_df = get_input_dataframe(input_files, selected_bands)
    shapefile_path = "../data/land_cover/cop/CLC18_IE_wgs84/CLC18_IE_wgs84.shp"
    ground_truth = "../data/land_cover/crookstown/raster/cropped_raster.tif"
    input_labels = get_input_labels(shapefile_path, ground_truth)
    input_labels.to_csv("../data/land_cover/crookstown/updated_labels1.csv")
    output_file = "../data/land_cover/crookstown/classified_raster.tif"
    csv_path = "../data/land_cover/crookstown/updated_labels1.csv"
    input_labels = pd.read_csv(csv_path, usecols=["CODE_18"])
    print(input_df.dtypes)
    print(input_labels.dtypes)
    train_model(input_labels, input_df, output_file)


# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"../data/{collection_name}/{today_string}"
#     ground_truth_file = "../data/land_cover/crookstown/raster/cropped_raster.tif"
#     labels = get_labels(ground_truth_file)
#     print(f"shape of labels {labels.shape}")
#     data_stack = get_data_stack(download_dir)
#     data_stack = np.nan_to_num(data_stack, nan=0)
#     print(f"shape of normalized_stack {data_stack.shape}")
#     print(f"shape of normalized_stack {data_stack[0].shape}")
#     output_file = "../data/land_cover/crookstown/classified_raster.tif"
#     train_model(labels, data_stack, output_file)

