import json
import ntpath
import os
from datetime import date

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from rasterio.mask import mask
from sentinelsat import read_geojson
from shapely.ops import unary_union

from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy.ndimage import zoom
from shapely.geometry import shape


def resample_shapefile(input_shapefile, output_shapefile, original_resolution, target_resolution):
    print(f"resample_shapefile|input_shapefile: {input_shapefile}")
    print(f"resample_shapefile|output_shapefile: {output_shapefile}")
    print(f"resample_shapefile|original_resolution: {original_resolution}")
    print(f"resample_shapefile|target_resolution: {target_resolution}")
    gdf = gpd.read_file(input_shapefile)
    print(f"resample_shapefile|gdf: {gdf}")
    # Step 2: Define raster parameters (100m resolution as input)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"resample_shapefile|bounds: {bounds}")
    x_res = 0.01  # Original resolution
    y_res = 0.01
    print(f"resample_shapefile|x_res: {x_res}|y_res: {y_res}")
    print(f"resample_shapefile|width: {(bounds[2] - bounds[0])}")
    print(f"resample_shapefile|height: {(bounds[3] - bounds[1])}")
    width = int((bounds[2] - bounds[0]) / x_res)
    height = int((bounds[3] - bounds[1]) / y_res)
    print(f"resample_shapefile|width: {width}|height: {height}")
    transform = from_origin(bounds[0], bounds[3], x_res, y_res)

    # Step 3: Rasterize the shapefile
    shapes_iter = ((geom, 1) for geom in gdf.geometry)
    raster = rasterize(
        shapes_iter,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Step 4: Upsample the raster to 10m resolution
    upsample_factor = 10  # Factor to upsample from 100m to 10m
    upsampled_raster = zoom(raster, upsample_factor, order=1)  # Bilinear interpolation

    # Step 5: Define new transform for the upsampled raster
    new_x_res = x_res / upsample_factor
    new_y_res = y_res / upsample_factor
    new_transform = from_origin(bounds[0], bounds[3], new_x_res, new_y_res)

    # Step 6: Vectorize the upsampled raster back to polygons
    polygons = []
    values = []
    for geom, value in shapes(upsampled_raster, transform=new_transform):
        if value > 0:  # Ignore background values
            polygons.append(shape(geom))
            values.append(value)

    # Step 7: Create GeoDataFrame for the vectorized polygons
    vectorized_gdf = gpd.GeoDataFrame(
        {"geometry": polygons, "value": values}, crs=gdf.crs
    )

    # Step 8: Save the output to a shapefile
    vectorized_gdf.to_file(output_shapefile)

    print(f"Upsampled shapefile saved to: {output_shapefile}")


def load_cmap(file_path = "config/color_map.json"):
    # Color map is https://collections.sentinel-hub.com/corine-land-cover/readme.html
    lc = json.load(open(file_path))
    lc_df = pd.DataFrame(lc)
    # lc_df["palette"] = "#" + lc_df["palette"]
    values = lc_df["values"].to_list()
    palette = lc_df["palette"].to_list()
    labels = lc_df["label"].to_list()

    # Create colormap from values and palette
    cmap = ListedColormap(palette)

    # Patches legend
    patches = [
        mpatches.Patch(color=palette[i], label=labels[i]) for i in range(len(values))
    ]
    legend = {
        "handles": patches,
        "bbox_to_anchor": (1.05, 1),
        "loc": 2,
        "borderaxespad": 0.0,
    }
    return cmap, legend

def load_cmap_selected(picked, file_path = "config/color_map.json"):
    print(f"load_cmap_selected|picked:{picked}")
    lc = json.load(open(file_path))
    print(f"load_cmap_selected|lc:{lc}")
    lc_df = pd.DataFrame(lc)
    lc_df = lc_df.loc[lc_df['values'].isin(picked)]
    values = lc_df["values"].to_list()
    palette = lc_df["palette"].to_list()
    labels = lc_df["label"].to_list()

    # Create colormap from values and palette
    cmap = ListedColormap(palette)

    # Patches legend
    patches = [
        mpatches.Patch(color=palette[i], label=labels[i]) for i in range(len(values))
    ]
    legend = {
        "handles": patches,
        "bbox_to_anchor": (1.05, 1),
        "loc": 2,
        "borderaxespad": 0.0,
    }
    return cmap, legend

def view_tiff(file_path, title="Land Cover"):
    tiff = rasterio.open(file_path)
    print(f"view_tiff|tiff:{tiff}")
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap, legend = load_cmap()
    print(f"view_tiff|cmap:{cmap}")
    ax.legend(**legend)
    rasterio.plot.show(tiff, cmap=cmap, ax=ax, title=title)
    print(f"view_tiff|cmap:{cmap}")
    plt.show()

def clip_tiff(tiff_path, output_path, roi_polygon):
    print(f"roi_polygon: {roi_polygon}")
    with rasterio.open(tiff_path) as src:
        out_image, out_transform = mask(src, [shape(roi_polygon)], crop=True)
        print(f"out_image.shape: {out_image.shape}")
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_image)


def select_directory_list(directory_path, prefix, depth):
    directory_list = []
    for root, dirs, files in os.walk(directory_path):
        if root[len(directory_path):].count(os.sep) < depth:
            for dir in dirs:
                if prefix:
                    if dir.endswith(prefix):
                        directory_list.append(dir)
                else:
                    directory_list.append(dir)
    directory_list.reverse()
    return directory_list

def get_parent_directories(download_dir):
    parent_list = select_directory_list(download_dir, ".SAFE", 2)
    dir_list = []
    for parent_dir in parent_list:
        parent_dir_path = f"{download_dir}/{parent_dir}"
        dir_list.append(parent_dir_path)
    return dir_list

def get_polygon(path = "config/crookstown.geojson"):
    geojson = read_geojson(path)
    polygon_jsons = geojson["features"]
    polygon_json = polygon_jsons[0]
    geometry_data = polygon_json["geometry"]
    print(f"get_polygon|geometry_data:{geometry_data}")
    polygon = shape(geometry_data)
    return polygon

def read_shape_file(file_path):
    gdf = gpd.read_file(file_path)
    print(gdf)

def read_raster_file(file_path):
    view_tiff(file_path, title="Land Cover")
    with rasterio.open(file_path) as src:
        # Read the raster data
        band1 = src.read(1)
        print(f"read_raster_file|band1:{band1}")
        print(f"read_raster_file|band1 shape:{band1.shape}")
        # Get metadata
        crs = src.crs
        print(f"read_raster_file|crs:{crs}")
        transform = src.transform
        print(f"read_raster_file|transform:{transform}")
        # Print unique values and their counts
        unique_values, counts = np.unique(band1, return_counts=True)
        print(unique_values, counts)

from tifffile import imread

def read_raster(file_path):
    image = imread(file_path)
    print(image.shape)

def make_union_polygon(polygons):
    print()
    """
    This algorithm goes through all polygons and adds them to union_poly only if they're
    not already contained in union_poly.
    (in other words, we're only adding them to union_poly if they can increase the total area)
    """
    union_poly = polygons[0]
    union_parts = [polygons[0], ]
    for p in polygons[1:]:
        common = union_poly.intersection(p)
        if p.area - common.area < 0.001:
            pass
        else:
            union_parts.append(p)
            union_poly = union_poly.union(p)
    return union_parts

def get_min_covering(union_polygons):
    """
    This algorithm computes a minimal covering set of the entire area.
    This means we're going to eliminate some of the images. We do this
    by checking the union of all polygons before and after removing
    each image.
    If by removing the image, the total area is the same, then the image
    can be eliminated since it didn't have any contribution.
    If the area decreases by removing the image, then it can stay.
    """

    whole = unary_union(union_polygons)
    print(f"whole: {whole}")
    L = union_polygons
    V = []
    i = 0
    j = 0
    while j < len(union_polygons):
        without = unary_union(L[:i] + L[i + 1:])
        if whole.area - without.area < 0.001:
            L.pop(i)
        else:
            V.append(union_polygons[j])
            i += 1
        j += 1

        if j % 20 == 0:
            print(i, j, len(L))
    return V

def get_polygon_from_shapefile(file_path = "data/land_cover/crookstown/wgs84/crookstown.shp"):
    gdf = gpd.read_file(file_path)
    union_polygons = make_union_polygon(gdf['geometry'].tolist())
    min_area_polygon = unary_union(union_polygons)
    print(f"get_polygon_from_shapefile|min_area_polygon:{min_area_polygon}")
    return min_area_polygon

def change_coordinate_system(input_path, output_path, coordinate_system):
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(epsg=coordinate_system)
    gdf.to_file(output_path)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def file_name_from_path(path):
    return path_leaf(path).split(".")[0]

def plot_color_map(file_path):
    lc = json.load(open(file_path))
    print(lc)
    lc_df = pd.DataFrame(lc)
    print(lc_df)
    values = lc_df["values"].to_list()
    print(f"values : {values}")
    print(f"values length : {len(values)}")
    palette = lc_df["palette"].to_list()
    print(f"palette : {palette}")
    print(f"palette length : {len(palette)}")
    labels = lc_df["label"].to_list()
    print(f"labels : {labels}")
    print(f"labels length : {len(labels)}")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, len(palette) * 0.2))  # Adjust height based on number of rows
    ax.axis("tight")
    ax.axis("off")

    # Create table data
    table_data = [["Color", "Description"]] + [[value, desc] for value, desc in zip(values, labels)]

    # Create the table
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")

    # Apply color to the cells in the "Color" column
    for i, hex_code in enumerate(palette, start=1):  # Skip header row
        table[(i, 0)].set_facecolor(hex_code)

    # Adjust font size and layout
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width([0, 1])

    # Show the plot
    plt.show()

def plot_color_map_selected(file_path, picked):
    lc = json.load(open(file_path))
    print(lc)
    lc_df = pd.DataFrame(lc)
    print(lc_df)

    lc_df = lc_df.loc[lc_df['values'].isin(picked)]

    values = lc_df["values"].to_list()
    print(f"values : {values}")
    print(f"values length : {len(values)}")
    palette = lc_df["palette"].to_list()
    print(f"palette : {palette}")
    print(f"palette length : {len(palette)}")
    labels = lc_df["label"].to_list()
    print(f"labels : {labels}")
    print(f"labels length : {len(labels)}")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, len(palette) * 0.2))  # Adjust height based on number of rows
    ax.axis("tight")
    ax.axis("off")

    # Create table data
    table_data = [["Color", "Description"]] + [[value, desc] for value, desc in zip(values, labels)]

    # Create the table
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")

    # Apply color to the cells in the "Color" column
    for i, hex_code in enumerate(palette, start=1):  # Skip header row
        table[(i, 0)].set_facecolor(hex_code)

    # Adjust font size and layout
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width([0, 1])

    # Show the plot
    plt.show()

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

def compare_with_ground_truth(path, predicted):
    ground_truth = f"{path}/ground_truth.tif"
    cmap, legend = load_cmap(file_path="../config/color_map.json")

    with rasterio.open(ground_truth) as f:
        actual = f.read(1)
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(20, 20))
        axs[0].set_title('Ground truth')
        axs[0].imshow(actual, origin='upper', cmap=cmap)

        axs[1].set_title('Predicted')
        axs[1].imshow(predicted, origin='lower', cmap=cmap)

        plt.savefig(f"{path}/classification_output.png")
        plt.show()

def covert_crs_of_shapefile(input_file, output_file, crs = 4326):
    gdf = gpd.read_file(input_file)
    print(f"Current CRS: {gdf.crs}")
    gdf = gdf.to_crs(epsg=crs)
    gdf.to_file(output_file)
    print("Coordinate system converted successfully")

# if __name__ == "__main__":
#     file_path = "data/land_cover/cork/clipped_raster.tif"
#     read_raster(file_path)
#     file_path = "data/SENTINEL-2/2024-12-03/stacked/stacked_bands.tiff"
#     read_raster(file_path)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     print(f"download_dir : {download_dir}")
#     stacked_dir = f"{download_dir}/stacked"
#     out_file = f"{stacked_dir}/stacked_bands.tiff"
#     tiff = rasterio.open(out_file)
#     print(f"view_tiff|tiff:{tiff}")
#     rasterio.plot.show(tiff)
#     plt.show()

# if __name__ == "__main__":
#     file_path = "data/land_cover/crookstown/wgs84/crookstown.shp"
#     output_path = "data/land_cover/crookstown/crookstown_raster.tif"
#     min_area_polygon = get_polygon_from_shapefile(file_path)
#     print(f"min_area_polygons: {min_area_polygon}")

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}"
#     file_path = "config/corine_landcover_2018.tif"
#     min_area_polygon = get_polygon_from_shapefile()
#     output_path = f"{download_dir}/ground_truth.tif"
#     clip_tiff(file_path, output_path, min_area_polygon)

# if __name__ == "__main__":
#     file_path = "data/land_cover/crookstown/raster/U2018_CLC2018_V2020_20u1.tif"
#     shapefile = "data/land_cover/crookstown/wgs84/crookstown.shp"
#     min_area_polygon = get_polygon_from_shapefile(shapefile)
#     output_path = "data/land_cover/crookstown/raster/cropped_raster.tif"
#     clip_tiff(file_path, output_path, min_area_polygon)
#     view_tiff("data/land_cover/crookstown/raster/cropped_raster.tif")

# if __name__ == "__main__":
#     # file_path_2006 = "data/land_cover/2006/U2012_CLC2006_V2020_20u1.tif"
#     # view_tiff(file_path_2006, 2006)
#     # file_path_2012 = "data/land_cover/2012/U2018_CLC2012_V2020_20u1_raster100m.tif"
#     # view_tiff(file_path_2012, 2012)
#     file_path = "data/land_cover/cork/U2018_CLC2018_V2020_20u1.tif"
#     geo_json = "config/cork.geojson"
#     output_path = "data/land_cover/cork/clipped_raster.tif"
#     clip_tiff(file_path, output_path, geo_json)
#     view_tiff("data/land_cover/cork/clipped_raster.tif")

# if __name__ == "__main__":
#     file_path = "config/color_map.json"
#     # plot_color_map(file_path)
#     plot_color_map_selected(file_path, [112, 131, 211, 231, 243, 311, 312, 313, 324, 999])
#
# if __name__ == "__main__":
#     input_shapefile = "data/land_cover/cop/CLC18_IE_wgs84/CLC18_IE_wgs84.shp"
#     output_shapefile = "data/land_cover/cop/resampled_10m/CLC18_IE_wgs84_10m.shp"
#     original_resolution = 100
#     target_resolution = 10
#     resample_shapefile(input_shapefile, output_shapefile, original_resolution, target_resolution)

if __name__ == "__main__":
    ""
    # input_shapefile = "C:\\Users\AdikariAdikari\DataCollection\catchments\crookstown\Crookstown_subbasin_all.shp"
    input_shapefile = "C:\\Users\AdikariAdikari\DataCollection\catchments\owenabue\Owenabue_all_subbasins_proj.shp"
    # output_shapefile = "config/wgs84/crookstown/crookstown.shp"
    output_shapefile = "config/wgs84/owenabue/owenabue.shp"
    covert_crs_of_shapefile(input_shapefile, output_shapefile)