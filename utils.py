import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import rasterio.plot
from matplotlib.colors import ListedColormap
from sentinelsat import read_geojson
from shapely.geometry import shape
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from datetime import date

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

def clip_tiff(tiff_path, output_path, polygon_path):
    roi_polygon = get_polygon(polygon_path)
    print(f"roi_polygon: {roi_polygon}")
    with rasterio.open(tiff_path) as src:
        out_image, out_transform = mask(src, [shape(roi_polygon)], crop=True)
        out_meta = src.meta.copy()
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

def get_polygon(path = "config/cork2.geojson"):
    geojson = read_geojson(path)
    polygon_jsons = geojson["features"]
    polygon_json = polygon_jsons[0]
    geometry_data = polygon_json["geometry"]
    polygon = shape(geometry_data)
    return polygon

def read_shape_file(file_path):
    gdf = gpd.read_file(file_path)
    print(gdf.head())

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

if __name__ == "__main__":
    file_path = "data/land_cover/cork/U2018_CLC2018_V2020_20u1.tif"
    geo_json = "config/cork2.geojson"
    output_path = "data/land_cover/cork2/clipped_raster.tif"
    clip_tiff(file_path, output_path, geo_json)
    view_tiff("data/land_cover/cork2/clipped_raster.tif")

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