import os
import os.path
import re
from datetime import date
import platform

import rasterio
import rasterio.mask
import rasterio.merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from pprint import pprint
import matplotlib.pyplot as plt

from utils import get_parent_directories
from utils import get_polygon_from_shapefile as get_polygon
# from utils import get_polygon
import glob
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.plot import show_hist
import numpy as np

"""
Explanation on preprocessing for sentinal 2 L2 data
https://sentiwiki.copernicus.eu/web/s2-processing
"""

def merge_file(tiff_paths, output_file, DEBUG):
    raster_list = [rasterio.open(f, mode='r', driver="GTiff") for f in tiff_paths]
    merged_data, out_trans = rasterio.merge.merge(raster_list)

    if DEBUG:
        fig, ax = plt.subplots(figsize=(14, 14))
        show(merged_data, cmap='terrain', ax=ax)
        plt.show()

    merged_meta = raster_list[0].meta.copy()
    print(f"merged_meta : {merged_meta}")
    merged_meta.update({"driver": "GTiff",
                            "height": merged_data.shape[1],
                            "width": merged_data.shape[2],
                            "transform": out_trans,
                            "crs": raster_list[0].crs
                            # "count": 3,
                            })
    if DEBUG:
        for x in [x.meta for x in raster_list] + [merged_meta]:
            pprint(x)

    with rasterio.open(output_file, mode="w", **merged_meta) as dest:
        dest.write(merged_data)

def mosaic_images1(image_paths, output_file, DEBUG):
    src_files = [rasterio.open(fp) for fp in image_paths]
    mosaic, out_transform = merge(src_files)
    if DEBUG:
        fig, ax = plt.subplots(figsize=(14, 14))
        show(mosaic, cmap='viridis', ax=ax)
        plt.show()
    out_meta = src_files[0].meta.copy()
    out_meta.update({"driver": "GTiff",
                      "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_transform})
    if DEBUG:
        for x in [x.meta for x in src_files]:
            pprint(x)
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

def mosaic_images(image_paths, output_file, DEBUG):
    src_files = [rasterio.open(fp) for fp in image_paths]
    mosaic, out_transform = merge(src_files)
    if DEBUG:
        fig, ax = plt.subplots(figsize=(14, 14))
        show(mosaic, cmap='viridis', ax=ax)
        plt.show()
    out_meta = src_files[0].meta.copy()
    out_meta.update({"driver": "GTiff",
                      "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_transform})
    check_merge_accuracy(src_files, mosaic, DEBUG)
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)


def check_merge_accuracy(src_files, mosaic, DEBUG):
    if len(src_files) == 2:
        [src1, src2] = src_files
        src1_data = src1.read()
        src2_data = src2.read()

        # Calculate overlapping area
        overlap_mask = np.logical_and(
            ~np.all(src1_data == src1.nodata, axis=0),
            ~np.all(src2_data == src2.nodata, axis=0)
        )

        # Extract overlapping data
        src1_overlap = src1_data[:, overlap_mask]
        src2_overlap = src2_data[:, overlap_mask]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((src1_overlap - src2_overlap) ** 2))
        print(f"RMSE: {rmse}")

        if DEBUG:
            # Calculate correlation coefficient
            corr_coef = np.corrcoef(src1_overlap.flatten(), src2_overlap.flatten())[0, 1]
            print(f"Correlation Coefficient: {corr_coef}")

            # Plot histograms
            show_hist(src1, bins=50, histtype='stepfilled', lw=0.0, alpha=0.5, label='Image 1')
            show_hist(src2, bins=50, histtype='stepfilled', lw=0.0, alpha=0.5, label='Image 2')
            show_hist(mosaic, bins=50, histtype='stepfilled', lw=0.0, alpha=0.5, label='Merged')
            plt.show()
    else:
        pass

def convert_jp2_to_tiff(path):
    print("Converting " + path)
    with rasterio.open(path, mode="r") as src:
        profile = src.meta.copy()
        profile.update(driver="GTiff")
        outfile = re.sub("resampled", "converted", path)
        outfile = re.sub(".jp2", ".tiff", outfile)
        with rasterio.open(outfile, 'w', **profile) as dst:
            dst.write(src.read())

def perform_jp2_to_tiff_conversion(input_dir):
    print(f"perform_jp2_to_tiff_conversion|input_dir: {input_dir}")
    parent_dirs = get_parent_directories(input_dir)
    print(f"perform_jp2_to_tiff_conversion|parent_dirs: {parent_dirs}")
    for parent_dir in parent_dirs:
        resampled_dir = f"{parent_dir}/resampled"
        print(f"resampled_dir : {resampled_dir}")
        converted_dir = f"{parent_dir}/converted"
        print(f"converted_dir : {converted_dir}")
        os.makedirs(converted_dir, exist_ok=True)
        jp2_images = glob.glob(f"{resampled_dir}/*_10m.jp2")
        print(f"jp2_images : {jp2_images}")
        for jp2_image in jp2_images:
            convert_jp2_to_tiff(jp2_image)

def re_project_file(image_path, output_file, dst_crs):
    """
    Reprojecting the images to  EPSG:4326
    """
    print(f"reproject image_path : {image_path}")
    print(f"reproject output_file : {output_file}")
    with rasterio.open(image_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
        })
        with rasterio.open(output_file, mode="w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def merge_files(download_dir, resolution, band_list, debug):
    merged_dir = f"{download_dir}/merged"
    os.makedirs(merged_dir, exist_ok=True)
    parent_dirs = get_parent_directories(download_dir)
    for band in band_list:
        tiff_path_list = []
        for parent_dir in parent_dirs:
            print(f"parent_dir : {parent_dir}")
            tiff_path = f"{parent_dir}/converted/{band}_{resolution}m.tiff"
            if os.path.isfile(tiff_path):
                tiff_path_list.append(tiff_path)
        print(f"Band {band}m tiff files : {tiff_path_list}")
        mosaic_images(tiff_path_list, f"{merged_dir}/{band}_{resolution}m.tiff", debug)

def re_project_files(download_dir, resolution,  dst_crs='EPSG:4326'):
    reprojected_dir = f"{download_dir}/reprojected"
    os.makedirs(reprojected_dir, exist_ok=True)
    merged_dir = f"{download_dir}/merged"
    merged_files = glob.glob(f"{merged_dir}/*_{resolution}m.tiff")
    for merged_file in merged_files:
        output_file = re.sub("merged", "reprojected", merged_file)
        re_project_file(merged_file, output_file, dst_crs)


def crop_image(input_file, output_file, polygon, debug):
    print(f"crop_image input_file:{input_file}")
    print(f"crop_image output_file:{output_file}")
    print(f"crop_image polygon:{polygon}")
    with rasterio.open(input_file) as src:
        print(f"crop_image input.shape:{src.read(1).shape}")
        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        print(f"crop_image out_image.shape:{out_image.shape}")
        print(f"crop_image out_meta:{out_meta}")
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)
            if debug:
                fig, ax = plt.subplots(figsize=(14, 14))
                show(out_image, cmap='viridis', ax=ax)
                plt.show()

def crop_image_files(download_dir, resolution):
    reprojected_dir = f"{download_dir}/reprojected"
    roi_dir = f"{download_dir}/roi"
    os.makedirs(roi_dir, exist_ok=True)
    merged_files = glob.glob(f"{reprojected_dir}/*_{resolution}m.tiff")
    # selected_area = get_polygon(path="config/smaller_selected_map.geojson")
    selected_area = get_polygon()
    print(f"selected_area : {selected_area}")
    for merged_file in merged_files:
        output_file = re.sub("reprojected", "roi", merged_file)
        crop_image(merged_file, output_file, selected_area, True)

def stack_bands_together(input_dir, resolution, band_list):
    roi_dir = f"{input_dir}/roi"
    band_rasters = []
    for band in band_list:
        band_file = glob.glob(f"{roi_dir}/{band}_{resolution}m.tiff")
        band_rasters = band_rasters + band_file
    print(band_rasters)
    with rasterio.open(band_rasters[0]) as ds:
        metadata = ds.profile
        metadata.update({"count": len(band_rasters)})
        stacked_dir = f"{input_dir}/stacked"
        os.makedirs(stacked_dir, exist_ok=True)
        out_file = f"{stacked_dir}/stacked_bands.tiff"
        with rasterio.open(out_file, 'w', **metadata) as dest:
            for i, band_file in enumerate(band_rasters):
                band_data = rasterio.open(band_file)
                dest.write(band_data.read(1),i+1)
        print(f"Stacked file created {out_file}")

# if __name__ == "__main__":
#     today_string = date.today().strftime("%Y-%m-%d")
#     collection_name = "SENTINEL-2"  # Sentinel satellite
#     download_dir = f"data/{collection_name}/{today_string}"
#     resolution = 10
#     stack_bands_together(download_dir, resolution, ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])


# if __name__ == "__main__":
#     today_string = date.today().strftime("%Y-%m-%d")
#     collection_name = "SENTINEL-2"  # Sentinel satellite
#     download_dir = f"data/{collection_name}/{today_string}"
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     crop_image_files(download_dir, resolution)

# if __name__ == "__main__":
#     today_string = date.today().strftime("%Y-%m-%d")
#     collection_name = "SENTINEL-2"  # Sentinel satellite
#     download_dir = f"data/{collection_name}/{today_string}"
#     merged_band_dir = f"data/{collection_name}/{today_string}/merged"
#     band_list = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'SCL',
#                  'TCI', 'WVP']
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     perform_jp2_to_tiff_conversion(download_dir)
#     merge_files(download_dir, resolution, band_list, False)
#
#     re_project_files(download_dir, resolution)
#     crop_image_files(download_dir, resolution)
#     stack_bands_together(download_dir, 10, ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])

# if __name__ == "__main__":
#     today_string = date.today().strftime("%Y-%m-%d")
#     collection_name = "SENTINEL-2"  # Sentinel satellite
#     download_dir = f"data/{collection_name}/{today_string}"
#     # merged_band_dir = f"data/{collection_name}/{today_string}/merged"
#     band_list = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'SCL',
#                  'TCI', 'WVP']
#     resolution = 10  # Define the target resolution (e.g., 10 meters)
#     perform_jp2_to_tiff_conversion(download_dir)
#     merge_files(download_dir, resolution, band_list, True)

if __name__ == "__main__":
    today_string = date.today().strftime("%Y-%m-%d")
    collection_name = "SENTINEL-2"  # Sentinel satellite
    download_dir = f"data/{collection_name}/{today_string}"
    # merged_band_dir = f"data/{collection_name}/{today_string}/merged"
    band_list = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'SCL',
                 'TCI', 'WVP']
    resolution = 10  # Define the target resolution (e.g., 10 meters)
    perform_jp2_to_tiff_conversion(download_dir)
    merge_files(download_dir, resolution, band_list, False)
    re_project_files(download_dir, resolution)
    crop_image_files(download_dir, resolution)
    stack_bands_together(download_dir, 10, ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])





