import glob
import os
from datetime import date

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from utils import get_parent_directories


def re_project_image(input_file, output_file, dst_crs = 'EPSG:4326'):
    print(f"input_file : {input_file}")
    with rasterio.open(input_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        print(f"kwargs : {kwargs}")
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def perform_re_projection(input_dir, dst_crs = 'EPSG:4326'):
    parent_dirs = get_parent_directories(input_dir)
    for parent_dir in parent_dirs:
        reproject_dir = f"{parent_dir}/reprojected"
        os.makedirs(reproject_dir, exist_ok=True)
        resampled_dir = f"{parent_dir}/resampled/*.jp2"
        print(f"resampled_dir : {resampled_dir}")
        resampled_images = glob.glob(resampled_dir)
        print(f"resampled_images : {resampled_images}")
        for resampled_image in resampled_images:
            print(f"resampled_image : {resampled_image}")
            re_project_image(resampled_image, resampled_image.replace("resampled", "reprojected"), dst_crs)

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10 # Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    print(f"download_dir : {download_dir}")
    perform_re_projection(download_dir)
