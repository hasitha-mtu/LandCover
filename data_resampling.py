import rasterio
from rasterio.warp import Resampling
from datetime import date
import os
import glob
import shutil

from utils import select_directory_list


def perform_resampling(input_file, output_file, upscale_factor = 2):
    with rasterio.open(input_file) as dataset:
        resampled_data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.nearest)
        transform_dataset = dataset.transform * dataset.transform.scale(
            (dataset.width / resampled_data.shape[-1]),
            (dataset.height / resampled_data.shape[-2])
        )
        resampled_shape = resampled_data.shape

    with rasterio.open(output_file, mode="w", driver="JPEG", width=resampled_shape[2],
                       height=resampled_shape[1], count=1, crs=dataset.crs,
                       transform=transform_dataset, dtype=resampled_data.dtype) as dst:
        dst.write(resampled_data)

def band_image_directories(download_dir):
    parent_list = select_directory_list(download_dir, ".SAFE", 2)
    dir_list = []
    for parent_dir in parent_list:
        dir_path = f"{download_dir}/{parent_dir}/GRANULE"
        parent_list1 = select_directory_list(dir_path, None, 1)
        dir_path1 = f"{dir_path}/{parent_list1[0]}/IMG_DATA"
        parent_dir_path = f"{download_dir}/{parent_dir}"
        dir_list.append([parent_dir_path, dir_path1])
    return dir_list

def get_image_for_the_band(dir_path, resolution, band_name):
    rez_dir = f"{dir_path}/R{resolution}m"
    images = glob.glob(f"{rez_dir}/*_{band_name}_{resolution}m.jp2")
    return images

def resample_images(target_resolution, source_dir, resolution_list=None, band_list=None):
    if band_list is None:
        band_list = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'SCL',
                     'TCI', 'WVP']
    if resolution_list is None:
        resolution_list = [10, 20, 60]
    image_dir_list = band_image_directories(source_dir)
    resolution_list.remove(target_resolution)
    for [parent_dir, image_dir] in image_dir_list:
        print("---------------------------------------------------------------------------------")
        resample_dir = f"{parent_dir}/resampled"
        os.makedirs(resample_dir, exist_ok=True)
        for band in band_list:
            band_images = get_image_for_the_band(image_dir, target_resolution, band)
            if len(band_images) > 0:
                shutil.copyfile(band_images[0], f"{resample_dir}/{band}_{target_resolution}m.jp2")
                print(band_images[0])
            else:
                for resolution in resolution_list:
                    band_images = get_image_for_the_band(image_dir, resolution, band)
                    if len(band_images) > 0:
                        perform_resampling(band_images[0], f"{resample_dir}/{band}_{target_resolution}m.jp2",
                                           upscale_factor=resolution/target_resolution)
                        print(band_images[0])

if __name__ == "__main__":
    collection_name = "SENTINEL-2"
    resolution = 10 #  Define the target resolution (e.g., 10 meters)
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    print(f"download_dir : {download_dir}")
    print(select_directory_list(download_dir, None, 1))
    resolution_list = [10, 20, 60]
    band_list = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'SCL',
                 'TCI', 'WVP']
    resample_images(10, download_dir, resolution_list=resolution_list, band_list=band_list)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"
#     resolution = 10 # Define the target resolution (e.g., 10 meters)
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}/raw"
#     print(f"download_dir : {download_dir}")
#     print(select_directory_list(download_dir, None, 1))



