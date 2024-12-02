from datetime import date

from config import get_resolution_list, get_band_list, get_collection_name
from data_re_projection import perform_re_projection
from data_resampling import resample_images


def perform_data_preprocessing(resolution = 10): # Define the target resolution (e.g., 10 meters)
    collection_name = get_collection_name()
    today_string = date.today().strftime("%Y-%m-%d")
    download_dir = f"data/{collection_name}/{today_string}"
    print(f"download_dir : {download_dir}")
    resolution_list = get_resolution_list()
    band_list = get_band_list()
    resample_images(resolution, download_dir, resolution_list=resolution_list, band_list=band_list)
    perform_re_projection(download_dir)

if __name__ == "__main__":
    perform_data_preprocessing()



