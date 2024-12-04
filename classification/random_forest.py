from osgeo import gdal, gdal_array


if __name__ == "__main__":
    img_ds = gdal.Open('/content/sample_images/data/LE70220491999322EDC01_stack.gtif', gdal.GA_ReadOnly)
    roi_ds = gdal.Open('/content/sample_images/data/training_data.gtif', gdal.GA_ReadOnly)

