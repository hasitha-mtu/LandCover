import glob
import multiprocessing
import os
import zipfile
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape
from shapely.ops import unary_union

from utils import get_polygon

copernicus_user = "adikari.adikari@mycit.ie"  # copernicus User
copernicus_password = "Hasitha@4805"  # copernicus Password
valid_http_response_codes = (301, 302, 303, 307)


def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]


def download_data(download_location, area_foot_print, start_date, end_date,
                  data_collection = "SENTINEL-2"):
    print(
        f"Download data from data_collection: {data_collection} from  {start_date} to {end_date} "
        f"for the area {area_foot_print}")
    products_json = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{area_foot_print}') "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T00:00:00.000Z&$count=True&$top=1000").json()

    # products_json = requests.get(
    #     f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' "
    #     f"and OData.CSC.Intersects(area=geography'SRID=4326;{area_foot_print}') "
    #     f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
    #     f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value eq '0.3') "
    #     f"and ContentDate/Start lt {end_date}T00:00:00.000Z&$count=True&$top=1000").json()

    product_df = pd.DataFrame.from_dict(products_json['value'])
    print(f"products_df : {product_df.shape}")
    if product_df.shape[0] > 0:
        print(f"product_df info : {product_df.info()}")
        product_df["geometry"] = product_df["GeoFootprint"].apply(shape)
        geo_df = gpd.GeoDataFrame(product_df).set_geometry("geometry")  # Convert PD to GPD
        print(f"geo_df : {geo_df}")
        geo_df = geo_df[~geo_df["Name"].str.contains("L1C")]  # Remove L1 data from dataframe
        geo_df["identifier"] = geo_df["Name"].str.split(".").str[0]
        feature_count = len(geo_df)
        print(f"Total feature count in product_df : {feature_count}")
        if feature_count > 0:
            try:
                os.makedirs(download_location, exist_ok=True)
                for i, feature in enumerate(geo_df.iterfeatures()):
                    with multiprocessing.Pool() as pool:
                        pool.map(download_image_zip, [(feature, download_location)])
            except Exception as e:
                print(f"Problem with server error: {e}")
        else:
            print(f"Features not found|{feature_count}")
    else:
        print(f"No data found for the given criteria")


def download_data1(download_location, area_foot_print, start_date, end_date,
                  data_collection = "SENTINEL-2"):
    print(
        f"Download data from data_collection: {data_collection} from  {start_date} to {end_date} "
        f"for the area {area_foot_print}")
    products_json = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{area_foot_print}') "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T00:00:00.000Z&$count=True&$top=1000").json()
    print(f"products_json : {products_json}")
    # products_json = requests.get(
    #     f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' "
    #     f"and OData.CSC.Intersects(area=geography'SRID=4326;{area_foot_print}') "
    #     f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
    #     f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value eq '0.3') "
    #     f"and ContentDate/Start lt {end_date}T00:00:00.000Z&$count=True&$top=1000").json()

    product_df = pd.DataFrame.from_dict(products_json['value'])
    print(f"products_df : {product_df.head()}")
    if len(product_df.index) == 0:
        raise Exception("No images for selected period")
    print(f"header values of product_df: {product_df.columns.values}")
    product_df = product_df.sort_values(['PublicationDate'], ascending=True)
    print(f"products_df : {product_df.head()}")

    product_df["geometry"] = product_df["GeoFootprint"].apply(shape)
    geo_df = gpd.GeoDataFrame(product_df).set_geometry("geometry")  # Convert PD to GPD

    geo_df = geo_df[~geo_df["Name"].str.contains("L1C")]  # Remove L1 data from dataframe
    geo_df["identifier"] = geo_df["Name"].str.split(".").str[0]

    tile_footprints = []
    for x in geo_df[["identifier", "Id", "Name", "geometry", "Footprint", "GeoFootprint"]].T.to_dict().items():
        tile_footprints.append({**x[1], "index": x[0]})

    print(tile_footprints[:3])
    union_polygons = make_union_polygon(tile_footprints)
    print(f"union_polygons: {union_polygons}")
    min_area_polygons = get_min_covering(union_polygons)
    print(f"min_area_polygons: {min_area_polygons}")
    try:
        os.makedirs(download_location, exist_ok=True)
        for i, feature in enumerate(min_area_polygons):
            print(f"{i} => {feature}")
            print(feature['Id'])
            print(feature['identifier'])
            print(feature['Name'])
            print(feature['geometry'])
            print(feature['Footprint'])
            print(feature['GeoFootprint'])
            print(feature['index'])
            download_image_zip1(feature['Id'], download_location)
    except Exception as e:
        print(f"Problem with server error: {e}")

def download_image_zip(file_info):
    print(f"download_image_zip|file_info : {file_info}")
    feature, dir_path = file_info
    try:
        session = requests.Session()
        keycloak_token = get_keycloak(copernicus_user, copernicus_password)
        session.headers.update({"Authorization": f"Bearer {keycloak_token}"})
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feature['properties']['Id']})/$value"
        response = session.get(url, allow_redirects=False)
        print(f"response : {response}")
        while response.status_code in valid_http_response_codes:
            url = response.headers["Location"]
            response = session.get(url, allow_redirects=False)
        print(feature["properties"]["Id"])
        file = session.get(url, verify=False, allow_redirects=True)
        with open(f"{dir_path}/{feature['properties']['identifier']}.zip", "wb", ) as p:
            print(feature["properties"]["Name"])
            p.write(file.content)
    except Exception as e:
        print(f"Problem with server error: {e}")

def download_image_zip1(product_id, dir_path):
    try:
        session = requests.Session()
        keycloak_token = get_keycloak(copernicus_user, copernicus_password)
        session.headers.update({"Authorization": f"Bearer {keycloak_token}"})
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        response = session.get(url, allow_redirects=False)
        print(f"response : {response}")
        while response.status_code in valid_http_response_codes:
            url = response.headers["Location"]
            response = session.get(url, allow_redirects=False)
        file = session.get(url, verify=False, allow_redirects=True)
        with open(f"{dir_path}/{product_id}.zip", "wb", ) as p:
            p.write(file.content)
    except Exception as e:
        print(f"Problem with server error: {e}")

def unzip_downloaded_files(zip_dir):
    for zip_file_path in glob.glob(f"{zip_dir}/*.zip"):
        print(f"Unzipping file : {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)
        os.remove(zip_file_path) # Remove original zip file after extraction

def make_union_polygon(sentinal_polygons):
    """
    This algorithm goes through all polygons and adds them to union_poly only if they're
    not already contained in union_poly.
    (in other words, we're only adding them to union_poly if they can increase the total area)
    """
    union_poly = sentinal_polygons[0]["geometry"]
    union_parts = [sentinal_polygons[0], ]
    for fp in sentinal_polygons[1:]:
        p = fp["geometry"]
        common = union_poly.intersection(p)
        if p.area - common.area < 0.001:
            pass
        else:
            union_parts.append(fp)
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

    whole = unary_union([x["geometry"] for x in union_polygons])
    L = [x["geometry"] for x in union_polygons]
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

if __name__ == "__main__":
    today = date.today()
    today_string = today.strftime("%Y-%m-%d")
    yesterday = today - timedelta(days=10)
    yesterday_string = yesterday.strftime("%Y-%m-%d")
    selected_area = get_polygon('config/mallow.geojson')
    collection_name = "SENTINEL-2"  # Sentinel satellite
    download_dir = f"data/{collection_name}/{today_string}"
    print(f"download_dir : {download_dir}")
    download_data1(download_dir, selected_area, yesterday_string,
                  today_string, data_collection=collection_name)
    unzip_downloaded_files(download_dir)

# if __name__ == "__main__":
#     collection_name = "SENTINEL-2"  # Sentinel satellite
#     today_string = date.today().strftime("%Y-%m-%d")
#     download_dir = f"data/{collection_name}/{today_string}/raw"
#     print(f"download_dir : {download_dir}")
#     unzip_downloaded_files(download_dir)
