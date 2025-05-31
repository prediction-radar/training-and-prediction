import os
import requests
import gzip
import shutil
import subprocess
import glob
import random
import math
import numpy as np
import xarray as xr
from datetime import datetime, timezone

# Configuration
current_time = datetime.now(timezone.utc).strftime('%d-%m-%Y-%H%M')
file_url = 'https://mrms.ncep.noaa.gov/data/2D/MergedReflectivityAtLowestAltitude/MRMS_MergedReflectivityAtLowestAltitude.latest.grib2.gz'
grib_directory = '/var/www/radar_data/grib_files'
csv_directory = '/var/www/radar_data/csv_files'
npy_directory = '/var/www/radar_data/npy_files'

compressed_file_path = os.path.join(grib_directory, f'{current_time}.grib2.gz')
decompressed_file_path = os.path.join(grib_directory, f'{current_time}_decompressed.grib2')
cropped_grib_file = os.path.join(grib_directory, f'{current_time}_cropped.grib2')
idx_cropped_grib_file = os.path.join(grib_directory, f'{current_time}_cropped.grib2.9093e.idx')
csv_file = os.path.join(csv_directory, f'{current_time}_cropped.csv')
npy_file = os.path.join(npy_directory, f'{current_time}.npy')

# Ensure the local directory exists
os.makedirs(grib_directory, exist_ok=True)

def download_file(url, compressed_file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(compressed_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {url} to {compressed_file_path}")

def decompress_file(compressed_path, decompressed_path):
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(decompressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed {compressed_path} to {decompressed_path}")

def crop_grib(input_file, output_file, left_long, right_long, top_lat, bottom_lat):
    # Format the command and arguments

    # Construct the wgrib2 command
    command = [
        'wgrib2',
        input_file,
        '-small_grib',
        f'{left_long}:{right_long}',
        f'{top_lat}:{bottom_lat}',
        output_file
    ]

    # Run the command
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'Successfully cropped the GRIB2 file: {output_file}')
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f'Error occurred: {e.stderr.decode("utf-8")}')

def random_point_in_bbox(top_latitude, bottom_latitude, left_longitude, right_longitude, file_path):
    """
    Generates a random latitude and longitude point within the given bounding box.

    Parameters:
    top_latitude (float): The northernmost latitude of the bounding box.
    bottom_latitude (float): The southernmost latitude of the bounding box.
    left_longitude (float): The westernmost longitude of the bounding box.
    right_longitude (float): The easternmost longitude of the bounding box.

    Returns:
    tuple: A tuple containing a random latitude and longitude.
    """
        # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Read the contents of the file
        with open(file_path, 'r') as file:
            lat, lon = map(float, file.readline().strip().split(','))
            return lat, lon
    else:
        # Generate a random latitude between the bottom and top latitudes
        latitude = random.uniform(bottom_latitude, top_latitude)
        
        # Generate a random longitude between the left and right longitudes
        longitude = random.uniform(left_longitude, right_longitude)
        
        # Write the random latitude and longitude to the file
        with open(file_path, 'w') as file:
            file.write(f"{latitude},{longitude}")
        
        return latitude, longitude

def calculate_bounding_box(lat, lon, distance_miles):
    """
    Calculate a bounding box around a given latitude and longitude.

    :param lat: Latitude of the central point (in degrees)
    :param lon: Longitude of the central point (in degrees)
    :param distance_miles: Distance for the bounding box from the central point (in miles)
    :return: A dictionary with top, bottom, left, and right bounding coordinates
    """

    # Approximate conversion constants
    miles_per_degree_lat = 69.0  # Approximate miles per degree of latitude
    miles_per_degree_lon = 69.172 * math.cos(math.radians(lat))  # Approximate miles per degree of longitude

    # Calculate the distance in degrees
    delta_lat = distance_miles / miles_per_degree_lat
    delta_lon = distance_miles / miles_per_degree_lon

    # Calculate the bounding coordinates
    top_lat = lat + delta_lat / 2
    bottom_lat = lat - delta_lat / 2
    left_lon = lon - delta_lon / 2
    right_lon = lon + delta_lon / 2

    return {
        'top_lat': top_lat,
        'bottom_lat': bottom_lat,
        'left_lon': left_lon,
        'right_lon': right_lon
    }

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f"File {file_path} does not exists, so it could not be deleted.")

def output_to_csv(grib_file_location, csv_file_location):
    ds = xr.open_dataset(grib_file_location, engine='cfgrib')
    df = ds.to_dataframe().reset_index()
    df.to_csv(csv_file_location, index=False)

def output_to_npy(grib_file_location, npy_file_location):
    ds = xr.open_dataset(grib_file_location, engine='cfgrib')
    data_array = ds.to_array().values
    np.save(npy_file_location, data_array)

def delete_idx_files(directory_path):

    pattern = os.path.join(directory_path, '*.grib2.*.idx')

    # Find all files ending with .idx in the directory
    idx_files = glob.glob(pattern)

    # Loop through the list and delete each file
    for file_path in idx_files:
        os.remove(file_path)
        print(f"Deleted: {file_path}")

def main():
    random_latitude_and_longitude = random_point_in_bbox(50, 25, -125, -65, '/root/data_processing/random_coords.txt')
    latitude = random_latitude_and_longitude[0]
    longitude = random_latitude_and_longitude[1]
    distance = 100
    try:
        if not os.path.exists(compressed_file_path):
            download_file(file_url, compressed_file_path)
        else:
            print(f"The file {compressed_file_path} is already downloaded.")

        if not os.path.exists(decompressed_file_path):
            decompress_file(compressed_file_path, decompressed_file_path)
            #bb = calculate_bounding_box(latitude, longitude, distance)
            #crop_grib(decompressed_file_path, cropped_grib_file, bb['left_lon'], bb['right_lon'], bb['bottom_lat'], bb['top_lat'])
            #output_to_csv(cropped_grib_file, csv_file)
            output_to_npy(decompressed_file_path, npy_file)
            delete_file(compressed_file_path)
            delete_file(decompressed_file_path)
            delete_file(cropped_grib_file)
            delete_idx_files(grib_directory)
        else:
            print(f"The decompressed file {decompressed_file_path} already exists.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
