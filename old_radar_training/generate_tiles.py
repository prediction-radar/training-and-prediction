import csv
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def make_square_and_resize(image_path, output_path, size=(256, 256)):
    # Open the image
    with Image.open(image_path) as img:
        # Calculate the size to make the image square
        short_side = min(img.size)
        # Find the coordinates to crop the image to a square
        left = (img.width - short_side) / 2
        top = (img.height - short_side) / 2
        right = (img.width + short_side) / 2
        bottom = (img.height + short_side) / 2
        # Crop the image to a square
        img_square = img.crop((left, top, right, bottom))
        # Resize the image
        img_resized = img_square.resize(size, Image.ANTIALIAS)
        # Save the resized image
        img_resized.save(output_path)

def process_tile(input_grib2_file, row):
    """Function to process a single tile and convert it to PNG."""
    zoom_level = row['Zoom Level']
    x = row['X']
    y = row['Y']
    top_lat = row['Top Latitude']
    bottom_lat = row['Bottom Latitude']
    left_lon = row['Left Longitude']
    right_lon = row['Right Longitude']

    # Define the output directories and file names
    output_grib2_dir = os.path.join('output_grib2', zoom_level, x)
    work_dir = 'work'
    results_dir = 'results'
    os.makedirs(output_grib2_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    grib2_file_path = os.path.join(output_grib2_dir, f"{zoom_level}_{y}_{x}.grib2")
    intermediate_tif = os.path.join(work_dir, f"intermediate_{zoom_level}_{y}_{x}.tif")
    output_colored_tif = os.path.join(work_dir, f"output_colored_{zoom_level}_{y}_{x}.tif")
    final_png = os.path.join(results_dir, f"{zoom_level}_{y}_{x}.png")

    # CDO command to slice the input grib2 file
    cdo_command = f"cdo sellonlatbox,{left_lon},{right_lon},{top_lat},{bottom_lat} {input_grib2_file} {grib2_file_path}"
    try:
        subprocess.run(cdo_command, check=True, shell=True)
        print(f"Generated: {grib2_file_path}")

        # Convert .grib2 to .tif
        gdal_translate_to_tif = f"gdal_translate -a_nodata 0 -of GTiff {grib2_file_path} {intermediate_tif}"
        subprocess.run(gdal_translate_to_tif, check=True, shell=True)

        # Apply color relief
        gdal_color_relief = f"gdaldem color-relief {intermediate_tif} colortable.txt {output_colored_tif}"
        subprocess.run(gdal_color_relief, check=True, shell=True)

        # Convert .tif to .png
        gdal_translate_to_png = f"gdal_translate -of PNG {output_colored_tif} {final_png} -a_nodata 0"
        subprocess.run(gdal_translate_to_png, check=True, shell=True)
        
        #make_square_and_resize(final_png, final_png, size=(256, 256))


    except subprocess.CalledProcessError as e:
        print(f"Error processing {grib2_file_path}: {e}")

def process_grib2_file(csv_file_path, input_grib2_file, max_workers=10):
    """
    Reads tiles information from a CSV file and processes each tile in parallel
    to create smaller .grib2 files and convert them to PNG.
    """
    with open(csv_file_path, newline='') as csvfile, ThreadPoolExecutor(max_workers=max_workers) as executor:
        reader = csv.DictReader(csvfile)
        futures = [executor.submit(process_tile, input_grib2_file, row) for row in reader]

        for future in futures:
            future.result()

# Specify the CSV file path and the input .grib2 file
csv_file_path = 'position_filtered_tiles_data.csv'
input_grib2_file = 'today_grib.grib2'

# Call the function to start processing
process_grib2_file(csv_file_path, input_grib2_file, max_workers=10)