from datetime import datetime, timedelta, timezone
import bs4 as bs
import os, shutil
import requests
import pandas as pd

# Downloading file
root_dir = os.getcwd() + "/"
file_type = "npy"

# Function to get the latest file based on filename date format
def get_last_file_timestamp(destination_folder):
    latest_time = None
    for filename in os.listdir(destination_folder):
        try:
            # Extract date-time from filename
            file_time_str = filename.split('.')[0]  # 'd-m-Y-HM'
            file_time = datetime.strptime(file_time_str, '%d-%m-%Y-%H%M')
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
        except ValueError:
            print(f"Skipping file with incorrect format: {filename}")
    return latest_time

def download_files(folder_url, destination_folder, start_time, end_time, file_time):

  try:

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
      os.makedirs(destination_folder)

    # Delete old files
    for filename in os.listdir(destination_folder):
      file_path = os.path.join(destination_folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))


    # Get the list of files in the folder
    response = requests.get(folder_url)
    data = bs.BeautifulSoup(response.text, "html.parser")

    csv_files = data.find_all("a", href=lambda href: href and href.endswith(f'.{file_type}'))

    # Filter CSV files based on time range
    filtered_files = []
    for file_name in csv_files:
        file_type_filename = file_name['href'].split('/')[-1]
        try:
            # Parse the filename to extract the creation or modification time
            file_time_str = file_type_filename.split('.')[0]
            file_time = datetime.strptime(file_time_str, '%d-%m-%Y-%H%M')
            if start_time <= file_time <= end_time:
                filtered_files.append(file_name)
        except ValueError as e:
            # Handle parsing errors (e.g., invalid filename format)
            print(f"Error parsing filename: {file_type_filename}, {e.args[0]}")
    
    if len(filtered_files) != 30:
        raise Exception(f"Number of files must equal 30 but equaled {len(filtered_files)}")

    for file_name in filtered_files:
      file_type_url = file_name['href']  # Get the CSV file URL
      file_type_filename = file_type_url.split('/')[-1]  # Extract the filename
      link = folder_url + file_type_filename
      destination_link = os.path.join(destination_folder, file_type_filename)

      response = requests.get(link)
      with open(destination_link, 'wb') as f:
         f.write(response.content)

    print("Downloading finished, outcome unknown.")

  except requests.exceptions.RequestException as e:
    print(f"Error downloading files: {e}")

current_destination_folder = root_dir + f'data/{file_type}_files'

# Get the last file timestamp in the destination folder
last_file_time = get_last_file_timestamp(current_destination_folder)

# Initiate download of files
folder_url = f'http://66.213.177.43/{file_type}_files/'
download_destination_folder = root_dir + f'data/new_{file_type}_files'
if last_file_time is None:
    print("No valid files found in the folder.")
else:
    start_time = last_file_time + timedelta(minutes=2)
    end_time = start_time + (timedelta(hours=1) - timedelta(minutes=2))
    
    # Download files within the new time range
    download_files(folder_url, download_destination_folder, start_time, end_time, file_type)