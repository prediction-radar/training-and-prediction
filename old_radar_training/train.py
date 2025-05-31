import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import os
import shutil
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import AdamW
import subprocess

# Get root dir
root_dir = os.path.dirname(os.getcwd()) + "/radar_data/"

# Load the .npy files into a list
npy_files = sorted(glob(root_dir + 'data/npy_files/*.npy'))


# Make sure data is in expected range
for i in enumerate(npy_files):
    array = np.load(i[1])
    
    array[array < 20] = 0
    array[array > 80] = 80

    np.save(i[1], array)

sequence_length = 15 # Number of .npy files to look back on
prediction_horizon = 15 # Number of .npy files to predict

# Helper function to split into a grid
def split_single_grid(file, square_size, height, width):
    grid = np.load(file).reshape(height, width, 1)
    squares = [
        grid[i:i+square_size, j:j+square_size]
        for i in range(0, height, square_size)
        for j in range(0, width, square_size)
    ]
    return np.array(squares)

# Function to split all grids sequentially (no parallelization)
def split_all_grids_sequential(npy_files, square_size=250, height=3500, width=7000):
    num_squares_per_grid = (height // square_size) * (width // square_size)
    num_files = len(npy_files)

    # Preallocate space for all the grids that will be split into squares
    all_squares = np.empty((num_files, num_squares_per_grid, square_size, square_size, 1), dtype=np.float32)

    # Sequentially process each file and store the result
    for idx, file in enumerate(npy_files):
        squares = split_single_grid(file, square_size, height, width)
        all_squares[idx] = squares

    return all_squares

# Assume npy_files, sequence_length, and prediction_horizon are already defined
square_size = 175
height, width = 3500, 7000  # Original image size
num_squares = 800

# Pre-split all grids into squares sequentially
all_squares = split_all_grids_sequential(npy_files, square_size=square_size, height=height, width=width)

# Now, treat each square as an independent sample
num_files = len(npy_files)
num_samples = (num_files - sequence_length - prediction_horizon + 1) * num_squares

# Pre-allocate arrays for input (X) and output (y) sequences
X = np.empty((num_samples, sequence_length, square_size, square_size, 1), dtype=np.float32)
y = np.empty((num_samples, prediction_horizon, square_size, square_size, 1), dtype=np.float32)

# Populate X and y arrays using pre-split squares
sample_idx = 0
for i in range(num_files - sequence_length - prediction_horizon + 1):
    for j in range(num_squares):
        # For each sample, take the sequence for that square over time
        x_sample = all_squares[i:i+sequence_length, j]
        X[sample_idx] = x_sample

        y_sample = all_squares[i+sequence_length:i+sequence_length+prediction_horizon, j]
        y[sample_idx] = y_sample

        sample_idx += 1

def remove_duplicates(X, y):
    """
    Removes duplicates from X and y where every pixel in the image sequences is the same.
    Only unique (X, y) pairs are retained.
    
    Args:
    - X: numpy array of shape (num_samples, sequence_length, image_width, image_height, 1)
    - y: numpy array of shape (num_samples, prediction_horizon, image_width, image_height, 1)
    
    Returns:
    - X_unique: numpy array of unique input sequences
    - y_unique: numpy array of corresponding unique output sequences
    """
    valid_indices = []
    num_samples = X.shape[0]
    threshold = 100000
    
    for i in range(num_samples):
        # Sum the pixel values for both X[i] and y[i]
        x_sum = np.sum(X[i])
        y_sum = np.sum(y[i])
        
        # If the sum is greater than or equal to the threshold, keep this sample
        if x_sum + y_sum >= threshold:
            valid_indices.append(i)
    
    # Select only the valid indices for X and y
    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]
    
    return X_filtered, y_filtered

X_unique, y_unique = remove_duplicates(X, y)

X_unique = X_unique / 80
y_unique = y_unique / 80

# Split data into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# First split: train and temp (which will later be split into validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_unique, y_unique, test_size=(1 - train_ratio)
)

# Second split: validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(test_ratio / (test_ratio + val_ratio))
)

# current model name
model_name = "model6_7.keras"

# Path to the model
model_path = root_dir + f"model/{model_name}"

# load model
model = load_model(model_path)

# Create an Adam optimizer with a custom learning rate
optimizer = AdamW()  # Set the learning rate here

# Compile the model with the optimizer
model.compile(loss='mae', optimizer=optimizer)

# define checkpoint to save the best model
cp = ModelCheckpoint(model_path, save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5)

# Train the model
history = model.fit(X_train, y_train, 
          batch_size=1, 
          epochs=20, 
          callbacks=[cp, early_stopping, reduce_lr],
          validation_data=(X_val, y_val))


val_loss = history.history['val_loss']  # List of validation losses for each epoch
average_val_loss = sum(val_loss) / len(val_loss)  # Compute the average val_loss
num_epochs = len(val_loss)  # Number of epochs is the length of the val_loss list

# Create the commit message including average val_loss and number of epochs
commit_message = f"Trained model: {model_name} with avg_val_loss: {average_val_loss:.4f} epochs: {num_epochs}"

commands = [
    "git config --global user.email 'tw@trevorwiebe.com'",
    "git config --global user.name 'Trevor Wiebe'",
    f"git add {model_path}", 
    f'git commit -m "{commit_message}"',
    "git push"
]

# Execute Git commands
for command in commands:
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}\nError: {e}")


# Refresh data for next run

# Constants for folder paths
FOLDER_A = root_dir + 'data/new_npy_files'
FOLDER_B = root_dir + 'data/npy_files'
FOLDER_C = root_dir + 'data/old_npy_files'

# Helper function to parse the file name into a datetime object
def parse_filename_to_datetime(filename):
    try:
        return datetime.strptime(filename, "%d-%m-%Y-%H%M")
    except ValueError:
        raise ValueError(f"File name {filename} does not follow the expected format")

# Step 1: Find the most recent file in Folder B
def get_last_file_date(folder_b):
    # List only files (not directories)
    files = [f for f in os.listdir(folder_b) if os.path.isfile(os.path.join(folder_b, f))]
    
    # Filter out any non-matching files (if needed)
    files = [f for f in files if f.endswith('.npy')]  # Adjust extension if needed
    if not files:
        raise FileNotFoundError("Folder B is empty or no valid files found")

    # Sort files based on parsed date from the file name
    files.sort(key=lambda f: parse_filename_to_datetime(f[:-4]))  # Adjusting for file extension
    
    # Get the latest file
    last_file = files[-1]
    last_file_date = parse_filename_to_datetime(last_file[:-4])  # Parse date from file name
    return last_file_date

# Step 2: Move all files from Folder B to Folder C
def move_files_from_b_to_c(folder_b, folder_c):
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)
    
    for file_name in os.listdir(folder_b):
        source = os.path.join(folder_b, file_name)
        
        # Ensure it's a file, not a directory
        if os.path.isfile(source):
            destination = os.path.join(folder_c, file_name)
            shutil.move(source, destination)
            print(f"Moved {file_name} from {folder_b} to {folder_c}")

# Step 3: Move 1 hour of files from Folder A to Folder B
def move_files_from_a_to_b(folder_a, folder_b, last_file_date):
    # Calculate the 1-hour window
    start_time = last_file_date
    end_time = start_time + timedelta(hours=1) + timedelta(minutes=2)

    # List and sort files in Folder A
    files = [f for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))]
    
    for file_name in sorted(files):
        try:
            # Extract the date from the file name
            file_date = parse_filename_to_datetime(file_name[:-4])  # Removing file extension if needed
            
            # Check if the file falls within the 1-hour window
            if start_time <= file_date < end_time:
                # Check if the file already exists in Folder B
                if os.path.exists(os.path.join(folder_b, file_name)):
                    raise FileExistsError(f"File {file_name} already exists in Folder B. Stopping process.")
                
                # Move file to Folder B
                shutil.move(os.path.join(folder_a, file_name), os.path.join(folder_b, file_name))
                print(f"Moved {file_name} from {folder_a} to {folder_b}")
        
        except ValueError:
            print(f"Skipping file {file_name}: Invalid date format")


def delete_files_in_folder(folder_path):
    # Use glob to find all files in the folder
    files = glob(os.path.join(folder_path, '*'))
    
    for file in files:
        try:
            if os.path.isfile(file):  # Check if it is a file (not a directory)
                os.remove(file)        # Delete the file
                print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

# Main script execution
if __name__ == "__main__":
    try:
        # Get the most recent file's date from Folder B
        last_file_date = get_last_file_date(FOLDER_B)
        print(f"Last file date in Folder B: {last_file_date}")

        # Step 2: Move all files from Folder B to Folder C
        move_files_from_b_to_c(FOLDER_B, FOLDER_C)

        # Step 3: Move 1-hour worth of files from Folder A to Folder B
        move_files_from_a_to_b(FOLDER_A, FOLDER_B, last_file_date)

        # Step 4: Delete all the files in Folder C
        delete_files_in_folder(FOLDER_C)

    except Exception as e:
        print(f"Error: {str(e)}")