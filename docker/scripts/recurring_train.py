from datetime import timedelta, datetime, timezone
import bs4 as bs
import os, shutil
import requests
import pandas as pd
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import LinearSegmentedColormap, Normalize
import random
from contextlib import redirect_stdout
from datetime import timedelta, datetime, timezone
import tensorflow as tf
import io
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, BatchNormalization, Input, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import AdamW
import subprocess


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

#
# Global variables
#

# Important variables that determine performance
# When any of these variables are changed, keep track of old variable to revert to if poor performance
root_dir = os.path.dirname(os.getcwd()) + "/"
data_dir = root_dir + "data/"

times_to_train = 10

sequence_length = 5 # Number of .npy files to look back on during training
prediction_horizon = 5 # Number of .npy files to predict

# Width and height of each square in the grid
square_size = 50

# Required total of dbZ in sample size
required_value_threshold = 250000

batch_size = 4

early_stopping_patience = 15

reduce_lr_patience = 5

training_epochs = 100

# Define the dimensions of the radar images.  This should never change
height, width = 3500, 7000

training_id = "recurring_train_3"



#
# Helper download functions
#

# Generates a random time window within the specified start and end dates.
def random_time_window(start_date: datetime, end_date: datetime, minutes: int):
    """
    Returns a random (start, end) datetime tuple within [start_date, end_date],
    where the difference between start and end is exactly `minutes`.
    """
    delta = end_date - start_date
    max_start = delta.total_seconds() - minutes * 60
    if max_start < 0:
        raise ValueError("Time window is too large for the given range.")
    random_offset = random.uniform(0, max_start)
    random_start = start_date + timedelta(seconds=random_offset)
    random_end = random_start + timedelta(minutes=minutes)
    return random_start, random_end

# Downloads files from a specified URL to a local folder, filtering by time and granularity.
def download_files(folder_url, destination_folder, start_time, end_time, granularity_minutes):
    """
    Downloads files from folder_url to destination_folder within the time window,
    only keeping files at intervals of granularity_minutes.
    """

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

        npy_files = data.find_all("a", href=lambda href: href and href.endswith(f'.npy'))

        # Filter files based on time range and granularity
        filtered_files = []
        last_selected_time = None
        for file_name in npy_files:
            file_type_filename = file_name['href'].split('/')[-1]
            try:
                # Parse the filename to extract the creation or modification time
                file_time_str = file_type_filename.split('.')[0]  # Extract '20250605-053644'
                file_time = datetime.strptime(file_time_str, '%Y%m%d-%H%M%S').replace(tzinfo=timezone.utc)
                # Ignore seconds and microseconds for granularity comparison
                file_time_rounded = file_time.replace(second=0, microsecond=0)
                if start_time <= file_time <= end_time:
                    if (last_selected_time is None or 
                        (file_time_rounded - last_selected_time).total_seconds() >= granularity_minutes * 60):
                        filtered_files.append(file_name)
                        last_selected_time = file_time_rounded
            except ValueError as e:
                # Handle parsing errors (e.g., invalid filename format)
                print(f"Error parsing filename: {file_type_filename}, {e.args[0]}")
        if len(filtered_files) == 0:
            print("No files matching that criteria")

        for file_name in filtered_files:
            file_type_url = file_name['href']  # Get the file URL
            file_type_filename = file_type_url.split('/')[-1]  # Extract the filename
            link = folder_url + file_type_filename
            destination_link = os.path.join(destination_folder, file_type_filename)

            response = requests.get(link)
            with open(destination_link, 'wb') as f:
                f.write(response.content)

        print("Downloading finished, outcome unknown.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading files: {e}")

def fetch_fresh_data():

    print("Fetching fresh data...")

    # The start time may need to be adjusted based on the actual data availability.
    data_start_time = datetime(2025, 7, 22, 3, 6, tzinfo=timezone.utc)
    data_end_time = datetime.now(timezone.utc) - timedelta(minutes=5)

    # granularity_minutes describes the time interval between files to be downloaded.  
    # The minimum value that can be used is 2 minutes, as the data is collected every 2 minutes.
    granularity_minutes = 4

    # Determine sample time span based on granularity_minutes to ensure that 
    # the training, validation, and test sets are appropriately sized.
    train_sample_span = granularity_minutes * 40
    validation_sample_span = granularity_minutes * 11
    test_sample_span = granularity_minutes * 11

    # Get a random time window for training, validation, and test data, based on the defined sample spans
    train_data_start_time, train_data_end_time = random_time_window(data_start_time, data_end_time, train_sample_span)
    validation_data_start_time, validation_data_end_time = random_time_window(data_start_time, data_end_time, validation_sample_span)
    test_data_start_time, test_data_end_time = random_time_window(data_start_time, data_end_time, test_sample_span)

    #
    # Initiate download of files
    #

    #remote server URL where the .npy files are hosted
    folder_url = f'http://66.213.177.43/v2_npy_files/'

    train_destination_folder = data_dir + f'train_npy_files'
    download_files(folder_url, train_destination_folder, train_data_start_time, train_data_end_time, granularity_minutes)

    validation_destination_folder = data_dir + f'validation_npy_files'
    download_files(folder_url, validation_destination_folder, validation_data_start_time, validation_data_end_time, granularity_minutes)

    test_destination_folder = data_dir + f'test_npy_files'
    download_files(folder_url, test_destination_folder, test_data_start_time, test_data_end_time, granularity_minutes)

    # 
    # Make sure all the data is between 0 and 80
    # 

    # Load the .npy files into a list
    npy_files = sorted(glob(data_dir + 'train_npy_files/*.npy'))
    npy_files += sorted(glob(data_dir + 'validation_npy_files/*.npy'))
    npy_files += sorted(glob(data_dir + 'test_npy_files/*.npy'))

    # Iterate through the list and modify the arrays
    for i in enumerate(npy_files):
        array = np.load(i[1])
        
        array[array < 20] = 0
        array[array > 80] = 80

        np.save(i[1], array)


#
# Helper training setup
#

def compute_total_squares(square_size):
    if height % square_size != 0 or width % square_size != 0:
        raise ValueError(f"{square_size} is not a common factor of both {height} and {width}.")

    rows = height // square_size
    columns = width // square_size
    return rows * columns

# Assume npy_files, sequence_length, and prediction_horizon are already defined
num_squares = compute_total_squares(square_size)

def split_single_grid(file, square_size, height, width):
    grid = np.load(file).reshape(height, width, 1)
    squares = [
        grid[i:i+square_size, j:j+square_size]
        for i in range(0, height, square_size)
        for j in range(0, width, square_size)
    ]
    return np.array(squares)

# Function to split all grids sequentially (no parallelization)
def split_all_grids_sequential(npy_files, square_size, height, width):
    num_squares_per_grid = (height // square_size) * (width // square_size)
    num_files = len(npy_files)

    # Preallocate space for all the grids that will be split into squares
    all_squares = np.empty((num_files, num_squares_per_grid, square_size, square_size, 1), dtype=np.float32)

    # Sequentially process each file and store the result
    for idx, file in enumerate(npy_files):
        squares = split_single_grid(file, square_size, height, width)
        all_squares[idx] = squares

    return all_squares


def split_into_x_y_squares(all_squares, sequence_length, prediction_horizon):
    num_files, num_squares, square_size, _, _ = all_squares.shape
    num_samples = (num_files - sequence_length - prediction_horizon + 1) * num_squares

    print(f"Total samples: {num_samples}, Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}, Number of squares: {num_squares}")

    # Pre-allocate arrays for input (X) and output (y) sequences
    X = np.empty((num_samples, sequence_length, square_size, square_size, 1), dtype=np.float32)
    y = np.empty((num_samples, prediction_horizon, square_size, square_size, 1), dtype=np.float32)

    sample_idx = 0
    for i in range(num_files - sequence_length - prediction_horizon + 1):
        for j in range(num_squares):
            # For each sample, take the sequence for that square over time
            x_sample = all_squares[i:i+sequence_length, j]
            X[sample_idx] = x_sample

            y_sample = all_squares[i+sequence_length:i+sequence_length+prediction_horizon, j]
            y[sample_idx] = y_sample

            sample_idx += 1

    return X, y

def remove_duplicates_by_max_value(X, y):
    valid_indices = []
    num_samples = X.shape[0]
    
    for i in range(num_samples):
        # Sum the pixel values for both X[i] and y[i]
        x_sum = np.sum(X[i])
        y_sum = np.sum(y[i])
        
        # If the sum is greater than or equal to the threshold, keep this sample
        if x_sum + y_sum >= required_value_threshold:
            valid_indices.append(i)
    
    # Select only the valid indices for X and y
    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]
    
    return X_filtered, y_filtered

def get_model_memory_usage(batch_size, model):
    features_mem = 0  # Initialize memory for features
    float_bytes = 4.0  # Float32 uses 4 bytes
    
    for layer in model.layers:
        # Use layer.output.shape to get the output shape instead of output_shape
        out_shape = layer.output.shape
        
        # Remove the batch size dimension (out_shape[0]) and None (which represents the batch dimension)
        out_shape = [dim for dim in out_shape if dim is not None]
        
        # Multiply all output shape dimensions to calculate the number of elements per layer
        single_layer_mem = 1
        for s in out_shape:
            single_layer_mem *= s
            
        # Convert to memory (in bytes and MB)
        single_layer_mem_float = single_layer_mem * float_bytes  # Multiply by 4 bytes (float32)
        single_layer_mem_MB = single_layer_mem_float / (1024 ** 2)  # Convert to MB
        
        print(f"Memory for layer {layer.name} with output shape {out_shape} is: {single_layer_mem_MB:.2f} MB")
        
        features_mem += single_layer_mem_MB  # Accumulate total feature memory
    
    # Calculate Parameter memory
    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes) / (1024 ** 2)
    
    print("_________________________________________")
    print(f"Memory for features in MB is: {features_mem * batch_size:.2f} MB")
    print(f"Memory for parameters in MB is: {parameter_mem_MB:.2f} MB")

    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB
    total_memory_GB = total_memory_MB / 1024  # Convert to GB
    
    return total_memory_GB

def save_visible_results(known_frames, should_be_frames, output_filename="prediction_output.png"):
    colors = [
        (0, 0, 0),         # White for values 0-15 (no precipitation or very light)
        (0, 0.7, 0),       # Green (light precipitation)
        (1, 1, 0),         # Yellow (moderate precipitation)
        (1, 0.65, 0),      # Orange (heavy precipitation)
        (1, 0, 0),         # Red (very heavy precipitation)
        (0.6, 0, 0.6)      # Purple (extreme precipitation)
    ]

    breakpoints = [0.0, .15/1.0, .40/1.0, .60/1.0, .70/1.0, 1.0]
    radar_cmap = LinearSegmentedColormap.from_list('radar', colors, N=80)
    norm = Normalize(vmin=0, vmax=1)

    predicted_frames = model.predict(np.expand_dims(known_frames, axis=0))
    predicted_frames = np.squeeze(predicted_frames, axis=0)

    fig, axes = plt.subplots(3, 5, figsize=(10, 7))

    for idx in range(5):
        ax = axes[0, idx]
        ax.imshow(np.squeeze(known_frames[idx]), cmap=radar_cmap, norm=norm)
        ax.set_title(f"Original {idx + 1}")
        ax.axis("off")

    for idx in range(5):
        ax = axes[1, idx]
        ax.imshow(np.squeeze(predicted_frames[idx]), cmap=radar_cmap, norm=norm)
        ax.set_title(f"Predicted {idx + 6}")
        ax.axis("off")

    for idx in range(5):
        ax = axes[2, idx]
        ax.imshow(np.squeeze(should_be_frames[idx]), cmap=radar_cmap, norm=norm)
        ax.set_title(f"Actual {idx + 6}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)

#
# Recurring training loop
#

fetch_fresh_data()

for i in range(times_to_train):

    print(f"Training iteration {i + 1} of {times_to_train}")

    # Load the .npy files into a list
    train_npy_files = sorted(glob(data_dir + 'train_npy_files/*.npy'))
    test_npy_files = sorted(glob(data_dir + 'test_npy_files/*.npy'))
    validation_npy_files = sorted(glob(data_dir + 'validation_npy_files/*.npy'))

    # Pre-split all grids into squares sequentially
    train_all_squares = split_all_grids_sequential(train_npy_files, square_size=square_size, height=height, width=width)
    test_all_squares = split_all_grids_sequential(test_npy_files, square_size=square_size, height=height, width=width)
    validation_all_squares = split_all_grids_sequential(validation_npy_files, square_size=square_size, height=height, width=width)

    # Split the pre-split squares into sequences for training, testing, and validation
    X_train, y_train = split_into_x_y_squares(train_all_squares, sequence_length=sequence_length, prediction_horizon=prediction_horizon)
    X_test, y_test = split_into_x_y_squares(test_all_squares, sequence_length=sequence_length, prediction_horizon=prediction_horizon)
    X_validation, y_validation = split_into_x_y_squares(validation_all_squares, sequence_length=sequence_length, prediction_horizon=prediction_horizon)

    X_train_unique, y_train_unique = remove_duplicates_by_max_value(X_train, y_train)
    X_test_unique, y_test_unique = remove_duplicates_by_max_value(X_test, y_test)
    X_validation_unique, y_validation_unique = remove_duplicates_by_max_value(X_validation, y_validation)

    # Normalize the data
    X_train_normalized = X_train_unique / 80
    y_train_normalized = y_train_unique / 80

    X_test_normalized = X_test_unique / 80
    y_test_normalized = y_test_unique / 80

    X_validation_normalized = X_validation_unique / 80
    y_validation_normalized = y_validation_unique / 80

    model_name = f"model_{training_id}.keras"
    model_path = root_dir + f"model/{model_name}"

    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
    except (OSError, IOError, ValueError):
        print(f"Model not found at {model_path}, creating a new one.")

        channels = 1  # Reflectivity is your feature, so 1 channel

        # Define the model using an Input layer for the input shape
        model = Sequential()

        # Add Input Layer
        model.add(Input(shape=(sequence_length, square_size, square_size, channels))) 

        # First ConvLSTM2D layer with return_sequences=True
        model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Second ConvLSTM2D layer with return_sequences=True to return all frames
        model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Third ConvLSTM2D layer
        model.add(ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Replace Conv3D with Conv2D to predict the next frame(s)
        model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='linear', padding='same'))

        optimizer = AdamW(learning_rate=0.001)

        # Compile the model
        model.compile(loss='mae', optimizer=optimizer)


    cp = ModelCheckpoint(root_dir + f"model/{training_id}/{model_name}", save_best_only=True)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", 
        patience=reduce_lr_patience,
        factor=0.5,
        min_lr=1e-7,
        min_delta=0.0001,
        cooldown=2,
        verbose=1
    )

    # Just before starting training, kick off a download for the next training iteration
    # Start fetching fresh data in a separate thread
    if(i + 1) < times_to_train:
        fetch_thread = threading.Thread(target=fetch_fresh_data)
        fetch_thread.start()

    # Train the model
    history = model.fit(X_train_normalized, y_train_normalized, 
            batch_size=batch_size, 
            epochs=training_epochs, 
            callbacks=[cp, early_stopping, reduce_lr],
            validation_data=(X_validation_normalized, y_validation_normalized))
    
    # Wait for the fetch thread to finish before proceeding
    if(i + 1) < times_to_train:
        fetch_thread.join()
    
    # Path to the model saved by ModelCheckpoint
    model_path = root_dir + f"model/{training_id}/{model_name}"

    val_loss = history.history['val_loss']  # List of validation losses for each epoch
    average_val_loss = sum(val_loss) / len(val_loss)  # Compute the average val_loss
    num_epochs = len(val_loss)  # Number of epochs is the length of the val_loss list

    # Create the message including average val_loss and number of epochs
    training_results_message = f"Trained model: {model_name} with avg_val_loss: {average_val_loss:.4f} epochs: {num_epochs}"

    mem_output = io.StringIO()
    with redirect_stdout(mem_output):
        mem_for_my_model = get_model_memory_usage(batch_size, model)
    mem_output_lines = mem_output.getvalue().splitlines()

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    results_lines = [
        f"###### TRAINING RESULTS FOR {i} OF {times_to_train} ######",
        ""
        "## What I tried new ##",
        "- Set the starting learning rate back to 0.001",
        f"Model trained on: {current_time}",
        "",
        training_results_message,
        "",
        "## Variables ##",
        f"sequence_length: {sequence_length}",
        f"prediction_horizon: {prediction_horizon}",
        f"square_size: {square_size}",
        f"batch_size: {batch_size}",
        f"early_stopping_patience: {early_stopping_patience}",
        f"reduce_lr_patience: {reduce_lr_patience}",
        f"training_epochs: {training_epochs}",
        f"required_value_threshold: {required_value_threshold}",
        "",
        "## Data Shape ##",
        f"X_train: {X_train_normalized.shape}",
        f"y_train: {y_train_normalized.shape}",
        f"X_test: {X_test_normalized.shape}",
        f"y_test: {y_test_normalized.shape}",
        f"X_validation: {X_validation_normalized.shape}",
        f"y_validation: {y_validation_normalized.shape}",
        "",
        "## Memory Needed Summary ##",
        f"Memory required for model: {mem_for_my_model:.2f} GB",
        *mem_output_lines,
        "\n \n"
    ]

    results_filename = f"results_{training_id}.txt"
    results_filepath = os.path.join(root_dir, f"model/{training_id}", results_filename)

    with open(results_filepath, "a") as f:  # Use "a" for append mode
        f.write("\n".join(results_lines))
        f.write("\n")

    save_dir = os.path.join(root_dir, f"model/{training_id}/train_imgs")
    os.makedirs(save_dir, exist_ok=True)
    trainDir = os.path.join(save_dir, f"img_{i}.png")
    trainChoice = np.random.choice(range(len(X_train_normalized)), size=1)[0]
    trainFrames = X_train_normalized[trainChoice]
    actual_train_frames = y_train_normalized[trainChoice]
    save_visible_results(trainFrames, actual_train_frames, trainDir)

    save_dir = os.path.join(root_dir, f"model/{training_id}/val_imgs")
    os.makedirs(save_dir, exist_ok=True)
    valDir = os.path.join(save_dir, f"img_{i}.png")
    valChoice = np.random.choice(range(len(X_validation_normalized)), size=1)[0]
    val_frames = X_validation_normalized[valChoice]
    val_actual_frames = y_validation_normalized[valChoice]
    save_visible_results(val_frames, val_actual_frames, valDir)

    save_dir = os.path.join(root_dir, f"model/{training_id}/test_imgs")
    os.makedirs(save_dir, exist_ok=True)
    testDir = os.path.join(save_dir, f"img_{i}.png")
    testChoice = np.random.choice(range(len(X_test_normalized)), size=1)[0]
    test_frames = X_test_normalized[testChoice]
    actual_test_frames = y_test_normalized[testChoice]
    save_visible_results(test_frames, actual_test_frames, testDir)
