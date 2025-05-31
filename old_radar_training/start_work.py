import multiprocessing
import subprocess
import os
import time
import sys

# Define the two scripts
root_dir = os.getcwd()
download_script = root_dir + "/download.py"
process_script = root_dir + "/train.py"

def run_script(script_name):
    # Start a script as a subprocess
    process = subprocess.Popen(["python3", script_name])
    process.wait()  # Wait for the script to finish
    return process.returncode

def start_scripts():
    # Start download.py and train.py in parallel
    download_process = multiprocessing.Process(target=run_script, args=(download_script,))
    train_process = multiprocessing.Process(target=run_script, args=(process_script,))

    # Start both processes
    download_process.start()
    train_process.start()

    # Wait for both processes to finish
    download_process.join()
    train_process.join()

    print("Both scripts finished. Restarting start_work.py...")
    # restart_script()

def restart_script():
    # Recursively restart start_work.py
    time.sleep(2)  # Optional delay before restarting
    os.execv(sys.executable, [sys.executable] + [os.path.abspath(__file__)] + sys.argv[1:])

if __name__ == "__main__":
    start_scripts()