#!/usr/bin/env python3
"""Debug GPU availability and TensorFlow configuration"""

import subprocess
import os
import sys

print("=" * 80)
print("GPU DEBUGGING INFORMATION")
print("=" * 80)

# Check NVIDIA driver
print("\n1. NVIDIA-SMI Output:")
print("-" * 40)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: nvidia-smi failed with return code {result.returncode}")
        print(result.stderr)
except Exception as e:
    print(f"ERROR: Could not run nvidia-smi: {e}")

# Check CUDA environment variables
print("\n2. CUDA Environment Variables:")
print("-" * 40)
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'LD_LIBRARY_PATH', 'PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"{var}: {value}")

# Check CUDA devices
print("\n3. CUDA Device Files:")
print("-" * 40)
cuda_devices = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm']
for device in cuda_devices:
    if os.path.exists(device):
        print(f"✓ {device} exists")
    else:
        print(f"✗ {device} NOT FOUND")

# Import TensorFlow with detailed logging
print("\n4. TensorFlow GPU Configuration:")
print("-" * 40)

# Set environment variables for debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages

try:
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")
    
    # List physical devices
    print("\nPhysical devices:")
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"  - {device}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nNumber of GPUs available: {len(gpus)}")
    
    if len(gpus) > 0:
        print("\nGPU Details:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                for key, value in details.items():
                    print(f"    {key}: {value}")
            except:
                pass
    
    # Try to create a simple tensor on GPU
    print("\n5. Testing GPU Computation:")
    print("-" * 40)
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✓ GPU computation successful: {c.numpy()}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        
except Exception as e:
    print(f"ERROR importing or using TensorFlow: {e}")
    import traceback
    traceback.print_exc()

# Check container runtime
print("\n6. Container Runtime Info:")
print("-" * 40)
print(f"Container runtime: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

# Vast.ai specific checks
print("\n7. Vast.ai Environment:")
print("-" * 40)
vast_vars = ['CONTAINER_ID', 'GPU_COUNT', 'PUBLIC_IPADDR', 'VAST_CONTAINERLABEL']
for var in vast_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"{var}: {value}")

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)