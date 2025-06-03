# Use NVIDIA's official TensorFlow image optimized for 2025
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

# Set metadata for the image
LABEL maintainer="tw@trevorwiebe.com"
LABEL description="TensorFlow GPU training environment for Vast.ai"
LABEL version="latest"

# Set environment variables for Vast.ai compatibility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure TensorFlow cache directories for optimal performance
ENV TORCH_HOME=/workspace/cache/torch
ENV HF_HOME=/workspace/cache/huggingface  
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers

# Update system packages and install essential dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tree \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories for ML libraries
RUN mkdir -p /workspace/cache/{torch,huggingface,transformers}

# Copy requirements file first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create workspace directory
WORKDIR /workspace

# Copy your training scripts into the container
COPY scripts/ /workspace/scripts/

# Set the working directory for script execution
WORKDIR /workspace

# Configure Jupyter for Vast.ai
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.allow_origin = '*'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py

# Default command - can be overridden by Vast.ai launch configuration
# For Jupyter mode on Vast.ai, use their Jupyter launch option instead
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]