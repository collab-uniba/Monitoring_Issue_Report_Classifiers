# Use the official NVIDIA PyTorch image with CUDA support
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables for Python, CUDA, and GPU support
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential wget curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install virtualenv
RUN python3 -m ensurepip && \
    pip install --no-cache-dir --upgrade pip setuptools virtualenv

# Install Python libraries
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122 && \
    pip install --no-cache-dir \
    transformers setfit

# Verify CUDA and GPU support
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available!'"

# Create a working directory
WORKDIR /app

# Copy your application code (if any) into the container
# ADD ./your_code /app/

# Set the default command (you can replace this with your own application entry point)
CMD ["python3"]
