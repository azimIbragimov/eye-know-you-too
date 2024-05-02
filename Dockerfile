# Use the PyTorch image with CUDA and CUDNN support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install necessary tools
RUN apt-get update && apt-get install -y \
    git \
    wget

# Clone the desired repository
RUN git clone https://github.com/azimIbragimov/eye-know-you-too.git /eye-know-you-too

# Change working directory to the cloned repository
WORKDIR /eye-know-you-too

# Install Python dependencies
RUN pip install -r requirements.txt
