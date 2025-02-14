# Use the latest PyTorch base image with CUDA support
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    nano \
    build-essential \
    g++ \
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    sox \
    && apt-get clean

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Run the main script when the container launches
CMD ["bash"]
