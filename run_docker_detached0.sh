#!/bin/bash
# Change to executable to run: chmod +x run_docker.sh
# Build the Docker image
docker build -t audio-db .

# Run the Docker container with volume mounts, GPU support, and execute the Python script
docker run --gpus all -d \
  --shm-size=2.5g \
  -v /dsi/gannot-lab1/datasets/VCTK:/data \
  -v /home/dsi/lipovai/clonedProject/RIRnewv:/app/ \
  -v /dsi/gannot-lab1/projects_2024/Ilya_Tomer/OUTPUTS/OUTPUTS_TRAIN:/app/OUTPUTS_DSI/OUTPUTS_TRAIN \
  -v /dsi/gannot-lab1/projects_2024/Ilya_Tomer/OUTPUTS/OUTPUTS_TEST:/app/OUTPUTS_DSI/OUTPUTS_TEST \
  -v /dsi/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN:/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN \
  --name "${USER}_test_detached_partition0" \
  audio-db python /app/generate_data.py --output_dir_train /app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/partition0
