#!/bin/bash

src_folder="."
dst_folder="/dsi/gannot-lab1/projects_2024/Ilya_Tomer/OUTPUTS/OUTPUTS_TRAIN"
prefix="output"

# Create the destination directory if it doesn't exist
mkdir -p "$dst_folder"

# Copy and rename the folders
for folder in $(ls $src_folder | grep ^$prefix | sort); do
    cp -r "$src_folder/$folder" "$dst_folder"
    echo "Copied $src_folder/$folder to $dst_folder"
done
