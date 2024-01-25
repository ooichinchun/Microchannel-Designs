# Script written by Sheares Toh and Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import os
import numpy as np
import re
from PIL import Image

# Please change directory for file saving here
#root_dir = '/content/drive/MyDrive/main/datas/' 
#os.chdir(root_dir)

# Please change source and destination directories here
folder_path = '/data/ooicc/microfluidic-channels-ph/combined_arrays/output_vof_images'  # Replace with the actual path to your folder
output_folder = '/data/ooicc/microfluidic-channels-ph/combined_arrays/inlet_pipe_counts'
os.makedirs(output_folder, exist_ok=True)

# Initialize counters for each file type
image_count = 0

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith('design'):
        image_count += 1

# Print the counts
print(f"Total number of images: {image_count}")


###Extract inlet pipe number for each image and save as numpy array

# Extract case number from file name
def extract_case_number(file_name):
    
    match = re.search(r'_(\d+)\.png', file_name)
	
    return int(match.group(1)) if match else -1


def inlet_pipes_counter(folder, to_save_folder):
    file_list = [f for f in os.listdir(folder)]
    file_list.sort(key=extract_case_number)  # Sort case numbers in ascending order

    for num, file_name in enumerate(file_list):
        file_path = os.path.join(folder, file_name)

        # Open the image and convert to grayscale
        with open(file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image = pil_image.convert("L")

        # Resize the image
        resized_image = pil_image.resize((64,64), resample=Image.BICUBIC) #Resize to (64,64) to match diffusion model input

        # Convert the resized image to binary image
        image_array = np.array(resized_image) > 128  # Thresholding to get binary image (1 and 0)

        # Count number of times switching from 0 to 1 and vice versa
        switch_count = int(np.round(np.sum(image_array[:-1, 0] != image_array[1:, 0]) / 2))

        result_array = np.full((64, 64, 1), switch_count, dtype=int)
        filename = f'num_pipe_{num}.npy'
        output_path = os.path.join(to_save_folder, filename)
        np.save(output_path, result_array)

        # Print processing information
        print(f'Done processing {file_name}. Value switch count: {switch_count}')


inlet_pipes_counter(folder_path, output_folder)

# Initialize counters for each file type
pipe_count = 0

# Iterate through the generated files in the folder
for filename in os.listdir(output_folder):
    if filename.startswith('num'):
        pipe_count += 1

# To double check number of files in training folder
# Print the counts - Both counts should match
print(f"Total number of images: {image_count}")
print(f"Total number of counts: {pipe_count}")
