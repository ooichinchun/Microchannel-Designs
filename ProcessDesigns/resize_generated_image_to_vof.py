# Script written by Sheares Toh and Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

"""### Re-create VOF for Numerical Simulation"""

#Variables to adjust

dimension = (256, 512) #Resolution of design space

rect_width = 2.0
rect_height = 1.0

# Create a grid
x = np.linspace(0 + rect_width/(dimension[1]*2), rect_width - rect_width/(dimension[1]*2), dimension[1])
y = np.linspace(0 + rect_height/(dimension[0]*2), rect_height - rect_height/(dimension[0]*2), dimension[0])
X, Y = np.meshgrid(x, y)
#grid_ls = np.column_stack((X.ravel(), Y.ravel()))
#print(X.shape)

def generated_to_vof(folder, to_save_folder):

    os.makedirs(to_save_folder, exist_ok=True)

    file_list = [f for f in os.listdir(folder)]
    file_list.sort()  # Sort case numbers in ascending order

    for num, file_name in enumerate(file_list):
        
		file_path = os.path.join(folder, file_name)
        
		#try:
        #    explore = np.genfromtxt(file_path, skip_header=1)
        #except FileNotFoundError:
        #    print(f"File not found: {file_path}")
        #    continue

        # Define the input filename and path
        filename = f'design_{num}.png'
        input_path = os.path.join(to_save_folder, filename)

        with open(input_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("L")

        # Define the output filename and path for the VOF image and Array (for simulation

        arr = pil_image.resize(dimension, resample=Image.BICUBIC)
        new_filename = f'design_resize_{num}.png'
        output_path = os.path.join(to_save_folder, new_filename)
        arr.save(output_path)

        arr = np.array(arr)
        pts_ls = np.column_stack((X.ravel(), Y.ravel(), arr.ravel()))
        filename_pts = to_save_folder + '/input_' + str(num).zfill(5)  + '.dat' #Level set value for each coordinate
        header = "x y ls"
        np.savetxt(filename_pts, pts_ls, delimiter=',', header=header, comments='')

        print(f'Done processing {file_name}')

# Please change source and destination directories here
generated_to_vof('generated_vof_images', 'output_vof')
