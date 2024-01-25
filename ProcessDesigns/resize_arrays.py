# Script written by Sheares Toh and Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import convolve2d
#from PIL import Image

### Create downsampled VOF and U,V,P,T for Geometry and Model Training

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

def downsampling_data(input, down_factor_x = 2, down_factor_y = 2):

    #down_factor = 2 # Downsampling is optional --> Set to scale = 1 if no downsampling
    kernel = np.ones((down_factor_x, down_factor_y))

    cur_sample = convolve2d(input, kernel, mode='valid')
    scaled_sample = cur_sample[::down_factor_x, ::down_factor_y] / (down_factor_x * down_factor_y)  # input argument for scaling
    #scaled_sample = scaled_sample.reshape(scaled_sample.shape[0], scaled_sample.shape[1], 1)

    return scaled_sample

def downsize_array(input_grid):

    output_grid = []

    for num in range(input_grid.shape[0]):

        short_grid = input_grid[num, 1:-1, 1:-1]

        # Setting for 2,4 is to reduce 512 x 256 down to 128 x 128
        downsampled_array = downsampling_data(short_grid,2,4)
        #print(downsampled_array.shape)

        output_grid.append(downsampled_array)

    output_grid = np.array(output_grid)
    
    return output_grid


def downsize_ls(ls_grid, to_save_folder):

    os.makedirs(to_save_folder, exist_ok=True)

    for num in range(ls_grid.shape[0]):

        #Change to VOF
        vof = ls_grid[num, 1:-1, 1:-1]

        for i in range(vof.shape[0]):
            for j in range(vof.shape[1]):
                if vof[i, j] <= 0:
                    vof[i, j] = 0
                else:
                    vof[i, j] = 1

        # Previous setting for 2,4 is to reduce 512 x 256 down to 128 x 128
        downsampled_array = downsampling_data(vof,1,1)
		#downsampled_array = downsampling_data(vof,2,4)
        #print(downsampled_array.shape)

        #old code to plot and visualize the VOF array
        #plt.imshow(downsampled_array, cmap=plt.cm.colors.ListedColormap(['white', 'green']), vmin=np.nanmin(downsampled_array),vmax=np.nanmax(downsampled_array), extent=[0, 1, 0, 1])
        #plt.imshow(downsampled_array, cmap='gray', vmin=np.nanmin(downsampled_array),vmax=np.nanmax(downsampled_array), extent=[0, 1, 0, 1])
        #plt.axis('off')

        # Define the output filename and path
        filename = f'design_{num}.png'
        output_path = os.path.join(to_save_folder, filename)
        #plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        #plt.close()
        plt.imsave(output_path, downsampled_array, cmap='gray', vmin=np.nanmin(downsampled_array),vmax=np.nanmax(downsampled_array))

        #print(f'Done processing {file_name}')
    print('Done processing')

##### Downsample and save the VOF array --> Fed into downstream models
input_grid = np.load('combined_ls_all.npy')
downsize_ls(input_grid, 'output_vof_images') #Please change source and destination directories here

##### Downsample the numerical arrays
#param = 'u'
#param = 'v'
#param = 'p'
#param = 'T'

#input_grid = np.load('combined_' + param + '_all.npy')
#output_grid = downsize_array(input_grid)
#print(output_grid.shape)
#np.save('combined_' + param + '_all_128x128.npy', output_grid)