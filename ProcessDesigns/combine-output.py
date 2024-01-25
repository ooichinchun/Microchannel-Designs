# Script written by Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import os
import numpy as np

#root_dir = ('/home/ooicc/microfluidic-channels-ph/output/')
#os.chdir(root_dir)

def create_array(folder):

    file_list = [f for f in os.listdir(folder)]
    file_list.sort() #Sort case numbers in ascending order

    #combined_array = []
    combined_x = []
    combined_y = []
    combined_u = []
    combined_v = []
    combined_p = []
    combined_T = []
    combined_ls = []
                            
    mass_balance = []

    for file_name in file_list:
        explore = np.genfromtxt(folder + '/' + file_name, skip_header=1)

        #Store individual variables
        x = explore[:, 0]
        y = explore[:, 1]
        u = explore[:, 2]
        v = explore[:, 3]
        p = explore[:, 4]
        T = explore[:, 5]
        ls = explore[:, 6]

        #Find shape of grid (Should be 514, 258)
        num_x_points = len(np.unique(x))
        num_y_points = len(np.unique(y))

        #Replace rows to None so as to mask plot background
        indices = np.where(ls > 0)
        u[indices] = None
        v[indices] = None
        p[indices] = None
        T[indices] = None


        x_grid = x.reshape((num_y_points, num_x_points))
        y_grid = y.reshape((num_y_points, num_x_points))

        u_grid = u.reshape((num_y_points, num_x_points))
        v_grid = v.reshape((num_y_points, num_x_points))
        p_grid = p.reshape((num_y_points, num_x_points))

        T_grid = T.reshape((num_y_points, num_x_points))

        ls_grid = ls.reshape((num_y_points, num_x_points))

        #combine = np.stack((x_grid, y_grid, u_grid, v_grid, p_grid, T_grid, ls_grid), axis=2)

        #combined_array.append(combine)
        combined_x.append(x_grid)
        combined_y.append(y_grid)
        combined_u.append(u_grid)
        combined_v.append(v_grid)
        combined_p.append(p_grid)
        combined_T.append(T_grid)
        combined_ls.append(ls_grid)                

        # Calculate the summation of values in the first and last columns of u_grid
        first_column_sum = np.nansum(u_grid[:, 0])
        last_column_sum = np.nansum(u_grid[:, -1])
        diff_sum = first_column_sum - last_column_sum

        # Append the summations to the list
        mass_balance.append(diff_sum)

    mass_balance = np.array(mass_balance)
    print('Max Mass Imbalance:', np.max(mass_balance))

    #combined_array = np.array(combined_array)
    #print(combined_array.shape)
    combined_x = np.array(combined_x)
    print(combined_x.shape)
    combined_y = np.array(combined_y)
    print(combined_y.shape)
    combined_u = np.array(combined_u)
    print(combined_u.shape)
    combined_v = np.array(combined_v)
    print(combined_v.shape)
    combined_p = np.array(combined_p)
    print(combined_p.shape)
    combined_T = np.array(combined_T)
    print(combined_T.shape)
    combined_ls = np.array(combined_ls)
    print(combined_ls.shape)


    np.save('combined_x_20k.npy', combined_x)
    np.save('combined_y_20k.npy', combined_y)
    np.save('combined_u_20k.npy', combined_u)
    np.save('combined_v_20k.npy', combined_v)
    np.save('combined_p_20k.npy', combined_p)
    np.save('combined_T_20k.npy', combined_T)
    np.save('combined_ls_20k.npy', combined_ls)
    #np.save('mass_bal_20k.npy', mass_balance)

    return 0

# Folder cna be inserted via a loop, but the npy array output needs to be changed
folder_name = 'output-2-2'
create_array(folder_name)


