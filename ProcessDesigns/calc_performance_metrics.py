# Script written by Sheares Toh and Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 


import os
import numpy as np

# Performance metrics to be calculated
# Pressure differences --> min(P_out) - max(P_min)
# Heat transfer rates (Qdot) --> sum(u_out * T_out) - sum(u_in * T_in)
# Mass imbalance --> sum(u_out) - sum(u_in)

#calc_fn = 'pressure-difference'
calc_fn = 'heat-transfer-rate'
#calc_fn = 'mass-imbalance'


def pressure_difference_arr(p_grid):

    #p_grid = input[:,:,:]

    pressure_differences = []

    for i in range(p_grid.shape[0]):
        # Obtain maximum difference between output (last) and input (first) columns
        max_difference = np.nanmax(p_grid[i, :, 0]) - np.nanmin(p_grid[i, :, -1])

        # Append the max_diff to the list
        pressure_differences.append(max_difference)

    # Convert list to vector
    pressure_differences = np.array(pressure_differences)

    return pressure_differences

   
def heat_transfer_rate_arr(u_grid, T_grid):

    #u_grid = input[:,:,:,2]
    #T_grid = input[:,:,:,5]

    heat_transfer_rate = [] #List

    for i in range(u_grid.shape[0]):

        #Obtain element wise products of first and last columns
        first = u_grid[i, :, 0] * T_grid[i, :, 0]
        last = u_grid[i, :, -1] * T_grid[i, :, -1]

        #Find difference between summation output (last) and summation input (first)
        diff = np.nansum(last) - np.nansum(first)

        # Append the diff to the list
        heat_transfer_rate.append(diff)

    # Convert the list to a one-dimensional NumPy array
    heat_transfer_rate = np.array(heat_transfer_rate)

    return heat_transfer_rate


def mass_imbal(u_grid):

    mass_balance = []

    for i in range(u_grid.shape[0]):

        # Calculate the summation of values in the first and last columns of u_grid
        first_column_sum = np.nansum(u_grid[i, :, 0])
        last_column_sum = np.nansum(u_grid[i, :, -1])
        diff_sum = first_column_sum - last_column_sum

        # Append the summations to the list
        mass_balance.append(diff_sum)

    mass_balance = np.array(mass_balance)
    print('Max Mass Imbalance:', np.max(mass_balance))

    return mass_balance

# The loop needs to be adjusted according to the provided files
for i in range(5, 35, 5):

    if calc_fn == 'heat-transfer-rate':

        u_filename = 'combined_u_' + str(i) + 'k.npy'
        u_grid = np.load(u_filename)
        T_filename = 'combined_T_' + str(i) + 'k.npy'
        T_grid = np.load(T_filename)

        output = heat_transfer_rate_arr(u_grid, T_grid)

        output_fn = 'Qdot_' + str(i) + 'k.npy'    

    elif calc_fn == 'pressure-difference':

        p_filename = 'combined_p_' + str(i) + 'k.npy'
        p_grid = np.load(p_filename)

        output = pressure_difference_arr(p_grid)

        output_fn = 'p_diff_' + str(i) + 'k.npy'    

    elif calc_fn == 'mass-imbalance':

        u_filename = 'combined_u_' + str(i) + 'k.npy'
        u_grid = np.load(u_filename)

        output = mass_imbalance(u_grid)

        output_fn = 'mass_bal_' + str(i) + 'k.npy'    

    np.save(output_fn,output)
