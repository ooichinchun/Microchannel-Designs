# Script written by Sheares Toh and Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pickle
import csv
import os

##Set up directory

#root_dir = '/content/drive/MyDrive/NTU/Programmes/FYP/IHPC/Test/' #Please change directory for file saving here
#os.chdir(root_dir)

##Existing definitions

class Pipe:
    def __init__(self, entry_point, exit_point, path, thickness, control_pts):
        self.entry_point = entry_point
        self.exit_point = exit_point
        self.path = path
        self.thickness = thickness
        self.control_pts = control_pts
		
# Analyze number of pipes in all the cases

def create_csv(folder):
    data = []
    file_list = [f for f in os.listdir(folder)]
    file_list.sort() #Sort case numbers in ascending order

    for file_name in file_list:
        case_number = file_name.split('_')[1].split('.')[0]
        with open('pipes/' + file_name, 'rb') as file:
            pipe_list = pickle.load(file)
        num_pipes = len(pipe_list)
        data.append((case_number, num_pipes))

    with open('cases_num_pipes.csv', 'w', newline='') as file:
        file.write('case_number, num_pipes\n')

        for row in data:
            file.write(f'{row[0]},{row[1]}\n')

def plot_output_full(folder):
    os.makedirs('output/u_velocity', exist_ok=True)
    os.makedirs('output/v_velocity', exist_ok=True)
    os.makedirs('output/pressure', exist_ok=True)
    os.makedirs('output/temperature', exist_ok=True)
    os.makedirs('output/levelset', exist_ok=True)

    file_list = [f for f in os.listdir(folder)]
    file_list.sort() #Sort case numbers in ascending order

    for file_name in file_list:
        case_number = file_name.split('_')[1].split('.')[0]
        explore = np.genfromtxt('output_dat/'+file_name, skip_header=1)

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

        u_grid = u.reshape((num_y_points, num_x_points))
        v_grid = v.reshape((num_y_points, num_x_points))
        p_grid = p.reshape((num_y_points, num_x_points))
        T_grid = T.reshape((num_y_points, num_x_points))
        ls_grid = ls.reshape((num_y_points, num_x_points))

        grids = [u_grid, v_grid, p_grid, T_grid, ls_grid]
        labels = ['u_velocity', 'v_velocity', 'pressure', 'temperature', 'levelset']

        for grid, label in zip(grids, labels):
            plt.figure()
            plt.imshow(grid, cmap='RdYlBu_r', vmin=np.min(grid), vmax=np.max(grid), extent=[0, 2, 0, 1])
            plt.colorbar(label=label)
            plt.xlabel('X')
            plt.ylabel('Y')
            filename = 'output/' + label + '/' + label + ' _' + str(case_number).zfill(5) + ".png"
            plt.savefig(filename)
            plt.close()

# Plots to visualize simulation output with a mask to only show fluid regions
def plot_output(folder):
    os.makedirs('output/u_velocity', exist_ok=True)
    os.makedirs('output/v_velocity', exist_ok=True)
    os.makedirs('output/pressure', exist_ok=True)
    os.makedirs('output/temperature', exist_ok=True)
    os.makedirs('output/levelset', exist_ok=True)

    file_list = [f for f in os.listdir(folder)]
    file_list.sort() #Sort case numbers in ascending order

    for file_name in file_list:
        case_number = file_name.split('_')[1].split('.')[0]
        explore = np.genfromtxt('output_dat/'+file_name, skip_header=1)

        #Store individual variables
        x = explore[:, 0]
        y = explore[:, 1]
        u = explore[:, 2]
        v = explore[:, 3]
        p = explore[:, 4]
        T = explore[:, 5]
        ls = explore[:, 6]

        #Replace rows to None so as to mask plot background
        indices = np.where(ls > 0)
        u[indices] = None
        v[indices] = None
        p[indices] = None
        T[indices] = None


        #Find shape of grid (Should be 514, 258)
        num_x_points = len(np.unique(x))
        num_y_points = len(np.unique(y))

        u_grid = u.reshape((num_y_points, num_x_points))
        v_grid = v.reshape((num_y_points, num_x_points))
        p_grid = p.reshape((num_y_points, num_x_points))
        T_grid = T.reshape((num_y_points, num_x_points))
        ls_grid = ls.reshape((num_y_points, num_x_points))

        grids = [u_grid, v_grid, p_grid, T_grid, ls_grid]
        labels = ['u_velocity', 'v_velocity', 'pressure', 'temperature', 'levelset']

        for grid, label in zip(grids, labels):
            plt.figure()
            plt.imshow(grid, cmap='RdYlBu_r', vmin=np.nanmin(grid), vmax=np.nanmax(grid), extent=[0, 2, 0, 1])
            plt.colorbar(label=label)
            plt.xlabel('X')
            plt.ylabel('Y')
            filename = 'output/' + label + '/' + label + ' _' + str(case_number).zfill(5) + ".png"
            plt.savefig(filename)
            plt.close()


# Helper function to check for errors with divergence in numerical simulation - more useful as diagnostic in early stage
def filter_out_divergence(folder):
    data = []
    file_list = [f for f in os.listdir(folder)]
    file_list.sort() #Sort case numbers in ascending order

    for file_name in file_list:
        case_number = file_name.split('_')[1].split('.')[0]
        with open('mon/'+file_name, 'r') as file:
            lines = file.readlines()

            if 'The ERROR L2NORM of Ux+Vy :' in lines[-1]: #To check if converged
                line1, line2, line3, line4  = lines[-4:]
                time = line1.split()[0]
                t = line1.split()[1]
                time_step = line2.split()[-3]
                cpu_time = line3.split()[-1]
                error = line4.split()[-1]
            else:
                time, t, time_step, cpu_time, error = None, None, None, None, None


        data.append((case_number, time, t, time_step, cpu_time, error))

    with open('filter.csv', 'w', newline='') as file:
            file.write('case_number, time, t, time_step, cpu_time, error\n')
            for row in data:
                file.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n')


#Create csv.csv file given pipe folder name
create_csv("pipes")

#Plot u_velocity, v_velocity, pressure, temperature and level set plots from output.dat files in folder
plot_output("output")

#To extract TIME, T, TIME STEP, CPU_TIME, ERROR and tabulate in filter.csv
filter_out_divergence('mon/')