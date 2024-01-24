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

# Bezier curve control points are generated for x between 0-2 and y between 0-1

#Given entry and exit point coordinates, output a set of coordinates of a bezier curve

# 1st Variant forces pipes to merge through the center of the domain
def generate_random_bezier_path_prev(start, end):
    control_points = np.array([[start[0] + np.random.uniform(0.1, 0.2), start[1] + np.random.uniform(-0.02, 0.02)],
                               [start[0] + np.random.uniform(0.2, 0.3), start[1] + np.random.uniform(-0.05, 0.05)],
                               [start[0] + np.random.uniform(0.3, 0.4), start[1] + np.random.uniform(-0.1, 0.1)],
                               [np.random.uniform(0.4, 0.6), np.random.uniform(0.35, 0.65)],
                               [np.random.uniform(0.6, 1.4), 0.5],
                               [np.random.uniform(1.4, 1.6), np.random.uniform(0.3, 0.65)],
                               [end[0] - np.random.uniform(0.3, 0.4), end[1] + np.random.uniform(-0.1, 0.1)],
                               [end[0] - np.random.uniform(0.2, 0.3), end[1] + np.random.uniform(-0.05, 0.05)],
                               [end[0] - np.random.uniform(0.1, 0.2), end[1] + np.random.uniform(-0.02, 0.02)]])

    t = np.linspace(0, 1, num_of_bezier_points)
    path_x = (1 - t)**8 * start[0] + 8 * (1 - t)**7 * t * control_points[0, 0] + 28 * (1 - t)**6 * t**2 * control_points[1, 0] + 56 * (1 - t)**5 * t**3 * control_points[2, 0] + 70 * (1 - t)**4 * t**4 * control_points[3, 0] + 56 * (1 - t)**3 * t**5 * control_points[4, 0] + 28 * (1 - t)**2 * t**6 * control_points[5, 0] + 8 * (1 - t) * t**7 * control_points[6, 0] + t**8 * end[0]
    path_y = (1 - t)**8 * start[1] + 8 * (1 - t)**7 * t * control_points[0, 1] + 28 * (1 - t)**6 * t**2 * control_points[1, 1] + 56 * (1 - t)**5 * t**3 * control_points[2, 1] + 70 * (1 - t)**4 * t**4 * control_points[3, 1] + 56 * (1 - t)**3 * t**5 * control_points[4, 1] + 28 * (1 - t)**2 * t**6 * control_points[5, 1] + 8 * (1 - t) * t**7 * control_points[6, 1] + t**8 * end[1]

    return np.column_stack((path_x, path_y))

# 2nd Variant forces pipes to merge through the center of the domain with some deviation

def generate_random_bezier_path_prev_w_dev(start, end, dev):
    control_points = np.array([[start[0] + np.random.uniform(0.1, 0.2), start[1] + np.random.uniform(-0.02, 0.02)],
                               [start[0] + np.random.uniform(0.2, 0.3), start[1] + np.random.uniform(-0.05, 0.05)],
                               [start[0] + np.random.uniform(0.3, 0.4), start[1] + np.random.uniform(-0.1, 0.1)+0.2*dev],
                               [np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6)+0.5*dev],
                               [np.random.uniform(0.6, 1.4), np.random.uniform(0.45, 0.55)+dev],
                               [np.random.uniform(1.4, 1.6), np.random.uniform(0.4, 0.6)+0.5*dev],
                               [end[0] - np.random.uniform(0.3, 0.4), end[1] + np.random.uniform(-0.1, 0.1)+0.2*dev],
                               [end[0] - np.random.uniform(0.2, 0.3), end[1] + np.random.uniform(-0.05, 0.05)],
                               [end[0] - np.random.uniform(0.1, 0.2), end[1] + np.random.uniform(-0.02, 0.02)]])

    t = np.linspace(0, 1, num_of_bezier_points)
    path_x = (1 - t)**8 * start[0] + 8 * (1 - t)**7 * t * control_points[0, 0] + 28 * (1 - t)**6 * t**2 * control_points[1, 0] + 56 * (1 - t)**5 * t**3 * control_points[2, 0] + 70 * (1 - t)**4 * t**4 * control_points[3, 0] + 56 * (1 - t)**3 * t**5 * control_points[4, 0] + 28 * (1 - t)**2 * t**6 * control_points[5, 0] + 8 * (1 - t) * t**7 * control_points[6, 0] + t**8 * end[0]
    path_y = (1 - t)**8 * start[1] + 8 * (1 - t)**7 * t * control_points[0, 1] + 28 * (1 - t)**6 * t**2 * control_points[1, 1] + 56 * (1 - t)**5 * t**3 * control_points[2, 1] + 70 * (1 - t)**4 * t**4 * control_points[3, 1] + 56 * (1 - t)**3 * t**5 * control_points[4, 1] + 28 * (1 - t)**2 * t**6 * control_points[5, 1] + 8 * (1 - t) * t**7 * control_points[6, 1] + t**8 * end[1]

    return np.column_stack((path_x, path_y))


# 3rd Variant - pipes are un-constrained, except to avoid the top and bottom surfaces
# Currently used version
def generate_random_bezier_path(start, end):
    control_points = np.array([[start[0] + np.random.uniform(0.05, 0.1), start[1] + np.random.uniform(-0.02, 0.02)],
                               [start[0] + np.random.uniform(0.1, 0.2), start[1] + np.random.uniform(-0.05, 0.05)],
                               [start[0] + np.random.uniform(0.2, 0.3), start[1] + np.random.uniform(-0.1, 0.1)],
                               [np.random.uniform(0.3, 0.6), np.random.uniform(0.2, 0.8)],
                               [np.random.uniform(0.6, 1.4), np.random.uniform(0.2, 0.8)],
                               [np.random.uniform(1.4, 1.7), np.random.uniform(0.2, 0.8)],
                               [end[0] - np.random.uniform(0.2, 0.3), end[1] + np.random.uniform(-0.1, 0.1)],
                               [end[0] - np.random.uniform(0.1, 0.2), end[1] + np.random.uniform(-0.05, 0.05)],
                               [end[0] - np.random.uniform(0.05, 0.1), end[1] + np.random.uniform(-0.02, 0.02)]])

    t = np.linspace(0, 1, num_of_bezier_points)
    path_x = (1 - t)**8 * start[0] + 8 * (1 - t)**7 * t * control_points[0, 0] + 28 * (1 - t)**6 * t**2 * control_points[1, 0] + 56 * (1 - t)**5 * t**3 * control_points[2, 0] + 70 * (1 - t)**4 * t**4 * control_points[3, 0] + 56 * (1 - t)**3 * t**5 * control_points[4, 0] + 28 * (1 - t)**2 * t**6 * control_points[5, 0] + 8 * (1 - t) * t**7 * control_points[6, 0] + t**8 * end[0]
    path_y = (1 - t)**8 * start[1] + 8 * (1 - t)**7 * t * control_points[0, 1] + 28 * (1 - t)**6 * t**2 * control_points[1, 1] + 56 * (1 - t)**5 * t**3 * control_points[2, 1] + 70 * (1 - t)**4 * t**4 * control_points[3, 1] + 56 * (1 - t)**3 * t**5 * control_points[4, 1] + 28 * (1 - t)**2 * t**6 * control_points[5, 1] + 8 * (1 - t) * t**7 * control_points[6, 1] + t**8 * end[1]

    return np.column_stack((path_x, path_y)), control_points



#Given number of pipes required, create corresponding number of pipe objects and return a list of objects

def generate_pipe_network(num_pipes, rect_width = 2.0, rect_height = 1.0):
    #rect_width = 2.0
    #rect_height = 1.0
    pipes = []

    for _ in range(num_pipes):
        entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
        exit_point = (rect_width, np.random.uniform(0.1, 0.9) * rect_height)
        thickness = np.random.uniform(0.08, 0.2) * rect_height
        path, control_points = generate_random_bezier_path(entry_point, exit_point)
        pipe = Pipe(entry_point, exit_point, path, thickness, control_points)
        pipes.append(pipe)

    return pipes


#To flip sdf values based on whether it is within or outside pipe region
def sign_flipper_v2(sdf, grid_resolution, to_multiply, upper_path, lower_path):

    horizontal_grid = np.linspace(0 + 2/(grid_resolution[0]*2), 2 - 2/(grid_resolution[0]*2), grid_resolution[0])
    vertical_grid = np.linspace(0 + 1/(grid_resolution[1]*2), 1 - 1/(grid_resolution[1]*2), grid_resolution[1])

    for j in range(grid_resolution[0]):
        horizontal_grid_val = horizontal_grid[j]
        vert_upper_y = upper_path[(np.argmin(np.abs(upper_path[:,0] - horizontal_grid_val))),1]
        vert_lower_y = lower_path[(np.argmin(np.abs(lower_path[:,0] - horizontal_grid_val))),1]

        for i in range(grid_resolution[1]):
            if (vertical_grid[i] < vert_upper_y) and (vertical_grid[i] > vert_lower_y):
                sdf[i,j] *= -1
                to_multiply[i,j] = -1

    return sdf, to_multiply


#This function receives a set of pipe networks, returns the sdf for individual centerlines and also a matrix containing the signs

def process_pipes(design, grid_resolution, grid_points):

    sdf_list = []
    sdf_vof = []

    # Create a grid
    #x = np.linspace(0 + 2/(grid_resolution[0]*2), 2 - 2/(grid_resolution[0]*2), grid_resolution[0])
    #y = np.linspace(0 + 1/(grid_resolution[1]*2), 1 - 1/(grid_resolution[1]*2), grid_resolution[1])
    #X, Y = np.meshgrid(x, y)
    #grid_points = np.column_stack((X.ravel(), Y.ravel()))

    for pipe in design:
        # Extract path and thickness
        path = pipe.path
        thickness = pipe.thickness

        #Find upper and lower path curves
        upper_path = path + np.array([0, thickness/2])
        lower_path = path - np.array([0, thickness/2])

        # Calculate distances to upper and lower paths
        upper_distances = np.min(cdist(grid_points, upper_path), axis = 1)
        lower_distances = np.min(cdist(grid_points, lower_path), axis = 1)

        sdf = np.minimum(upper_distances, lower_distances)
        sdf = sdf.reshape(dimension)

        #Initialise a numpy array for multiplication for each pipe
        to_multiply = np.ones(dimension)

        # Apply the sign flip logic
        sdf_pipe, to_multiply_pipe = sign_flipper_v2(sdf, grid_resolution, to_multiply, upper_path, lower_path)

        sdf_list.append(sdf_pipe)
        sdf_vof.append(to_multiply_pipe)

    return sdf_list, sdf_vof

# Pipes are merged pair-wise with this function
def combine_vof(vof1,vof2):

  combine_vof_12 = 0.25*(vof1 + 1)*(vof2 + 1) - 0.25*(vof1 + 1)*(vof2 - 1) - 0.25*(vof1 - 1)*(vof2 + 1) - 0.25*(vof1 - 1)*(vof2 - 1)

  return combine_vof_12



#Variables to adjust

dimension = (256, 512) #Resolution of design space
num_of_bezier_points = 1500 #Number of discrete points for bezier curve

rect_width = 2.0
rect_height = 1.0

# Create a grid
x = np.linspace(0 + rect_width/(dimension[1]*2), rect_width - rect_width/(dimension[1]*2), dimension[1])
y = np.linspace(0 + rect_height/(dimension[0]*2), rect_height - rect_height/(dimension[0]*2), dimension[0])
X, Y = np.meshgrid(x, y)
grid_ls = np.column_stack((X.ravel(), Y.ravel()))

# VOF to Level-Set Factor - May need to be tuned 
eps = 1/dimension[0]

# Function to create the run_XXXXX.inp file for the numerical simulation

def create_run(case_number):
    lines = '''6         !number of threads for OpenMP
0         !CASE NUMBER
500       !Re
4.3389    !Pr
512 256   !INPUT TOTAL Volumes for X and Y
1         !CHOOSE STEADY-STATE OR REAL TRANSIENT(1 for steady, 2 for transient)
2         !CHOOSE TRANSIENT SCHEME:(1 for 1st-order, 2 for 2nd order)
0         !INPUT STARTING IMPLCIIT EULER STEP
1e-6      !INPUT Tol
4.0       !INPUT END TIME
1e-3      !INPUT Dt
15.0      !INPUT pseudo time for T
1e-3      !INPUT Dt for T
4         !INPUT RK_stage
3.0       !INPUT CFL_DIV, CFL > CFL_DIV THEN STOP
1000      !INPUT HOW MANY STEPS PER PLOT
0         !PLOT PLOT(0: dont plot, 1: plot)
1         !INPUT DIS_IB
1         !CHOOSE CONVECTION SOLVER(1 for DRP, 2 for DRP+limiter)
4         !CHOOSE P SOLVER (4 for MG)
1         !CHOOSE MASS-CORRECTED METHOD (1 for VELOCITY MODIFICATION)
1         !CHOOSE EPS_SMOOTH METHOD (1 for NON-SMOOTH)'''.split('\n')

    lines = [line + '\n' for line in lines]
    lines[1] = f"{case_number}         !CASE NUMBER\n"

    with open('run/run_' + str(case_number).zfill(5) + '.inp', 'w') as file:
        file.writelines(lines)


# Function to create the pipe designs and save the corresponding files for the numerical simulation
# /vof and /ls are the illustrations of the designs ; /pipes store the pipe object for future referencing
# /pts_ls contains input_XXXXX.dat which has the starting Level-Set array for the numerical solver to initialize and solve
# /run contains run_XXXXX.inp which contains the numerical solver parameters (from create_run)
# files in /pts_ls and /run need to be put in the same folder with the batchrun.py script and a.out (numerical solver) to run
def pipe_generator(case_number):

    os.makedirs('pipes', exist_ok=True)
    os.makedirs('vof', exist_ok=True)
    os.makedirs('ls', exist_ok=True)
    os.makedirs('pts_ls', exist_ok=True)
    os.makedirs('run', exist_ok=True)

    #Saving files
    filename_pipe = 'pipes/pipes_' + str(case_number).zfill(5)  + '.p' #Pickle file for pipe network
    filename_vof = 'vof/vof_' + str(case_number).zfill(5)  + '.png' #Standard plot for pipe network
    filename_ls = 'ls/ls_' + str(case_number).zfill(5)  + '.png' #SDF plot for pipe network
    filename_pts = 'pts_ls/input_' + str(case_number).zfill(5)  + '.dat' #Level set value for each coordinate

    #Generate pipe network
    num_pipes = np.random.randint(3, 5) #Randomly select number of pipes for each design
    pipes = generate_pipe_network(num_pipes,rect_width, rect_height)
    pickle.dump(pipes, open(filename_pipe, 'wb'))

    #Obtain centerlines
    sdf_list_individual_centerline, sign_matrix_list = process_pipes(pipes, (dimension[1],dimension[0]), grid_ls)

    #Merge pipes and save standard plot of putative VOF
    #cur_vof = np.tanh( -sdf_list_individual_centerline[0] / eps)
    #for i in range(len(sdf_list_individual_centerline)-1):
    #    old_vof = np.copy(cur_vof)
    #    new_vof = np.tanh( -sdf_list_individual_centerline[i+1] / eps)
    #    cur_vof = combine_vof(old_vof, new_vof)
    #plt.figure()
    #plt.contourf(X,Y,cur_vof, levels=[0,1])
    #plt.savefig(filename_vof)
    #plt.close()

    #Save level set csv for initialization -> Soln is not accurate yet
    #combine_ls = -eps * np.arctanh(cur_vof*(1-1e-10))
    #pts_ls = np.column_stack((X.ravel(), Y.ravel(), combine_ls.ravel()))
    #np.savetxt(filename_pts, pts_ls, delimiter=',')

    #Merge pipes and save standard plot of putative VOF
    cur_vof = np.tanh( -sdf_list_individual_centerline[0] / eps)
    for i in range(len(sdf_list_individual_centerline)-1):
        old_vof = np.copy(cur_vof)
        new_vof = np.tanh( -sdf_list_individual_centerline[i+1] / eps)
        cur_vof = combine_vof(old_vof, new_vof)
    plt.figure()
    plt.contourf(X,Y,cur_vof, levels=[0,1])
    plt.savefig(filename_vof)
    plt.close()

    #Save level set points for initialization -> Soln is not accurate yet ; will be refined by numerical solver
    combine_ls = -eps * np.arctanh(cur_vof*(1-1e-10))
    pts_ls = np.column_stack((X.ravel(), Y.ravel(), combine_ls.ravel()))
    header = "x y ls"
    np.savetxt(filename_pts, pts_ls, delimiter=',', header=header, comments='')

    #Save level set plot
    sdf_grid = combine_ls.reshape(dimension)
    sdf_grid = np.flipud(sdf_grid)
    plt.figure()
    plt.imshow(sdf_grid, cmap='RdYlBu_r', vmin=-0.0463261680228711, vmax=0.0463261680228711, extent=[0, 2, 0, 1])
    plt.colorbar(label='Level Set')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename_ls)
    plt.close()

    #Create run file
    create_run(case_number)

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


# Create single batch run file to run from command line for set number of cases

def create_batchrun(start,end):
  lines = ""

  for case_number in range(start, end+1):
      lines = lines + f"./a.out < run_" + str(case_number).zfill(5) + ".inp > mon_" + str(case_number).zfill(5) + ".inp \n"

  with open('batchrun.sh', 'w') as file:
      file.writelines(lines)

def create_batchrun_folder(folder):
    data = []
    file_list = [f for f in os.listdir(folder)]
    file_list.sort() #Sort case numbers in ascending order

    for file_name in file_list:
        case_number = file_name.split('_')[1].split('.')[0]
        data.append(case_number)

    lines = ""

    for case_number in data:
      lines = lines + f"./a.out < run_" + str(case_number).zfill(5) + ".inp > mon_" + str(case_number).zfill(5) + ".inp \n"

    with open('batchrun.sh', 'w') as file:
      file.writelines(lines)


##Main

#Select number of cases
#num_of_cases = 50

#for i in range(num_of_cases):
#  pipe_generator(i) #Generates n number of pipe networks and saves as pipes.p, vof.png, ls.dat and ls.png

for i in range(20000,30000):
  pipe_generator(i) #Generates n number of pipe networks and saves as pipes.p, vof.png, ls.dat and ls.png
  print(i)

#Create batchrun.sh file given run folder
create_batchrun_folder("run")

#Create csv.csv file given pipe folder name
create_csv("pipes")



#Plot u_velocity, v_velocity, pressure, temperature and level set plots from output.dat files in folder
#plot_output("output_dat")

#To extract TIME, T, TIME STEP, CPU_TIME, ERROR and tabulate in filter.csv
#filter_out_divergence('mon/')