## Script to create a set of microchannel designs with various processed outputs for numerical simulation


### Step 1: Adjust parameterizations in Design Generator Script
[generator_v2.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/CreateDesigns/generator_v2.py) creates a set of microchannel designs with specific parameterizations.

Key parameterizations are:
1) Number of pipes and thickness (in <u> generate_pipe_network</u>
2) Domain dimension = (256, 512) for a 2 x 1 rectangular domain
3) Bezier point parameterization including I) <u>num_of_bezier_points</u> and II) control points definition in <u>generate_random_bezier_path</u>
4) Key simulation settings in <u>create_run</u>

### Step 2: Call the functions with the following 3 commands

Sequence: 
1) Call <u>pipe_generator(i)</u> in a loop to generate the corresponding numbered designs
2) Call <u>create_batchrun_folder('run')</u> to create the batchrun.sh file to batch the numerical simulator in Linux
3) Call <u>create_csv('pipes')</u> to create a csv with a count of the number of pipes for later use

This will generate the following outputs:
1) 'vof' and 'ls' are matplotlib contour plots of the designs
2) 'pipes' store the pipe object for future referencing
3) 'pts_ls' contains input_XXXXX.dat which has the starting Level-Set array for the numerical solver to initialize and solve
4) 'run' contains run_XXXXX.inp which contains the numerical solver parameters (from create_run)

Files in '/pts_ls' and '/run' need to be put in the same folder with the batchrun.sh (shell script) and a.out (numerical solver) to batch run

### Step 3: Additional diagnostic helper functions are available to visualize and check simulation outputs

[pipe_diagnostics.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/CreateDesigns/pipe_diagnostics.py) contains helper functions to check the output from the numerical simulation.

With the generated output from the numerical solver, we can also batch the extraction of contour plots from the output, and the extraction of numerical convergence information from the monitor files. 

The python script does the following:
1) Call <u>plot_output('output')</u> to plot contour plots of all the physical fields (u, v, p, T, ls)
2) Call <u>filter_out_divergence('mon')</u> to extract the time taken and convergence metrics from the monitor files generated during simulation