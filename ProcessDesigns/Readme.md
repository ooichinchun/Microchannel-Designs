## Script to process the completed microchannel design simulations from the output files


### Step 1: Run scripts to parse numerical simulation output from .dat files into numpy arrays
[combine-output.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/ProcessDesigns/combine-output.py) combines the numerical simulation output from a sub-set of individual runs into a larger array.

[calc_performance_metrics.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/ProcessDesigns/calc_performance_metrics.py) calculates the key performance metrics of {Pressure Drop, Heat Transfer Rate and Mass Imbalances} from the collated sub-set of arrays from *combine-output.py*

Notes:
1) The arrays require a lot of memory and are first processed in sets of about 5k designs
2) Simulation outputs are first parsed by *combine-output.py*
3) Arrays are further loaded and processed for the 3 performance metrics in *calc_performance_metrics.py*

### Step 2: Run collect_all.py
[collect_all.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/ProcessDesigns/collect_all.py) concatenates all the previously collected arrays of sub-sets of microchannel design simulation results.

Sub-components: 
1) Load and concatenate {u, v, P, T, ls} arrays from the earlier sub-sets and save the full array
2) Load and concatenate {Mass imbalance, Pressure Drop, Heat Transfer Rate} metrics from the earlier sub-sets and save the full array
3) Compare "Mass Imbalances" to threshold of 1e-6 and identify cases with potential convergence issues from full folder of output files (file_list)

This will generate the following outputs:
1) combined_{u, v, P, T, ls}_all.npy
2) {mass_bal, p_diff, Qdot}_all.npy and {mass_bal, p_diff, Qdot}_all.csv
3) Cases with higher mass imbalances will be printed out in console for recording


### Step 3: Resize {u,v,P,T} arrays and generate VOF images for Visualization and as png files for models

[resize_arrays.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/ProcessDesigns/resize_arrays.py) contains helper functions to downsize the outputs from the numerical simulation and generate VOF .png images.

The python script has the following 2 functions:
1) **downsize_array(input_array)** downsizes the physical fields {u, v, p, T} from 512 x 256 to 128 x 128
2) **downsize_ls(input_ls_array, output_folder)** saves the VOF as a 512 x 256 image. Downsizing can be adjusted within the function


### Step 4: Resize generated images from ML model back to VOF images with correct size for Visualization and as input_XXXXX.dat files for numerical simulation

[resize_generated_image_to_vof.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/ProcessDesigns/resize_generated_image_to_vof.py) has been created to use PIL libraries to do image downsizing, saving of VOF .png images and creation of input_XXXXX.dat files for numerical simulation. 

Current script has not been tested.

### Step 5: Resize generated images from ML model back to VOF images with correct size for Visualization and as input_XXXXX.dat files for numerical simulation

[n_pipes_counter.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/ProcessDesigns/n_pipes_counter.py) has been created to use PIL libraries to do image re-sizing and creation of matching arrays for input into conditional diffusion model. 

Input and output folders need to be defined.

Current script has not been tested.
