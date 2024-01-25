## Script to create a set of shell scripts for numerical simulation from runs in their aggregated folders

[create_batchrun.py](https://github.com/ooichinchun/Microchannel-Designs/blob/main/SimulateDesigns/create_batchrun.py) creates a set of shell scripts for the numerical simulation

Key steps are:
1) Collect designs (input_XXXXX.dat) and run information (run_XXXXX.inp) in a folder
2) Run script to generate the batchrun.sh file
3) Remove limit on memory with the following command: ulimit -s unlimited
4) Ensure batchrun.sh is executable by running: chmod +x batchrun.sh
5) Run from linux command line with ./batchrun.sh

