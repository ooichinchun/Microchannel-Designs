# Script written by Sheares Toh and Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import os

# Create single batch run file to run from command line for set number of cases
def create_batchrun(start,end):
  lines = ""

  for case_number in range(start, end+1):
      lines = lines + "./a.out < run_" + str(case_number).zfill(5) + ".inp > mon_" + str(case_number).zfill(5) + ".inp \n"

  with open('batchrun.sh', 'w') as file:
      file.writelines(lines)

def create_batchrun_folder(folder):
  data = []
  file_list = [f for f in os.listdir(folder)]
  file_list.sort() #Sort case numbers in ascending order

  for file_name in file_list:
    if 'run_' in file_name:
      case_number = file_name.split('_')[1].split('.')[0]
      data.append(case_number)

  lines = ""

  for case_number in data:
    lines = lines + "./a.out < run_" + str(case_number).zfill(5) + ".inp > mon_" + str(case_number).zfill(5) + ".inp \n"

  file_path = os.path.join(folder,'batchrun.sh')

  with open(file_path, 'w') as file:
    file.writelines(lines)


#Create batchrun.sh file given run folder containing all run files to be simulated
for i in range(1,11):
  folder_name = 'test-run-' + str(i).zfill(2)
  create_batchrun_folder(folder_name)

#create_batchrun_folder("test-run-01")
#create_batchrun_folder("test-run-02")
#create_batchrun_folder("test-run-03")
#create_batchrun_folder("test-run-04")
#create_batchrun_folder("test-run-05")
#create_batchrun_folder("test-run-06")
#create_batchrun_folder("test-run-07")
#create_batchrun_folder("test-run-08")
#create_batchrun_folder("test-run-09")
#create_batchrun_folder("test-run-10")

#Create batchrun.sh file given a range of case numbers
#create_batchrun(0,9)