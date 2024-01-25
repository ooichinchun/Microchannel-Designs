# Script written by Ooi Chin Chun
# Institute of High Performance Computing, Singapore
# Copyright (c) 2023. 

import numpy as np
import os

##########
# combine the processed 258 x 514 original arrays

#array_name = 'combined_u_'
#array_name = 'combined_v_'
#array_name = 'combined_p_'
#array_name = 'combined_T_'
array_name = 'combined_ls_'

fn = array_name + '5k.npy'
data_all = np.load(fn)

# Loop needs to be adjusted - Previous cases were extracted in groups of 5k for memory restrictions
for i in range(10, 35, 5):

  fn = array_name + str(i) + 'k.npy'
  data_all = np.concatenate((data_all, np.load(fn)))
  
print(data_all.shape)

output_fn = array_name + 'all'
np.save(output_fn,data_all)

##########
# Combine the processed metrics
# Mass Imbalance / Delta Pressure / Rate of Transfer (Qdot)

#calc_name = 'mass_bal_'
#calc_name = 'p_diff_'
#calc_name = 'Qdot_'

#data_all = np.array([])
#for i in range(5, 35, 5):

#  fn = calc_name + str(i) + 'k.npy'
#  data_all = np.hstack((data_all, np.load(fn)))
  
#print(data_all.shape)

#output_fn = calc_name + 'all'
#np.save(output_fn,data_all)
#np.savetxt(output_fn+'.csv',data_all, delimiter=',')


##########
# Check mass imbalance and check the corresponding case number

#if calc_name == 'mass_bal_':

#  file_list = os.listdir('/home/ooicc/microfluidic-channels-ph/output/')
#  file_list.sort() #Sort case numbers in ascending order

#  idx_exclude = []

#  for i in range(len(data_all)):

#    if data_all[i] > 1e-6:
#      print(i)
#      print(file_list[i])
#      idx_exclude.append(i)
    
#  idx_exclude = np.array(idx_exclude)

#  np.save('exclude_idx.npy',idx_exclude)