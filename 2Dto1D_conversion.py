# Before running this .py code in terminal, set the path as
# $ export PATH=~/miniforge3/envs/capillarywave/bin:$PATH

import numpy as np
import h5py
import os
import sys

# Get filenames from command line arguments
read_file_name = sys.argv[1]   # e.g., 11252025_b_0.3vpp_data_roi-none_cal-true.hdf5
save_file_name = sys.argv[2]   # e.g., Q_1D_0p30vpp_b.h5

read_file_path = os.path.join('/disk/hyk049/DHM_new_experiment/0p30', read_file_name)
save_file_path = os.path.join('/disk/hyk049/DHM_new_1Dcenter', save_file_name)

data_dict = {}


# Read file & save it in a new dictionary
print(f"Reading the file {read_file_name}...")
with h5py.File(read_file_path, 'r') as f:
    for key in f['main'].keys():
        data_dict[key] = f['main'][key][:]
        
with h5py.File(read_file_path, 'r') as f:
    t = f['meta']['t'][:]
    x = f['meta']['x'][:]
    y = f['meta']['y'][:]
    
time_steps = len(data_dict)
rows, cols = 200, 200

Q = np.zeros((rows * cols, time_steps))

print("Converting data to 2D matrix...")
for i, key in enumerate(sorted(data_dict.keys(), key=int)):
    Q[:, i] = data_dict[key].reshape(-1)
    
del data_dict
Q_subspace = Q.reshape(200, 200, Q.shape[1])            

Q_1D = Q_subspace[:,100,:]  # center point

print(f"Saving 1D data to {save_file_name}...")
with h5py.File(save_file_path, "w") as f:
    f.create_dataset("Q_1D", data=Q_1D)
    f.create_dataset("t", data=t.astype(np.float64))
    f.create_dataset("x", data=x.astype(np.float64))
    
print("Data conversion complete.")