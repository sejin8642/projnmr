#!/usr/bin/env python3

#SBATCH --job-name=diff_noise

#SBATCH --partition=shared
#SBATCH --time=1-12:00:00 ## time format is DD-HH:MM:SS

## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G ## max amount of memory per node you require

#SBATCH --output=diff_noise.%A.out

# add directories in which necessary modules are located
import platform
from pathlib import Path
import sys

os_name = platform.system()
if os_name == 'Darwin':
	gd_path = str(Path.home()/'Library/Mobile Documents/com~apple~CloudDocs/gd')
elif os_name == 'Linux':
	gd_path = str(Path.home()/'gd')
else:
	raise ValueError('You are not on either Darwin or Linux OS')
project_path = gd_path + '/projects'
sys.path.insert(0, project_path + '/ftnmr/scripts')
sys.path.insert(0, project_path + '/projnmr/scripts')

# for error "not creating xla devices tf_xla_enable_xla_devices not set"
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# for error "Successfully opened dynamic library libcudart.so.10.1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# script modules
from concurrent import futures
import ftnmr
import projnmr
from projnmr import metaboliteGenerator as mg
from string import ascii_letters as al

import numpy as np
import h5py

# spectrometer and repeat, noise number parameters
spec = ftnmr.spectrometer(shift_maximum=128.0,  shift_minimum=7)
std_increase = 2e-05
repeat = 100 
num_noise = 45
num_chunk = 16
seg_size = int(spec.shift.shape[0]/num_chunk)

# slice object to select appropriate chemical shift
start = 2
end = 14
diff = end - start
new_range = seg_size*diff
SR = slice(seg_size*start, seg_size*end)

# initialization of spectra and target
spectra_arrays = np.zeros((num_noise, repeat, new_range))
target_arrays = np.zeros((num_noise, repeat, new_range))

def spectra_target_diff_noise(n):
    # create spectrometer object and choose appropriate noise level
    spec = ftnmr.spectrometer(shift_maximum=128.0,  shift_minimum=7)
    spec.std = 1e-04 + n*std_increase

    # n is a noise level
    for r in range(repeat):
        # random molecule solution generation
        ru = np.random.uniform
        rr = np.random.randint
        moles = {al[26+k]:(mg(), ru(0, 50)) for k in range(0, rr(1, 15))}

        # different molecules and different artifacts with the same noise level 
        spec.artifact(baseline=True, phase_shift=True)
        spec.measure(moles=moles)
        spectra, target = spec()
        spectra_arrays[n][r], target_arrays[n][r] = spectra[SR], target[SR]

# process the fn
with futures.ThreadPoolExecutor(max_workers=23) as executor:
    futures_list = []
    for n in range(num_noise):
        future = executor.submit(
                spectra_target_diff_noise, 
                n) 
                
        futures_list.append(future)

    for future in futures.as_completed(futures_list):
        result = future.result()

# once parallel computation is done, save the data as npy files
np.save('spectra_different_noises.npy', spectra_arrays)
np.save('target_different_noises.npy', target_arrays)

