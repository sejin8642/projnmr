#!/usr/bin/env python3

#SBATCH --job-name=peaks_SNR

#SBATCH --partition=shared
#SBATCH --time=1-12:00:00 ## time format is DD-HH:MM:SS

## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=4G ## max amount of memory per node you require

#SBATCH --output=SNR.%A.out

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

# modules to import
from concurrent import futures
import tensorflow as tf
from tensorflow import keras

import ftnmr
import NN_models
import nmrglue as ng
import numpy as np

def noise_gen(mean = 0, std_dev = 0.02):
    return np.random.normal(mean, std_dev, size=4096)

def loc_identify(peak, true_peak, diff = 100):
    """
    returns a index number for peak table that is closest to the true peak location
    """
    amp = 0
    loc_true = next(iter(true_peak))[0]
    num_model = None
    for index, table in enumerate(peak):
        diff_temp = abs(table[0] - loc_true)
        if diff_temp < diff:
            diff = diff_temp
            amp = table[3]
            num_model = index
        elif diff_temp == diff:
            if amp < table[3]:
                diff = diff_temp
                amp = table[3]
                num_model = index

    return peak[num_model]

def loc_identify2(peak):
    """
    returns a table for peak table that has the highest peak amplitude
    """
    table_temp = None
    amp = 0
    for index, table in enumerate(peak):
        if amp < table[3]:
            amp = table[3]
            table_temp = table
            
    return table_temp

# parameters and arras to store 
base_height = 4
num_heights = 10
SNR = np.zeros((num_heights, ))
num_samples = 200
tables = np.zeros((num_heights, 5, num_samples, 4), dtype='float')

# load the model
model_path = Path.home() / Path('data/may_12_SNR/model_1519027.hdf5')
loaded_model = keras.models.load_model(model_path, compile=False)
print("model loaded")

def parallel_fn(m):
    """
    simple parellelization of double for loops
    """
    hydrogens = {'a':(m, 0.5, 100)}
    couplings = []
    mole = ftnmr.molecule(hydrogens=hydrogens, couplings=couplings)
    moles = {'A': (mole, 1)}

    # measure the sample
    spec = ftnmr.spectrometer(shift_maximum=128.0, shift_minimum=0.6, std=0.0)
    spec.measure(moles=moles)

    # peak picking parameters
    m = m-base_height
    peak_height = np.max(spec.spectra)
    pthres = 0.8*peak_height
    SNR[m] = peak_height/0.02

    # true peak without processing and noise
    true_peak = ng.analysis.peakpick.pick(spec.spectra, pthres=pthres)

    # iterate to obtain peak picking results
    for n in range(num_samples):
        # different noise for different iterations
        noise = noise_gen()

        # noisy peak picking
        spectrum_real = spec.spectra+noise
        peaks_noisy = ng.analysis.peakpick.pick(spectrum_real, pthres=pthres)
        noisy_temp = loc_identify2(peaks_noisy)
        tables[m][0][n][:] = tuple(noisy_temp)

        # process the noisy and clean signal (model and ng)
        model_output_noisy = loaded_model((spectrum_real)[np.newaxis, :])
        model_numpy_noisy = model_output_noisy.numpy()[0]
        model_numpy_clean = model_numpy_noisy - noise
        ng_output_noisy = ng.proc_bl.baseline_corrector(spectrum_real)
        ng_output_clean = ng_output_noisy - noise

        # peak picking (model + ng)
        peaks_model_noisy = ng.analysis.peakpick.pick(model_numpy_noisy, pthres=pthres)
        peaks_model_clean = ng.analysis.peakpick.pick(model_numpy_clean, pthres=pthres)
        peaks_ng_noisy = ng.analysis.peakpick.pick(ng_output_noisy, pthres=pthres)
        peaks_ng_clean = ng.analysis.peakpick.pick(ng_output_clean, pthres=pthres)

        # store peak picking results
        model_temp_noisy = loc_identify2(peaks_model_noisy)
        tables[m][1][n][:] = tuple(model_temp_noisy)
        model_temp_clean = loc_identify(peaks_model_clean, true_peak)
        tables[m][2][n][:] = tuple(model_temp_clean)

        ng_temp_noisy = loc_identify2(peaks_ng_noisy)
        tables[m][3][n][:] = tuple(ng_temp_noisy)
        ng_temp_clean = loc_identify(peaks_ng_clean, true_peak)
        tables[m][4][n][:] = tuple(ng_temp_clean)

# process the fn
with futures.ThreadPoolExecutor(max_workers=11) as executor:
    futures_list = []
    print("futures list created before parallel looping")
    for m in range(base_height, base_height+num_heights):
        future = executor.submit(
                parallel_fn, 
                m) 
                
        futures_list.append(future)

    for future in futures.as_completed(futures_list):
        result = future.result()

# once parallel computation is done, save the data as npy files
np.save('tables.npy', tables)
np.save('SNR.npy', SNR)

print("The program outputs two numpy arrays: tables and SNR. tables contain peak information of 200 samples over 10 different SNRs (peak location as an index, identifier, linewidth, peak amplitude). The SNR array contains SNR of the corresponding peaks")
