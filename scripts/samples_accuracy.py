#!/usr/bin/env python3

#SBATCH --job-name=FID_samples

#SBATCH --partition=shared
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS

## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=2G ## max amount of memory per node you require

#SBATCH --output=fid_samples.%A.out

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

import numpy as np
from scipy.optimize import fsolve
import pickle

# functions to use
# return spectrum based on abundance, T2, phase shift angle
def spectrum_gen(abundance=50, T2=100, angle=0.0, cs=0.5, shift_minimum=0.6, std=0.0):
    spec = ftnmr.spectrometer(shift_maximum=128.0, shift_minimum=shift_minimum, std=std)
    couplings = []

    # prepare sample and measure the spectrum
    hydrogens = {'a':(abundance, cs, T2)}
    mole = ftnmr.molecule(hydrogens=hydrogens, couplings=couplings)
    moles = {'A': (mole, 1)}
    spec.ps[1] = float(2*np.pi*angle/180.0)
    spec.measure(moles=moles)

    return spec.spectra

# return width, amplitude, and symmetricity based on abundance and T2
# the argument is a list of abundance, T2, and phase shift angle (degree)
def width_amp_sym(args, pthres=0.2):
    spec = ftnmr.spectrometer(shift_maximum=128.0, shift_minimum=0.6, std=0.0)
    couplings = []

    # prepare sample and measure the spectrum
    hydrogens = {'a':(args[0], 0.5, args[1])}
    mole = ftnmr.molecule(hydrogens=hydrogens, couplings=couplings)
    moles = {'A': (mole, 1)}
    spec.ps[1] = float(2*np.pi*args[2]/180.0)
    spec.measure(moles=moles)
    
    # get peak
    h = 0.5*np.max(spec.spectra)
    if h < pthres:
        pthres = h
    loc, _, w, A = ng.analysis.peakpick.pick(spec.spectra, pthres=pthres)[0]

    # get peak area
    scale = 2
    peak_area = spec.spectra[int(loc-scale*w):int(loc+scale*w+1)]
    area_len = len(peak_area)
    
    # get difference sum divided by amplitude around the peak
    difference = spec.spectra[:(area_len//2)+1] - spec.spectra[::-1][:area_len//2+1]
    sym = np.sum(difference)/A
    
    return [w, A, sym]

# fn to estimate abundance, T2, angle from measured width, amp, sym
def estimate_ATA(WAS, database, max_angle=10, angle_step=0.1):
    width, amplitude, sym = WAS
    # Calculate the distance from (a, b) for each element in the last dimension
    weight = float(amplitude/width)
    distances = np.sqrt(( weight*(database[:, :, 0] - width) )**2 + (database[:, :, 1] - amplitude)**2)

    # Find the indices of the minimum distance for a_base and T2_base
    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    a_base, T2_base = float(min_index[0] + 25), float(min_index[1] + 25)

    # parameters for peak analysis to get angle
    pthres = 0.2
    scale = 2
    sym_target = np.abs(sym)
    sign = sym/sym_target
    angle_base = -sign*max_angle

    # loop through the angle
    for angle in np.arange(0, max_angle, angle_step):
        # get w and A for angle extimation
        data = spectrum_gen(a_base, T2_base, angle, 0.5)
        peak = ng.analysis.peakpick.pick(data, pthres=pthres)
        loc, _, w, A = peak[0]

        # get temporary symmetricity to compare sym
        peak_area = data[int(loc-scale*w):int(loc+scale*w+1)]
        area_len = len(peak_area)
        difference = data[:(area_len//2)+1] - data[::-1][:area_len//2+1]
        sym_temp = np.sum(difference)/A

        # get the base angle right after sym threshold
        if sym_target < -sym_temp:
            angle_base = -sign*angle
            break

    return a_base, T2_base, angle_base

def inverse_function(y, initial_guess, weights=[1, 1, 1], xtol=1.49012e-08, maxfev=400):
    def equation(x):
        w, A, s = width_amp_sym(x)
        return [
            weights[0]*(w - y[0]),
            weights[1]*(A - y[1]),
            weights[2]*(s - y[2])]

    return fsolve(equation, initial_guess, xtol=xtol, maxfev=maxfev)

# Get the main sample directory and a list of all subdirectories
dir_path = Path.home() / Path('BMSE')
subdirectories = [subdir.name for subdir in dir_path.iterdir() if subdir.is_dir()]
subdirectories.sort()
num_samples = len(subdirectories)

# load the model
model_path = Path.home() / Path('BMSE/model_may_22.hdf5')
loaded_model = keras.models.load_model(model_path, compile=False)

samples = [0]*num_samples

def noise_gen(mean = 0, std_dev = 0.02, size=4096):
    return np.random.normal(mean, std_dev, size=size)
    
def sample_accuracy(ind):
    # read in the bruker formatted data
    fid_path = dir_path / Path(subdirectories[ind])
    dic, data_original = ng.bruker.read(str(fid_path))

    # remove the digital filter
    data = ng.bruker.remove_digital_filter(dic, data_original)

    # process the spectrum
    data = ng.proc_base.zf_size(data, 32768)    # zero fill to 32768 points
    data = ng.proc_base.fft(data)               # Fourier transform
    #data = ng.proc_base.ps(data, p0=-50.0)      # phase correction
    #data = ng.proc_base.di(data)                # discard the imaginaries
    #data = ng.proc_base.rev(data)               # reverse the data

    # initial correction to get a sense of its size
    ng_output = ng.proc_autophase.autops(data, fn='acme', p0=0.1, p1=0.1)
    ng_output = ng.proc_bl.baseline_corrector(ng_output)

    # rescale the data
    max_height = 6.0
    scale_data = max_height/np.max(ng_output.real)
    data_rescaled = scale_data*data

    # get noise
    NR = noise_gen(mean = 0, std_dev = 0.02, size=data_rescaled.shape[0])
    NI = noise_gen(mean = 0, std_dev = 0.02, size=data_rescaled.shape[0])
    noise = NR + NI*1j

    # select the data for model processing
    MAX = np.max(data_rescaled)
    MIN = np.min(data_rescaled)
    MAXr = np.abs(MAX.real)
    MAXi = np.abs(MAX.imag)
    MINr = np.abs(MIN.real)
    MINi = np.abs(MIN.imag)
    MAXES = np.array([MAXr, MAXi, MINr, MINi])
    max_index = np.argmax(MAXES)

    if max_index == 0:
        data_model = data_rescaled.real
    if max_index == 1:
        data_model = data_rescaled.imag
    if max_index == 2:
        data_model = -data_rescaled.real
    if max_index == 3:
        data_model = -data_rescaled.imag

    # obtain model and ng output
    model_input = np.array([data_model + noise.real])
    model_output = loaded_model(model_input)
    model_numpy = model_output.numpy()[0] - noise.real

    ng_output = ng.proc_autophase.autops(data_rescaled + noise, fn='acme', p0=0.1, p1=0.1)
    ng_output = ng.proc_bl.baseline_corrector(ng_output) - noise

    # peak, width, amplitude, location threshold for ng_output and model output 
    pthres = 0.2
    wthres = 15.0
    Athres = 25.0
    loc_thres = 5

    # pick peaks 
    peak_model = ng.analysis.peakpick.pick(model_numpy, pthres=pthres)
    peak_ng = ng.analysis.peakpick.pick(ng_output.real, pthres=pthres)

    # Find matching elements within the threshold
    peaks_ng = []
    peaks_model = []
    for i in range(len(peak_ng)):
        for j in range(len(peak_model)):
            if abs(peak_ng[i][0] - peak_model[j][0]) <= loc_thres:
                if Athres < peak_ng[i][3] and wthres < peak_ng[i][2]:
                    peaks_ng.append(peak_ng[i])
                    peaks_model.append(peak_model[j])

    # get width, amplitude, and symmetricity values for each peak
    peak_num = len(peaks_model)
    WAS_values = np.zeros((2, peak_num, 3))
    for n in range(peak_num):
        # get loc, w, A of peaks (model and ng)
        loc1, _, w1, A1 = peaks_model[n]
        loc2, _, w2, A2 = peaks_ng[n]
        scale = 2

        # get symmetricity for model
        model_area = model_numpy[int(loc1-scale*w1):int(loc1+scale*w1+1)]
        model_len = len(model_area)
        difference_model = model_numpy[:(model_len//2)+1] - model_numpy[::-1][:model_len//2+1]
        sym1 = np.sum(difference_model)/A1
        WAS_values[0][n] = w1, A1, sym1

        # get symmetricity for ng
        ng_area = ng_output.real[int(loc2-scale*w2):int(loc2+scale*w2+1)]
        ng_len = len(ng_area)
        difference_ng = ng_output.real[:(ng_len//2)+1] - ng_output.real[::-1][:ng_len//2+1]
        sym2 = np.sum(difference_ng)/A2
        WAS_values[1][n] = w2, A2, sym2

    # guess abundance, T2, and angle values for fsolve
    ATA_guesses = np.zeros((2, peak_num, 3))
    weights = np.zeros((2, peak_num, 3))

    for n in range(2):
        for m in range(peak_num):
            w1, w2, w3 =  WAS_values[n][m]
            ATA_guesses[n][m][:] = estimate_ATA(WAS_values[n][m], database_all, max_angle=45)
            weights[n][m][:] = 1/(w1), 1/(w2), 1/(np.abs(w3))

    # get abundance, T2, and angle values of the peaks
    ATA_values = np.zeros((2, peak_num, 3))
    xtol=1e-16
    maxfev=2000

    for n in range(2):
        for m in range(peak_num):
            try:
                output_temp = inverse_function(
                	WAS_values[n][m], 
                	ATA_guesses[n][m], 
                	weights=weights[n][m], 
                	xtol=xtol, 
                	maxfev=maxfev)
            except IndexError:
                ATA_guesses[n][m][1] += 1.0
                output_temp = inverse_function(
                	WAS_values[n][m], 
                	ATA_guesses[n][m], 
                	weights=weights[n][m], 
                	xtol=xtol, 
                	maxfev=maxfev)

            ATA_values[n][m][:] = output_temp

    # get accuracies
    accuracies = np.zeros((2, peak_num, 2))
    for n in range(2):
        for m in range(peak_num):
            a, T2, angle = ATA_values[n][m]

            # get the base spectrum and peak
            data = spectrum_gen(a, T2, 0.0)

            # get the peaks
            _,_,w,A = ng.analysis.peakpick.pick(data, pthres=0.2)[0]

            # get accuracies of width and amplitude
            accuracies[n][m][0] = 100*np.abs(w-WAS_values[n][m][0])/w
            accuracies[n][m][1] = 100*np.abs(A-WAS_values[n][m][1])/A
            
    WAS_ATA_accu = [WAS_values, ATA_values, accuracies]
    samples[ind] = WAS_ATA_accu

# process the fn
with futures.ThreadPoolExecutor(max_workers=24) as executor:
    futures_list = []
    for n in range(num_noise):
        future = executor.submit(
                sample_accuracy, 
                n) 
                
        futures_list.append(future)

    for future in futures.as_completed(futures_list):
        result = future.result()

# once parallel computation is done, save the data by serializing with pickle
with open("samples.accuracies.pickle", "wb") as f:
    pickle.dump(samples, f)

