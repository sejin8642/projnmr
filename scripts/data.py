#!/usr/bin/env python3

# python  script to generate random data set using ftnmr module with baseline artifact
import sys
sys.path.insert(1, '/home/sejin8642/gd/ftnmr/scripts')

import h5py
import numpy as np
from ftnmr import *
from projnmr import metaboliteGenerator as mg
from numpy.random import uniform, randint
from concurrent import futures
from string import ascii_letters as al
import logging
import datetime as dt

def generateData(
        n,
        N,
        hours,
        filename='filename',
        data_size=50,
        dtype='float32'):
    """
    generateData simulates ftnmr.spectrometer.measure and saves its output as hdf5 files
    
    Parameters
    ----------
    n: int
        n-th data block file index used for hdf5 file naming
    N: int
        Total number of data blocks
    hours: int
        hours to delay timestamp for log info (used on remote clusters that have different time)
    filename: str
        Saved data file name without hdf5 extension (default filename)
    data_size: int
        Mininum file size for each data and target in megabytes (default 256 mb)
    dtype: str
        Data type for the data (default float32)
    """

    # file name configurations
    file_number = str(n).zfill(len(str(N))
    logname = filename + file_number + dt.today().strftime(".%Y-%m-%d.log")

    # spectrometer object instantiation with total number of measurements
    spec = spectrometer()
    number_of_measurements = int(data_size*pow(2, 20)/spec.shift.nbytes)
    NofM_str = str(number_of_measurements)

    # logging configuration with input log name to append info to 
    logging.basicConfig(filename = logname, level=logging.DEBUG)

    # preallocation of data
    data = np.zeros((number_of_measurements, spec.nf)).astype(dtype)
    targets = np.zeros((number_of_measurements, spec.nf)).astype(dtype)

    # random generation and measurements of metabolites
    for m in range(number_of_measurements):
        moles = {al[25+k]:(mg(), uniform(0, 50)) for k in range(1, randint(1, 15))} 
        spec.artifact(baseline=True)
        spec.measure(moles=moles)
        data[m], targets[m] = spec()
        message = str(m).zfill(len(NofM_str)) + '/' + NofM_str + " measurements are done "
        timestamp = dt.datetime.now() + dt.timedelta(hours=hours)

        # log info to append
        logging.info(message + timestamp.strftime("(%H:%M:%S)."))

    # save simulated data and targets
    with h5py.File(filename + '_data' + file_number + '.hdf5', 'w') as f:
        f.create_dataset('data', data=data, dtype=np.float32)

    with h5py.File(filename + '_targ' + file_number + '.hdf5', 'w') as f:
        f.create_dataset('target', data=targets, dtype=np.float32)

def main():
    # chemical shift range for the data
    spec = spectrometer()
    with h5py.File('chemical_shift.hdf5', 'w') as f:
        f.create_dataset('shift', data=spec.shift, dtype=np.float32)

    N = 20 # number of data blocks for each data and targets
    hours = 14 # hours to delay timestamp for log info

    # get that data!!!
    with futures.ProcessPoolExecutor() as executor:
        for n in range(N):
            executor.submit(generateData, n, N, hours, 'baseline', 128)

if __name__ == '__main__': 
    mai()
