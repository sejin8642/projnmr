#!/usr/bin/env python3

# python  script to generate random data set using ftnmr module with baseline artifact
from pathlib import Path
import sys
sys.path.insert(1, str(Path.home()/'gd'/'ftnmr'/'scripts'))
sys.path.insert(1, str(Path.home()/'gd'/'projnmr'/'scripts'))

import h5py
import numpy as np
from ftnmr import spectrometer
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
        data_name='filename',
        dir_name='data',
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
    data_name: str
        Saved data file name without hdf5 extension (default filename)
    dir_name: str
        directory name in which to save data
    data_size: int
        Mininum file size for each data and target in megabytes (default 256 mb)
    dtype: str
        Data type for the data (default float32)
    """

    # file name configurations
    file_number = str(n).zfill(len(str(N)))
    log_date = dt.datetime.today() + dt.timedelta(hours=hours) 
    log_name = data_name + file_number + log_date.strftime('.%Y-%m-%d.log')
    file_name = filename + file_number + '.hdf5'

    # spectrometer object instantiation with total number of measurements
    spec = spectrometer()
    number_of_measurements = int(data_size*pow(2, 19)/spec.shift.nbytes)
    NofM_str = str(number_of_measurements)
    digits = len(NofM_str)

    # logging configuration with input log name to append info to 
    logging.basicConfig(
            filename = Path.cwd() / dir_name / log_name,
            level=logging.DEBUG)

    # dynamically simulate and write data to hdf5 file
    with h5py.File(Path.cwd() / data_dir_name / file_name, 'w') as f:
        f.create_dataset('data', (number_of_measurements, spec.nf), dtype=np.float32)
        f.create_dataset('target', (number_of_measurements, spec.nf), dtype=np.float32)

        # random generation and measurements of metabolites
        for m in range(number_of_measurements):
            moles = {al[25+k]:(mg(), uniform(0, 50)) for k in range(1, randint(1, 15))} 
            spec.artifact(baseline=True)
            spec.measure(moles=moles)
            f['data'][m, :], f['target'][m, :] = spec()

            # log info to append
            message = str(m).zfill(digits) + '/' + NofM_str + " measurements are done "
            timestamp = dt.datetime.now() + dt.timedelta(hours=hours)
            logging.info(message + timestamp.strftime('(%H:%M:%S).'))

def main():
    # exits script if data directory already exists
    data_dir_name = 'data'
    if Path(data_dir_name).exists():
        sys.exit(1)

    # chemical shift range for the data
    spec = spectrometer()
    with h5py.File(Path.cwd() / data_dir_name / 'chemical_shift.hdf5', 'w') as f:
        f.create_dataset('shift', data=spec.shift, dtype=np.float32)

    N = 16 # number of hdf5 data 
    hours = 14 # hours to delay timestamp for log info

    # get that data!!!
    with futures.ProcessPoolExecutor() as executor:
        for n in range(N):
            executor.submit(generateData, n, N, hours, 'baseline', dir_name, 256)

if __name__ == '__main__': 
    main()

