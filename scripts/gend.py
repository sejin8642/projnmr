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
        hours=0,
        dir_path=Path.cwd(),
        file_name='filename',
        timestamp='2022',
        data_dir='data',
        log_dir='log',
        data_size_power=7,
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
    dir_path: PosixPath (or None)
        Posix directory path to put all the output (default cwd). If no path was provided,
        all data will be saved in the current directory of the scrypt file
    file_name: str
        Saved data and log file name without file extension (default filename)
    data_dir: str
        directory name in which to save data
    log_dir: str
        directory name in which to save logs
    data_size_power: int
        Exponent of two that yields mininum hdf5 file size in megabytes 
        (default 7, which is pow(2, 7) or 128 megabytes)
    dtype: str
        Data type for the data (default float32)
    """
    # append index number to file name
    file_name = file_name + str(n).zfill(len(str(N)))

    # spectrometer object instantiation with total number of measurements
    spec = spectrometer()
    number_of_measurements = int(pow(2, data_size_power+19)/spec.shift.nbytes)
    log_step_size = int(pow(2, np.log2(number_of_measurements) - 4))
    NofM_str = str(number_of_measurements)
    digits = len(NofM_str)

    # logging configuration with input log name to append info to 
    logging.basicConfig(
            filename = dir_path/log_dir/(file_name + '.log'),
            level=logging.DEBUG)

    # dynamically simulate and write data to hdf5 file
    with h5py.File(dir_path/data_dir/(file_name + '.hdf5'), 'w') as f:
        f.create_dataset('data', (number_of_measurements, spec.nf), dtype=np.float32)
        f.create_dataset('target', (number_of_measurements, spec.nf), dtype=np.float32)

        # random generation and measurements of metabolites
        for p in range(0, number_of_measurements, log_step_size):
            # simulate and save data as hdf5
            for m in range(p, p + log_step_size):
                moles = {al[25+k]:(mg(), uniform(0, 50)) for k in range(1, randint(1, 15))} 
                spec.artifact(baseline=True)
                spec.measure(moles=moles)
                f['data'][m, :], f['target'][m, :] = spec()

            # log info to append
            message = str(p+log_step_size).zfill(digits) + '/' + NofM_str + " measurements done"
            logging.info(message)

def main():
    N = 4 # number of hdf5 data 
    hours = 14 # hours to delay timestamp for log info (remote system might have different time)
    dir_path = Path.cwd() # current working directory

    # create directories in which to save data and logs
    timestamp = (dt.datetime.now() + dt.timedelta(hours=hours)).strftime('.%Y-%m-%d~%I:%M%p')
    data_dir = 'data' + timestamp
    log_dir = 'log' + timestamp
    Path(data_dir).mkdir()
    Path(log_dir).mkdir()

    # chemical shift range for the data
    spec = spectrometer()
    with h5py.File(dir_path / 'chemical_shift.hdf5', 'w') as f:
        f.create_dataset('shift', data=spec.shift, dtype=np.float32)

    # get that data!!!
    with futures.ProcessPoolExecutor() as executor:
        for n in range(N):
            executor.submit(generateData, n, N,
                    hours=hours, 
                    dir_path=dir_path, 
                    file_name='baseline', 
                    data_dir=data_dir, 
                    log_dir=log_dir,
                    data_size_power=6,
                    dtype='float32')

if __name__ == '__main__': 
    main()
