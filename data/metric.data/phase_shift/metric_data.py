#!/usr/bin/env python3

#SBATCH --job-name=baseline

#SBATCH --partition=shared
#SBATCH --time=1-12:00:00 ## time format is DD-HH:MM:SS

## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=4G ## max amount of memory per node you require

#SBATCH --output=baseline.%A.out

# python  script to generate random data set using ftnmr module with baseline artifact
from pathlib import Path
import sys
import os
sys.path.insert(1, str(Path.home()/'gd'/'projects'/'ftnmr'/'scripts'))
sys.path.insert(1, str(Path.home()/'gd'/'projects'/'projnmr'/'scripts'))

import h5py
import numpy as np
from ftnmr import spectrometer
from projnmr import metaboliteGenerator as mg
from numpy.random import uniform, randint
from concurrent import futures
from string import ascii_letters as al
import logging
import secrets

def generateData(
        n,
        N,
        dir_path=Path.cwd(),
        file_name='filename',
        data_dir='data',
        log_dir='log',
        data_size_power=7,
        data_length=2**16,
        dtype='float32'):
    """
    generateData simulates ftnmr.spectrometer.measure and saves its output as hdf5 files
    
    Parameters
    ----------
    n: int
        n-th data block file index used for hdf5 file naming
    N: int
        Total number of data blocks
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
        (default 7, which is pow(2, 7) or 128 megabytes). 
    data_length: int
        Length of the data. It cannot be greater than 2^16, and it should be a some power of 2.
        For example, 2**10=1024 is an acceptable data length size (default 2**16)
    dtype: str
        Data type for the data (default float32)
    """
    ######## change your artifact types here before running the script #######
    baseline = False
    phase_shift = True
    smoothness = False
    ##########################################################################

    # append index number to file name
    file_name = file_name + str(n).zfill(len(str(N)))

    # spectrometer object instantiation with single data instance size
    spec = spectrometer()
    instance_size_in_bytes = np.dtype(dtype).itemsize*data_length 

    # make sure rescale ratio is greater than or equal to 1 to reduce the output data size
    assert data_length <= spec.nf 
    rescale_ratio = int(spec.nf/data_length)

    # processing the final spectrometer output if data_length is smaller then spec.nf
    # it basically shrinnks the size of the spectrometer output using np.max function
    if data_length == spec.nf:
        def dataProcess():
            return spec()
    else:
        def dataProcess():
            target = np.reshape(spec.target, (data_length, rescale_ratio))
            target2 = np.reshape(spec.target2, (data_length, rescale_ratio))
            spectra = np.reshape(spec.spectra, (data_length, rescale_ratio))
            return np.max(spectra, axis=1), np.max(target, axis=1), np.max(target2, axis=1)
   
    # total number of measurements based on bytes (19 = 10 + 10 - 1) and log step size
    number_of_measurements = int(2**(data_size_power+19)/instance_size_in_bytes)
    log_step_size = int(pow(2, np.log2(number_of_measurements) - 4))

    # strings for logging
    NofM_str = str(number_of_measurements)
    digits = len(NofM_str)

    # logging configuration with input log name to append info to 
    logging.basicConfig(
            filename = dir_path/log_dir/(file_name + '.log'),
            level=logging.DEBUG)

    # dynamically simulate and write data to hdf5 file
    logging.info(f"Total number of measurements are " + NofM_str)
    with h5py.File(dir_path/data_dir/(file_name + '.hdf5'), 'w') as f:
        f.create_dataset('data', (number_of_measurements, data_length), dtype=dtype)
        f.create_dataset('target', (number_of_measurements, data_length), dtype=dtype)
        f.create_dataset('target2', (number_of_measurements, data_length), dtype=dtype)

        # random generation and measurements of metabolites
        for p in range(0, number_of_measurements, log_step_size):
            # simulate and save data as hdf5
            for m in range(p, p + log_step_size):
                moles = {al[25+k]:(mg(), uniform(0, 50)) for k in range(1, randint(1, 15))} 
                spec.artifact(
                        baseline=baseline, 
                        phase_shift=phase_shift, 
                        smoothness=smoothness)
                spec.measure(moles=moles, extra_target=True)
                f['data'][m, :], f['target'][m, :], f['target2'][m, :]= dataProcess()

            # log info to append
            message = str(p+log_step_size).zfill(digits) + '/' + NofM_str + " measurements done"
            logging.info(message)

def main():
    N = 20 # number of hdf5 data 
    dir_path = Path.cwd() # current working directory

    # Retrieve the Slurm job allocation number from the environment variable
    job_id = os.environ.get('SLURM_JOBID')

    # create directories in which to save data, logs, and chemical shift range hdf5
    data_dir = f"data.{job_id}"
    log_dir = f"log.{job_id}"
    shift_range = f"chemical_shift.{job_id}.hdf5"
    Path(data_dir).mkdir()
    Path(log_dir).mkdir()

    # chemical shift range for the data
    data_length = 2**10
    spec = spectrometer()
    rescale_ratio = int(spec.nf/data_length)
    rescaled_shift = spec.shift[::rescale_ratio]
    with h5py.File(dir_path / shift_range, 'w') as f:
        f.create_dataset('shift', data=rescaled_shift, dtype=np.float32)


    # get that data!!!
    with futures.ProcessPoolExecutor() as executor:
        futures_list = []
        for n in range(N):
            future = executor.submit(
                    generateData, 
                    n, 
                    N,
                    dir_path=dir_path, 
                    file_name='baseline', 
                    data_dir=data_dir, 
                    log_dir=log_dir,
                    data_size_power=1,
                    data_length=data_length,
                    dtype='float32')
            futures_list.append(future)

        for future in futures.as_completed(futures_list):
            result = future.result()

if __name__ == '__main__': 
    main()
