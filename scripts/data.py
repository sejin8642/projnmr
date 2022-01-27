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

def generateData(n, N, filename='filename', data_size=50, dtype='float32'):
    """
    generateData simulates ftnmr.spectrometer.measure and saves its output as hdf5 files
    
    Parameters
    ----------
    n: int
        n-th data block file index used for hdf5 file naming
    m: int
        Total number of data blocks
    filename: str
        Saved data file name without hdf5 extension (default filename)
    data_size: int
        Mininum file size for each data and target in megabytes (default 256 mb)
    dtype: str
        Data type for the data (default float32)
    """
    spec = spectrometer()
    number_of_measurements = int(data_size*pow(2, 20)/spec.shift.nbytes)

    # preallocation of data
    data = np.zeros((number_of_measurements, spec.nf)).astype(dtype)
    targets = np.zeros((number_of_measurements, spec.nf)).astype(dtype)

    for m in range(number_of_measurements):
        moles = {al[25+k]:(mg(), uniform(0, 50)) for k in range(1, randint(1, 15))} 
        spec.artifact(baseline=True)
        spec.measure(moles=moles)
        data[m], targets[m] = spec()

    with h5py.File(filename + '_data' + str(n).zfill(len(str(N))) + '.hdf5', 'w') as f:
        f.create_dataset('data', data=data, dtype=np.float32)

    with h5py.File(filename + '_targ' + str(n).zfill(len(str(N))) + '.hdf5', 'w') as f:
        f.create_dataset('target', data=targets, dtype=np.float32)

def main():
    spec = spectrometer()
    with h5py.File('chemical_shift.hdf5', 'w') as f:
        f.create_dataset('shift', data=spec.shift, dtype=np.float32)

    N = 7 # number of data blocks for each data and targets
    with futures.ProcessPoolExecutor() as executor:
        for n in range(N):
            executor.submit(generateData, n, N, 'baseline', 128)

if __name__ == '__main__': 
    main()
