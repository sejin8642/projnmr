# python  script to generate random data set using ftnmr module with baseline artifact
import h5py
from ftnmr import *
from concurrent import futures

def generate(n, m, b, name='data'):
    """
    generate function simulates ftnmr.measure output and saves it as hdf5 files.
    
    Parameters
    ----------
    n: int
        Number of processed sample signal in each data block
    m: int
        m-th block of dataset
    b: int
        Number of data blocks
    name: str
        Saved data file name (default data)
    """
    spec = spectrometer(t_cut=1500, std=0.00001)
    data = np.zeros((n, spec.nf))
    for i in range(n):
        molecules = _
        spec.artifact(baseline=True)
        spec.measure(moles=molecules)
        data[i] = spec.spectra

    with h5py.File(name + str(m).zfill(len(str(b))), 'w') as f:
        pass

def main():
    with futures.ProcessPoolExecutor() as executor:
        pass

if __name__ == '__main__': main()
