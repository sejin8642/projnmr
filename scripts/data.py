# python  script to generate random data set using ftnmr module
from ftnmr import *
import h5py
import concurrent.futures as cf

def generate(n, m):
    with h5py.File(f"{n}", 'w') as f:
        pass

def main():
    with cf.ProcessPoolExecutor() as executor:
        pass

if __name__ == '__main__': main()
