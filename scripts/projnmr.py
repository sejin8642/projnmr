# A list of functions for ftnmr.spectrometer
import sys
sys.path.insert(1, '/home/sejin8642/gd/ftnmr/scripts')

import numpy as np
from ftnmr import spectrometer

def molecules():
    """
    Generates random molecules 

    Things to consider before modification
    hydrogens = {str:(total number, chemical shift)}
    couplings = (str, str, J-coupling) 
    return
    ------
    molecules: dict[str]:(molecule, float)
        Dictionary of molecules with their relative abundances
    """
    pass

