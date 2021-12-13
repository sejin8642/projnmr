# A list of functions for ftnmr.spectrometer
import sys
sys.path.insert(1, '/home/sejin8642/gd/ftnmr/scripts')

import numpy as np
from ftnmr import spectrometer

def moleculesGenerator(N = 9):
    """
    Generates random molecules 

    Things to consider before modification
    hydrogens = {str:(total number, chemical shift, T2)}
    couplings = (str, str, J-coupling) 

    Parameters
    ----------
    N: int
        Minimum number of hydrogen groups in a molecule

    return
    ------
    molecules: dict[str]:(molecule, float)
        Dictionary of molecules with their relative abundances
    """
    # hydrogen clusters (first and later) for groups list and their size
    FG = [[3], [3, 3], [3, 2, 3], [3, 2, 2, 3], [3, 3, 1, 3]]
    MG = [[2], [2, 3], [2, 2, 3], [2, 2, 2, 3], [2, 3, 1, 3]]
    size = [1, 2, 3, 4, 4] 
   
    # index for the first group
    ind = np.random.randint(0, 5)
    groups = FG[ind] # hydrogen groups with their hydrogen count
    length = size[ind] # length of groups 
    cut = [length] # indices for cluster separation

    while (length < N):
        # reduce one hydrogen from one group if attached to molecular backbone group
        ind = np.random.randint(0, length)
        if groups[ind] != 1: groups[ind] -= 1 

        # index for later groups
        ind = np.random.randint(0, 5)
        groups.extend(MG[ind])
        length += size[ind]
        cut.append(length)

    return groups, cut

