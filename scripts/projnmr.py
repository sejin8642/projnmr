# A list of functions for ftnmr.spectrometer
import sys
sys.path.insert(1, '/home/sejin8642/gd/ftnmr/scripts')

import numpy as np
from ftnmr import spectrometer

def moleculesGenerator(min_N = 4, max_N = 10):
    """
    Generates random molecules 

    Things to consider before modification
    hydrogens = {str:(total number, chemical shift, T2)}
    couplings = (str, str, J-coupling) 

    Parameters
    ----------
    min_N: int
        Minimum number of hydrogen groups in a molecule
    max_N: int
        Maximum number of hydrogen groups in a molecule

    return
    ------
    molecules: dict[str]:(molecule, float)
        Dictionary of molecules with their relative abundances
    """
    # hydrogen clusters (first and later) for groups list and their size
    backbones = [
            'methane',
            'ethane', 
            'ethylene', 
            'propane', 
            'propene1',
            'propene2',
            'butane',
            'butene1',
            'butene2',
            'butene3',
            'isobutane']
    FG = [
            [3],
            [3, 3], 
            [2, 2],
            [3, 2, 3],
            [2, 1, 3],
            [3, 1, 2],
            [3, 2, 2, 3], 
            [2, 1, 2, 3],
            [3, 1, 1, 3],
            [3, 2, 1, 2],
            [3, 3, 1, 3]]
    MG = [
            [2],
            [2, 3], 
            [1, 2],
            [2, 2, 3],
            [1, 2, 3],
            [2, 2, 2],
            [2, 2, 2, 3], 
            [1, 1, 2, 3],
            [2, 1, 1, 3],
            [2, 2, 1, 2],
            [2, 3, 1, 3]]
    size = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4] 
   
    # index for the first group
    ind = np.random.randint(0, 11)
    indices = [ind] # indices of backbones in order
    length = size[ind] # length of groups 
    groups = FG[ind] # hydrogen groups with their hydrogen count
    cut = [length] # indices for cluster separation
    isobutane = [ind//10] # to indicate which cluster is isobutane backbone

    # condition for while loop for more clusters
    def cond():
        if length < min_N:
            return True
        else:
            return bool(np.random.randint(0, 2))

    # backbone attachment for multiple clusters
    while (cond()):
        # reduce one hydrogen from one group if attached to molecular backbone group
        ind = np.random.randint(0, length)
        if groups[ind] != 1: groups[ind] -= 1 

        # index for later groups
        ind = np.random.randint(0, 11)
        if max_N < length + size[ind]:
            break
        indices.append(ind)
        length += size[ind]
        groups.extend(MG[ind])
        cut.append(length)
        isobutane.append(ind//10)

    # get rid of duplicate methyl groups for isobutane backbone
    for n, x in enumerate(isobutane):
        if x == 1:
            if (groups[cut[n]-1] == 3) and (groups[cut[n]-3] == 3):
                groups[cut[n]-1] = np.random.randint(1, 3) 

    # reduce one hydrogen from one random group multiple random times 
    while (bool(np.random.randint(0, 2))):
        ind = np.random.randint(0, length)
        if groups[ind] != 1: groups[ind] -= 1 

    # create equivalent hydrogen groups randomly and combine their total hydrogen count
    FG = [
            [3, 3], 
            [3, 2, 3],
            [3, 2, 2, 3], 
            [2, 1, 2, 3],
            [3, 2, 1, 2]]
    equi = [1, 3, 6, 7, 9]
    for n, ind in enumerate(indices):
        if ind in equi:
            if groups(cut[n]

    return groups, cut, isobutane, indices
