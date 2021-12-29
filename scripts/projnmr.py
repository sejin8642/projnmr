# A list of functions for ftnmr.spectrometer
import sys
sys.path.insert(1, '/home/sejin8642/gd/ftnmr/scripts')

import ftnmr
import numpy as np
from numpy.random import uniform
from string import ascii_letters as al

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
    # hydrogen clusters (first and later) for groups list and their sizes
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
    sizes = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4] 
   
    # backbone attachment for the first cluster
    ind = np.random.randint(0, 11) # index for the first group
    indices = [ind] # indices of backbones in order
    length = sizes[ind] # length of groups 
    groups = FG[ind] # hydrogen groups with their hydrogen count
    cut = [length] # indices for cluster separation

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

        # same code block from above
        ind = np.random.randint(0, 11)
        if max_N < length + sizes[ind]:
            break
        indices.append(ind)
        length += sizes[ind]
        groups.extend(MG[ind])
        cut.append(length)

    # get rid of duplicate methyl groups for isobutane backbone
    for n, x in enumerate(indices):
        if x == 10:
            if (groups[cut[n]-1] == 3) and (groups[cut[n]-3] == 3):
                groups[cut[n]-1] = np.random.randint(1, 3) 

    # reduce one hydrogen from one random group multiple random times 
    while (bool(np.random.randint(0, 2))):
        ind = np.random.randint(0, length)
        if groups[ind] != 1: groups[ind] -= 1 

    # create equivalent hydrogen groups randomly and combine their total hydrogen count
    def equiv(n, size):
        if np.random.randint(0, 2):
            if groups[cut[n]-1] == 3 and groups[cut[n]-2] == 2:
                groups[cut[n]-1] = np.random.choice([4, 6]) 
                groups[cut[n]-2] = 1

        if np.random.randint(0, 2):
            if groups[cut[n]-size] == 3 and groups[cut[n]-size+1] == 2:
                groups[cut[n]-size] = np.random.choice([4, 6]) 
                groups[cut[n]-size+1] = 1

    equi = [1, 3, 6, 7, 9]
    for n, ind in enumerate(indices):
        if ind in equi:
            equiv(n, sizes[ind])

    # random shift to incorporate aldehyde shift ~ 9.7
    def rand(x):
        if x != 1:
            return uniform(0.5, 6.0)
        else:
            if np.random.randint(0, 10):
                return uniform(0.5, 6.0)
            else:
                return uniform(9.0, 10.0)

    # hydrogen groups dictionary (100 < T2 < 250) and couplings list (2 < J <20)
    hydrogens = {al[n]:(x, rand(x), uniform(100.0, 250.0)) for n, x in enumerate(groups)}
    couplings = [(al[n-1], al[n], uniform(2.0, 20.0)) for n in range(1, length) if n not in cut]

    return ftnmr.molecule(hydrogens=hydrogens, couplings=couplings)

