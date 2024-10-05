#!/usr/bin/env python3

#SBATCH --job-name=ATA_WAS
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4G ## max amount of memory per node you require
#SBATCH --output=ATA_WAS.%A.out

print()
print("python script file name: ", __file__)
print()
'''
this script generates database numpy array from ATA (abundance, T2, angle) to WAS 
(width, amplitude, symmetricity). This numpy array is used to rough estimate the 
ATA values based on WAS values. Make sure the indices and ATA values do not match
exactly. For default, abundance = i+base, T2 = j+base, angle = 0.1*k where base
is 25 and i,j,k are indices of ATA database array (base can be specified).
'''

from pathlib import Path
import sys
import os
sys.path.insert(1, str(Path.home()/'gd'/'projects'/'ftnmr'/'scripts'))
sys.path.insert(1, str(Path.home()/'gd'/'projects'/'projnmr'/'scripts'))

import logging
import numpy as np
import time
import ftnmr
from concurrent import futures

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('ATA_WAS.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# params for ATA
base = 25.0
abundance_num = 400
T2_num = 300
angle_num = 200
start_time = time.time()

# preallocate output numpy array
ATA_to_WAS = np.zeros((abundance_num, T2_num, angle_num, 3), dtype='float32')

def A_to_WAS(i):
    # from index to abundance number with logging     abundance = base+i
    abundance = base+i
    logger.info(f"abundance {abundance} processing.....")

    # preallocate WAS output for the given abundance
    output = np.zeros((T2_num, angle_num, 3), dtype='float32')

    # obtain WAS values through T2 and angle values for the given abundance
    for j in range(T2_num):
        T2 = base+j
        logger.info(f"====T2 {T2} of {abundance} processing====")
        for k in range(angle_num):
            angle = 0.1*k
            output[j,k] = ftnmr.estimate_WAS([abundance, T2, angle])

    return output # to be copied to ATA_to_WAS[i]

with futures.ProcessPoolExecutor(max_workers=40) as executor:
    futures_list = []
    for i in range(abundance_num):
        future = executor.submit(A_to_WAS, i)
        futures_list.append([i, future])

    # copy results to ATA_to_WAS array
    for ind, future in futures_list:
        ATA_to_WAS[ind] = future.result()

# save the database
elapsed_time = (time.time() - start_time) / 60
print(f"code execution time: {elapsed_time:.2f} minutes")
np.save("database_ATA_WAS_sep_30.npy", ATA_to_WAS)

