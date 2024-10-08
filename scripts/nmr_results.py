#!/usr/bin/env python3

#SBATCH --job-name=NMR_accuracy
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4G ## max amount of memory per node you require
#SBATCH --output=accuracy.%A.out

print()
print("python script file name: ", __file__)
print()
'''
This script generates accuracy of NMR samples from BMSE
'''

from pathlib import Path
import sys
import os
sys.path.insert(1, str(Path.home()/'gd'/'projects'/'ftnmr'/'scripts'))
sys.path.insert(1, str(Path.home()/'gd'/'projects'/'projnmr'/'scripts'))

import logging
from ftnmr import NMR_result
import time
from concurrent import futures
import pickle

# job id and data parent directory
job_id = os.environ.get('SLURM_JOBID')
dir_path = Path.home() / Path('data/BMSE')

# Get a list of all subdirectories of BMSE data
subdirectories = [subdir.name for subdir in dir_path.iterdir() if subdir.is_dir()]
subdirectories.sort()
sample_paths = [str(dir_path / subdir) for subdir in subdirectories]

# paths for model and database
model_path = Path.home() / Path('data/jul_22_data/model_Jul_30.hdf5')
database_file = Path('data/sep_30_database/database_ATA_WAS_sep_30.npy')
database_path = Path.home() / database_file

def subprocess(path):
    # Create a log directory if it doesn't exist
    log_dir = Path(f'logs.{job_id}')
    log_dir.mkdir(exist_ok=True)

    # Create a unique log file name for this subprocess
    log_file = log_dir / f'{Path(path).name}.log'

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # process NMR_result with time
    start_time = time.time()
    result = NMR_result(model_path, path, database_path)
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f'Execution time: {execution_time:.2f} seconds')
    logger.removeHandler(handler)

    return result

results = [None]*len(sample_paths)

# parallel looping
with futures.ProcessPoolExecutor(max_workers=40) as executor:
    futures_list = []
    for i, path in enumerate(sample_paths):
        future = executor.submit(subprocess, path)
        futures_list.append([i, future])

    # pass results to results list
    for ind, future in futures_list:
        results[ind] = future.result()

with open(f'NMR_results.{job_id}.pkl', 'wb') as f:
    pickle.dump(results, f)

