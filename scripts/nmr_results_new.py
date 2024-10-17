#!/usr/bin/env python3

#SBATCH --job-name=NMR_accuracy
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00       # Time format: DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4G         # Memory per CPU
#SBATCH --output=accuracy.%A.out # STDOUT file

"""
This script generates accuracy of NMR samples from BMSE using a TensorFlow model and nmrglue.
Each sample is processed in parallel, with individual results and logs saved dynamically.
"""

import os
import sys
import time
import logging
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure custom script directories are in the Python path
sys.path.extend([
    str(Path.home() / 'gd' / 'projects' / 'ftnmr' / 'scripts'),
    str(Path.home() / 'gd' / 'projects' / 'projnmr' / 'scripts')
])

from ftnmr import NMR_result  # Ensure this module is accessible

def setup_directories(job_id):
    """
    Creates necessary directories for logs, results, and errors.
    
    Parameters:
    - job_id (str): The SLURM job ID.
    
    Returns:
    - dict: Paths to the created directories.
    """
    base_dir = Path.home()
    directories = {
        'log_dir': base_dir / f'logs.{job_id}',
        'results_dir': base_dir / f'results.{job_id}',
        'errors_dir': base_dir / f'errors.{job_id}'
    }
    for dir_path in directories.values():
        dir_path.mkdir(parent=True, exist_ok=True)
    return directories

def subprocess_task(path, model_path, database_path, dirs):
    """
    Processes a single NMR sample, computes accuracy, and saves the result.
    
    Parameters:
    - path (str): Path to the NMR sample directory.
    - model_path (Path): Path to the TensorFlow model file.
    - database_path (Path): Path to the ATA to WAS database file.
    - dirs (dict): Dictionary containing paths to log, results, and errors directories.
    
    Returns:
    - str: Sample name if successful, else None.
    """
    sample_name = Path(path).stem
    log_file = dirs['log_dir'] / f'{sample_name}.log'
    result_file = dirs['results_dir'] / f'{sample_name}.pkl'
    error_file = dirs['errors_dir'] / f'{sample_name}.txt'
    
    # Configure logger for this subprocess
    logger = logging.getLogger(sample_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info(f"Started processing sample: {sample_name}")
    
    start_time = time.time()
    try:
        # Instantiate the NMR_result object
        result = NMR_result(model_path, path, database_path)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f'Execution time: {execution_time:.2f} seconds')
        
        # Save the result to a pickle file
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f'Result saved to {result_file}')
        
        return sample_name  # Indicate success
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f'Execution time before error: {execution_time:.2f} seconds')
        logger.error(f'Error processing sample {sample_name}: {e}')
        
        # Save the exception details to an error file
        with open(error_file, 'w') as ef:
            ef.write(f'Error processing sample {sample_name}:\n{str(e)}')
        
        return None  # Indicate failure
    
    finally:
        # Clean up logger handlers to prevent memory leaks
        logger.removeHandler(handler)
        handler.close()

def main():
    # Retrieve SLURM job ID
    job_id = os.environ.get('SLURM_JOBID', 'local')  # 'local' for testing without SLURM
    
    # Define data directories
    data_parent_dir = Path.home() / 'data' / 'BMSE'
    
    # Collect all sample subdirectories
    sample_subdirs = sorted([subdir for subdir in data_parent_dir.iterdir() if subdir.is_dir()])
    sample_paths = [str(subdir) for subdir in sample_subdirs]
    
    if not sample_paths:
        print("No sample directories found. Exiting.")
        sys.exit(1)
    
    # Define paths for model and database
    model_path = Path.home() / 'data' / 'jul_22_data' / 'model_Jul_30.hdf5'
    database_path = Path.home() / 'data' / 'sep_30_database' / 'database_ATA_WAS_sep_30.npy'
    
    # Validate existence of model and database files
    if not model_path.is_file():
        print(f"Model file not found at {model_path}. Exiting.")
        sys.exit(1)
    if not database_path.is_file():
        print(f"Database file not found at {database_path}. Exiting.")
        sys.exit(1)
    
    # Set up directories for logs, results, and errors
    dirs = setup_directories(job_id)
    
    print(f"Job ID: {job_id}")
    print(f"Number of samples to process: {len(sample_paths)}")
    print(f"Logs directory: {dirs['log_dir']}")
    print(f"Results directory: {dirs['results_dir']}")
    print(f"Errors directory: {dirs['errors_dir']}")
    print("Starting parallel processing...")
    
    # Initialize ProcessPoolExecutor
    max_workers = 40  # Adjust based on system capabilities
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_sample = {
            executor.submit(subprocess_task, path, model_path, database_path, dirs): path
            for path in sample_paths
        }
        
        # Monitor task completion
        completed = 0
        failed = 0
        for future in as_completed(future_to_sample):
            path = future_to_sample[future]
            sample_name = Path(path).stem
            try:
                result = future.result()
                if result:
                    completed += 1
                    print(f"[{completed + failed}/{len(sample_paths)}] Completed: {sample_name}")
                else:
                    failed += 1
                    print(f"[{completed + failed}/{len(sample_paths)}] Failed: {sample_name}")
            except Exception as exc:
                failed += 1
                print(f"[{completed + failed}/{len(sample_paths)}] Exception for {sample_name}: {exc}")
    
    print("\nProcessing complete.")
    print(f"Total samples processed successfully: {completed}")
    print(f"Total samples failed: {failed}")
    print(f"Results saved in: {dirs['results_dir']}")
    print(f"Errors logged in: {dirs['errors_dir']}")

if __name__ == '__main__':
    main()

