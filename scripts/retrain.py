#!/usr/bin/env python3

#SBATCH --job-name=PHB_training

#SBATCH --partition=kill-shared
#SBATCH --time=1-12:00:00 ## time format is DD-HH:MM:SS

## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --gres=gpu:NV-RTX5000:1 
#SBATCH --mem=16gb
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --output=PHB_training.%A.out

# add directories in which necessary modules are located
import platform
from pathlib import Path
import sys

os_name = platform.system()
if os_name == 'Darwin':
	gd_path = str(Path.home()/'Library/Mobile Documents/com~apple~CloudDocs/gd')
elif os_name == 'Linux':
	gd_path = str(Path.home()/'gd')
else:
	raise ValueError('You are not on either Darwin or Linux OS')
project_path = gd_path + '/projects'
sys.path.insert(0, project_path + '/ftnmr/scripts')
sys.path.insert(0, project_path + '/projnmr/scripts')

# for error "not creating xla devices tf_xla_enable_xla_devices not set"
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# for error "Successfully opened dynamic library libcudart.so.10.1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# main fn modules
import time
from datetime import datetime
import ftnmr
import NN_models
import tensorflow as tf
from tensorflow import keras

def main():
    # keep track of training time
    start_time = time.time()

    # Retrieve the Slurm job allocation number from the environment variable
    job_id = os.environ.get('SLURM_JOBID')

    # load the data and split them into train and validation datasets
    files = os.listdir('.')
    isd = os.path.isdir 
    data_dir = [dt for dt in files if isd(dt) and dt.startswith('data')][0]
    PHB_datasets = ftnmr.sliced_spec_data(data_dir, batch_size=64, numpy_array=False)
    dataset_train = PHB_datasets[0]
    dataset_valid = PHB_datasets[1]

    # load model
    model_PHB = keras.models.load_model(
            'model_PHB.mar.31.2024.hdf5', 
            compile=False)

    # early stopping callback fn
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=10,
        min_delta=0,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    # save the best performant model by monitoring validation loss 
    weight_path = f"model_{job_id}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            weight_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min')

    # log files
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    csv_logger = tf.keras.callbacks.CSVLogger(f'./logs/training_log_{job_id}.csv', append=True)

    # model compilation
    model_PHB.compile(
        loss="mse",
        optimizer=keras.optimizers.Nadam(
            learning_rate=0.0001, 
            clipnorm=0.4,
            epsilon=1e-05),
        metrics=['mse'])

    # fit the model (takes longest)
    callbacks = [early_stopping, checkpoint, tensorboard, csv_logger]
    history = model_PHB.fit(
        dataset_train,
        validation_data=dataset_valid,
        epochs=64,
        callbacks=callbacks)

    # save history content
    history_hdf5_path = f'./logs/history_{job_id}.hdf5'
    ftnmr.save_history(history, history_hdf5_path) 

    # prints the total training time into sbatch output 
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

if __name__ == '__main__': 
    print("comments about the model training", end="\n\n")
    main()
