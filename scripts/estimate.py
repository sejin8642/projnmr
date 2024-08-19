#!/usr/bin/env python3

#SBATCH --job-name=monte_carlo

#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00 ## time format is DD-HH:MM:SS

## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --gres=gpu:NV-RTX-A4000:1 
#SBATCH --mem=32gb
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --output=means_vars.%A.out

# document and print out what this code this
print()
print("this script is to estimate random error using monte carlo method with dropout layers") 
print("in the model. The real data from BMRB is processed by DNN model to give mean and")
print("variance arrays. sample accuracy object file is loaded to ensure that only data that")
print("have at least one common peak for accuracy measurement are processed")
print()

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

# main modules
from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf
import ftnmr
import h5py

# objects
class CustomDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
def noise_gen(mean = 0, std_dev = 0.02, size=4096):
    return np.random.normal(mean, std_dev, size=size)

def main():
    # Retrieve the Slurm job allocation number from the environment variable
    job_id = os.environ.get('SLURM_JOBID')

    # load the model
    model_path = Path.home() / Path('data/jul_22_data/model_Jul_30.hdf5')
    model_load = keras.models.load_model(model_path, compile=False)

    # Get a list of all subdirectories of BMSE
    dir_path = Path.home() / Path('data/BMSE')
    subdirectories = [subdir.name for subdir in dir_path.iterdir() if subdir.is_dir()]
    subdirectories.sort()
    sample_paths = [str(dir_path/subdir) for subdir in subdirectories]

    # get NMR sample directories to process 
    with open("samples_aug_11_2024.pickle", "rb") as f:
        sample_accuracy = pickle.load(f)
    sample_dirs = [sample_paths[ind] for ind, samp in enumerate(sample_accuracy) if samp!=0]
    
    ### new model with dropouts ###
    # configuration
    input_length = None
    expand_layer = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
    GRU_unit = 32
    first_filter_num = 128
    transpose_layer = keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))
    second_filter_num = 64
    dropout_rate = 0.0001  # dropout rate

    # same as above execution of cells, but all the cells are merged
    seq_input = keras.layers.Input(shape=[input_length])

    expand_output = expand_layer(seq_input)

    GRU_output = keras.layers.Bidirectional(
        keras.layers.GRU(GRU_unit, return_sequences=True))(expand_output)

    expand_output2 = expand_layer(GRU_output)

    dropout_output1 = CustomDropout(dropout_rate)(expand_output2)

    cnn_layer1 = keras.layers.Conv2D(
        filters=first_filter_num,
        kernel_size=(1, 2*GRU_unit),
        activation='elu') # elu
    cnn_output1 = cnn_layer1(dropout_output1)

    transpose_output = transpose_layer(cnn_output1)

    dropout_output2 = CustomDropout(dropout_rate)(transpose_output)

    cnn_layer2 = keras.layers.Conv2D(
        filters=second_filter_num,
        kernel_size=(1, first_filter_num),
        activation='selu') # selu
    cnn2_output = cnn_layer2(dropout_output2)

    transpose2_output = transpose_layer(cnn2_output)

    dropout_output3 = CustomDropout(dropout_rate)(transpose2_output)

    cnn_layer3 = keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, second_filter_num),
        activation='LeakyReLU') # selu
    cnn3_output = cnn_layer3(dropout_output3)

    flat_output = keras.layers.Flatten()(cnn3_output)

    model_output = keras.layers.Add()([seq_input, flat_output])
    #############

    # get the new model with weights copied from the loaded model
    model = keras.Model(inputs=[seq_input], outputs=[model_output])
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Nadam(learning_rate=0.0005),
        metrics=['mse'])

    source_weights = model_load.get_weights()
    model.set_weights(source_weights)

    # preallocate numpy array to save
    data_length = 32768
    numpy_outputs = np.zeros((len(sample_dirs), 2, data_length), dtype=np.float32)

    # loop through NMR samples for model inference
    num_repeats = 1024
    for ind, sample_path in enumerate(sample_dirs):
        data, noise = ftnmr.bruker_data(sample_path)

        # reshape model input with noise
        model_input = np.array([data + noise])

        # repeat model inference with dropout
        output_samples = []
        for _ in range(num_repeats):
            model_output = model(model_input)
            model_numpy = model_output.numpy()[0] - noise
            output_samples.append(model_numpy)

        numpy_outputs[ind, 0] = np.mean(output_samples, axis=0)
        numpy_outputs[ind, 1] = np.var(output_samples, axis=0)

    # save model numpy outputs as hdf5
    with h5py.File(f"model_means_vars.{job_id}.hdf5", 'w') as f:
        # Create a dataset in the HDF5 file
        dset = f.create_dataset('model_outputs', data=numpy_outputs)

        # Optionally, add some metadata to the dataset
        dset.attrs['description'] = 'mean and vars arrays (dropout) for processed BMRB samples'
        dset.attrs['shape'] = numpy_outputs.shape
        dset.attrs['dtype'] = numpy_outputs.dtype.name
    
if __name__ == '__main__': 
    main()
