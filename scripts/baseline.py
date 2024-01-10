import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import itertools
from copy import deepcopy

# Initialize MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Define and compile your model inside the strategy's scope
with strategy.scope():
    model = create_your_model()  # Replace with your model creation function
    model.compile(optimizer='adam', loss='mse')  # Compile the model

# Define a ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    'multi_gpu_best_model.h5',  # Filepath to save the best model
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    save_best_only=True,  # Save only the best model
    mode='min',  # 'min' or 'max' depending on the metric (minimize loss)
    verbose=1  # Display messages
)

# Training loop
for epoch in range(num_epochs):
    # Create a dataset iterator
    train_iterable = tqdm(
        itertools.islice(dataset, num_batches), 
        bar_format=custom_bar_format,
        ncols=fixed_width,
        total=num_batches,
        leave=True
    )

    # Message before progress bar 
    epoch_str = str(epoch).zfill(z_num)
    train_iterable.set_description("Epoch [" + epoch_str + f"/{num_epochs}]")
    val_mse = "......"

    # Batch training loop
    for ind, (x_batch, y_batch) in enumerate(train_iterable):
        # Initialize gradients to zero
        gradients = deepcopy(grad_zeros)
        
        # Obtain gradients of mini batches for gradients accumulation
        for x_data, y_data in zip(x_batch, y_batch):
            with tf.GradientTape() as tape:
                y_pred = model(x_data, training=True)
                loss = loss_fn(y_data, y_pred)
    
            # Get smaller gradients and add them to accumulated gradients
            mini_grads = tape.gradient(loss, model.trainable_weights)
            gradients = [g1 + g2 for g1, g2 in zip(gradients, mini_grads)]

        # Optimize the weights from accumulated gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Reshape the batches so the model can process them
        S = x_batch.shape
        x_batch = tf.reshape(x_batch, (S[0] * S[1], *S[2:]))
        y_batch = tf.reshape(y_batch, (S[0] * S[1], *S[2:]))

        # Obtain MSE numerals to show after the progress bar
        y_pred = model(x_batch)
        mse_metric.update_state(y_batch, y_pred)
        mse = mods.float2SI(mse_metric.result().numpy())
        if ind + 1 == num_batches: 
            val_mse = update_metric(model, val_dataset, mse_metric)

        # Add training and validation MSE script after the progress bar
        train_iterable.set_postfix(mse=mse, val_mse=val_mse)

    # End of epoch training: reset metrics for later epoch loops
    mse_metric.reset_states()

    # Save the best model using the ModelCheckpoint callback
    checkpoint_callback.on_epoch_end(epoch, logs={'val_loss': val_mse})

