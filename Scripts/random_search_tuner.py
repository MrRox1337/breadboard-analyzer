import tensorflow as tf
import keras_tuner as kt
import pathlib
import os

# 1. Dataset Setup
data_dir = pathlib.Path('Dataset/augmented_dataset')
img_height, img_width = 224, 224
batch_size = 16

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size, crop_to_aspect_ratio=True)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size, crop_to_aspect_ratio=True)

# 2. Define a "Model Builder" function for the Tuner
def build_model(hp):
    """
    This function builds the CNN, but instead of hardcoding numbers,
    we let 'hp' (hyperparameters) pick random values from a range!
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
    
    # Let the tuner decide how many filters the first layer should have (between 16 and 64)
    hp_filters = hp.Int('conv_1_filter', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.Conv2D(hp_filters, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    
    # Standard static layers
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    
    # Let the tuner decide the size of the Dense layer
    hp_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    
    # Let the tuner test different dropout rates to prevent overfitting
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(hp_dropout))
    
    model.add(tf.keras.layers.Dense(2)) # 2 classes: PASS or FAIL

    # Let the tuner choose from three different learning rates
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# 3. Initialize the Random Search Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10, # Total random combinations we want to test
    executions_per_trial=1,
    directory='Logs',
    project_name='tuner_results'
)

print("Starting Random Search...\n")

# 4. Run the search
# We use a callback to stop training early if a model stops improving, saving time!
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(train_ds, validation_data=val_ds, epochs=15, callbacks=[stop_early])

# 5. Get the winning results
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n=========================================")
print("             SEARCH COMPLETE             ")
print("=========================================")
print(f"Optimal Filters in Layer 1: {best_hps.get('conv_1_filter')}")
print(f"Optimal Dense Units: {best_hps.get('dense_units')}")
print(f"Optimal Dropout Rate: {best_hps.get('dropout')}")
print(f"Optimal Learning Rate: {best_hps.get('learning_rate')}")

# Ensure Models directory exists
os.makedirs('Models', exist_ok=True)

# Save the absolute best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('Models/best_random_search_model.keras')
print("\nThe winning model has been saved to Models/best_random_search_model.keras!")