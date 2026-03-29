import tensorflow as tf
import pathlib
import itertools

# 1. Define your dataset path
data_dir_path = 'augmented_dataset'
data_dir = pathlib.Path(data_dir_path)
img_height, img_width = 224, 224
num_classes = 2

# 2. Define the exact parameters you want to test!
# The script will test EVERY combination of these.
batch_sizes = [16, 32]
learning_rates = [0.001, 0.0001]
dropouts = [0.0, 0.5] # 0.0 means no dropout, 0.5 means 50% dropout
epochs_list = [10, 20]

# Generate all possible combinations using itertools
combinations = list(itertools.product(batch_sizes, learning_rates, dropouts, epochs_list))
print(f"Total combinations to test: {len(combinations)}\n")

# Store the results to find the winner at the end
results = []
best_val_accuracy = 0.0
best_model_name = ""

# 3. Start the automated Grid Search loop
for i, (batch_size, lr, dropout_rate, epochs) in enumerate(combinations, 1):
    print(f"==================================================")
    print(f"TEST {i}/{len(combinations)}: Batch={batch_size}, LR={lr}, Dropout={dropout_rate}, Epochs={epochs}")
    print(f"==================================================")
    
    # Reload dataset because batch_size might have changed
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(img_height, img_width), batch_size=batch_size, crop_to_aspect_ratio=True)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(img_height, img_width), batch_size=batch_size, crop_to_aspect_ratio=True)

    # Build the model dynamically with the current parameters
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate), # Injects the current dropout rate
        tf.keras.layers.Dense(num_classes)
    ])

    # Compile with the current learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train the model (we hide verbose output so your console doesn't get flooded)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

    # Evaluate the final accuracy for this combination
    loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"--> RESULT: Validation Accuracy = {val_acc:.4f}\n")
    
    # Save the result
    results.append({
        'batch': batch_size, 'lr': lr, 'dropout': dropout_rate, 
        'epochs': epochs, 'val_acc': val_acc
    })
    
    # Save the physical model if it's the best one we've seen so far!
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_name = "best_breadboard_model.keras"
        model.save(best_model_name)
        print(f"*** New best model found and saved! ***\n")

# 4. Print the final summary scoreboard
print("\n##################################################")
print("               GRID SEARCH COMPLETE               ")
print("##################################################")
print("Top 3 Configurations:")

# Sort results by validation accuracy (highest first)
results.sort(key=lambda x: x['val_acc'], reverse=True)

for i, res in enumerate(results[:3], 1):
    print(f"{i}. Acc: {res['val_acc']:.4f} | Batch: {res['batch']}, LR: {res['lr']}, Drop: {res['dropout']}, Epochs: {res['epochs']}")

print(f"\nThe absolute best model has been saved to your folder as '{best_model_name}'!")