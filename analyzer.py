# ==========================================
# Import the required libraries
# ==========================================
import PIL # python image library
import PIL.Image
import tensorflow as tf
import pathlib # to manage the dataset path
import numpy as np

# ==========================================
# Define the data path
# ==========================================
# Assuming your dataset is in a folder named 'breadboard_dataset'
# Inside this folder, you should have two subfolders: 'PASS' and 'FAIL'
data_dir_path = 'breadboard_dataset'
data_dir = pathlib.Path(data_dir_path)
print(f"Dataset path: {data_dir}")

# Print the number of images in the dataset
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total images found: {image_count}")

# ==========================================
# Split into train and test datasets
# ==========================================
batch_size = 16 # Slightly larger batch size for stability, adjust as needed
img_height = 224 # Standard CNN input size (changed slightly from 244)
img_width = 224

# Train data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # Reserving 20% for validation
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Test/Validation data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Extract class names to verify ('PASS', 'FAIL')
class_names = train_ds.class_names
print(f"Classes detected: {class_names}")

# ==========================================
# Define the CNN Model
# ==========================================
num_classes = 2 # ONLY TWO LABELS: PASS and FAIL

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'), # Increased filters to learn better breadboard features
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# ==========================================
# Compile and Train the Model
# ==========================================
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

model.summary()

epochs = 10 # You may need to increase this if accuracy is low
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# ==========================================
# Evaluate Model
# ==========================================
print("\n--- Final Evaluation on Validation Data ---")
model.evaluate(val_ds)

# ==========================================
# Predict a single new image
# ==========================================
def test_new_breadboard(image_path):
    """
    Function to test a newly captured breadboard image.
    """
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    print(f"\n--- PREDICTION RESULT ---")
    print(f"This circuit is a {predicted_class} with a {confidence:.2f}% confidence.")
    
# Uncomment and update the path below to test a live photo:
# test_new_breadboard("path/to/test/board123.jpg")