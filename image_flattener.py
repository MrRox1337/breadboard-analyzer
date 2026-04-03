import os
import cv2
import pandas as pd

# ==========================================
# EXPORTING IMAGES TO CSV
# ==========================================
dataset_dir = 'augmented_dataset'
csv_filename = 'flattened_breadboards.csv'

# We downscale to 64x64 to prevent the CSV from becoming wildly large and crashing your RAM.
# 64 * 64 * 3 (RGB colors) = 12,288 columns per row.
img_size = 64 

data = []
labels = []

print("Starting Image Flattener Utility...")
print("Converting images to flattened 1D arrays...")

for label_name in ['FAIL', 'PASS']:
    folder_path = os.path.join(dataset_dir, label_name)
    
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_path}, not found.")
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            
            # Read image using OpenCV
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Resize and convert BGR (OpenCV default) to standard RGB
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # FLATTEN the 3D image matrix into a single 1D row
            flat_img = img.flatten()
            
            data.append(flat_img)
            labels.append(label_name)

print(f"Processed {len(data)} images. Building DataFrame...")

# Create a Pandas DataFrame (table)
df = pd.DataFrame(data)

# Make the last column the Pass/Fail label
df['label'] = labels

# Save to CSV
print(f"Saving to {csv_filename} (This might take a minute)...")
df.to_csv(csv_filename, index=False)
print(f"Success! CSV saved with {df.shape[0]} rows and {df.shape[1]} columns.")