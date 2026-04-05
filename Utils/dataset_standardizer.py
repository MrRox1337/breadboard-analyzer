import os

# Define the target dataset folder
dataset_dir = 'Dataset/breadboard_dataset'

# Process both categories
for category in ['PASS', 'FAIL']:
    folder_path = os.path.join(dataset_dir, category)
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Could not find folder {folder_path}. Skipping.")
        continue
        
    print(f"Renaming files in {folder_path}...")
    
    # Get a list of valid images
    valid_extensions = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    # Sort files to maintain consistency 
    files.sort()
    
    # Loop through and rename each file
    for index, filename in enumerate(files, start=1):
        # Extract the file extension (e.g., .jpg)
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Construct the new name (e.g., pass_photo_001.jpg or fail_photo_002.jpg)
        prefix = category.lower() # Converts 'PASS' to 'pass'
        new_name = f"{prefix}_photo_{index:03d}{file_extension}"
        
        # Get full paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        
    print(f"Successfully renamed {len(files)} files in the {category} folder.")

print("\nAll done! Your dataset is now cleanly named.")