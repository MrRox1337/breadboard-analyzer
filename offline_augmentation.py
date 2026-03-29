import os
from PIL import Image

# Define your input and output folders
input_dir = 'breadboard_dataset'
output_dir = 'augmented_dataset'

# The exact rotation angles you requested
angles = [0, 45, 90, 135, 180, 225, 270, 315]

# Process both folders
for category in ['PASS', 'FAIL']:
    in_path = os.path.join(input_dir, category)
    out_path = os.path.join(output_dir, category)
    
    # Create the output directories if they don't exist
    os.makedirs(out_path, exist_ok=True)
    
    # Make sure the input folder exists before trying to read it
    if not os.path.exists(in_path):
        print(f"Warning: Could not find folder {in_path}. Skipping.")
        continue

    # Loop through every image in the folder
    for filename in os.listdir(in_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(in_path, filename)
            
            with Image.open(img_path) as img:
                base_name = os.path.splitext(filename)[0]
                
                # --- BUG FIX ---
                # Convert RGBA (transparent) images to standard RGB before saving as JPEG
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                for angle in angles:
                    # 1. Rotate the image
                    # expand=False keeps the original image dimensions.
                    # fillcolor=(0,0,0) fills the empty corners created by 45-deg rotations with black.
                    rotated_img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0)) 
                    
                    # 2. Mirror the rotated image (Horizontal flip)
                    mirrored_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # 3. Save both to the new augmented dataset folder
                    rot_filename = f"{base_name}_rot{angle}.jpg"
                    mir_filename = f"{base_name}_rot{angle}_mirrored.jpg"
                    
                    rotated_img.save(os.path.join(out_path, rot_filename))
                    mirrored_img.save(os.path.join(out_path, mir_filename))
                    
    print(f"Finished augmenting images for {category}!")

print(f"\nAll done! Your new expanded dataset is located in the '{output_dir}' folder.")