import os
from PIL import Image
import numpy as np

def process_folder(folder_path):
    # Supported extensions to look for
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

    # Iterate through all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 1. Open image and convert to RGBA
                img = Image.open(file_path).convert("RGBA")
                data = np.array(img)

                # 2. Define the color range (202-209)
                # We check if R, G, and B are ALL within the range
                red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
                mask = (
                    (red >= 200) & (red <= 209) &
                    (green >= 200) & (green <= 209) &
                    (blue >= 200) & (blue <= 209)
                )

                # 3. If no pixels match, skip saving to save time/write cycles
                if not np.any(mask):
                    print(f"Skipping {filename} (no matching colors found).")
                    continue

                # 4. Apply transparency to matching pixels
                data[:,:,3][mask] = 0
                new_img = Image.fromarray(data)

                # 5. Handle Overwriting Logic
                # Check original extension
                file_root, file_ext = os.path.splitext(file_path)
                
                if file_ext.lower() in ['.jpg', '.jpeg', '.bmp']:
                    # These formats don't support alpha. We must save as PNG.
                    new_path = file_root + ".png"
                    new_img.save(new_path, format="PNG")
                    
                    # Remove the original file to "replace" it
                    os.remove(file_path)
                    print(f"Converted & Replaced: {filename} -> {os.path.basename(new_path)}")
                else:
                    # Supports alpha (e.g., PNG), just overwrite directly
                    new_img.save(file_path)
                    print(f"Overwritten: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# --- Usage ---
if __name__ == "__main__":
    target_folder = "dataset/val/images"
    
    if os.path.exists(target_folder):
        process_folder(target_folder)
        print("Batch processing complete.")
    else:
        print("The specified folder does not exist.")