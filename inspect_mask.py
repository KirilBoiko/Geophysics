import os
import numpy as np
from PIL import Image

DATA_DIR = "/Users/kirilboiko/Downloads/netherlands_f3_dataset"
MASK_DIR = os.path.join(DATA_DIR, "masks/inline_mask")

# Find first mask
masks = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
if not masks:
    print("No masks found!")
    exit()

mask_path = os.path.join(MASK_DIR, masks[0])
print(f"Loading mask: {mask_path}")

# Load raw
mask = Image.open(mask_path)
print(f"Format: {mask.format}, Mode: {mask.mode}, Size: {mask.size}")

# Convert to numpy
mask_np = np.array(mask)
unique_vals = np.unique(mask_np)

print(f"Unique pixel values (raw 0-255): {unique_vals}")
print(f"Number of unique classes: {len(unique_vals)}")

# Validate if they map cleanly to 0, 1, 2...
if len(unique_vals) < 10:
    print("Likely class labels.")
else:
    print("Likely continuous/probabilistic or many classes.")
