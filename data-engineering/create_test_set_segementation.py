"""
This script creates a test dataset for segmentation by splitting the validation dataset in half.
The script moves approximately 50% of the validation images and their corresponding masks
to a new test directory, maintaining the paired relationship between images and masks.
"""

import os
import shutil
from pathlib import Path
import random

# --- Define Paths ---
BASE_PROCESSED_DIR = Path("data/processed/binary_segmentation")
VAL_DIR = BASE_PROCESSED_DIR / "val"
TEST_DIR = BASE_PROCESSED_DIR / "test"

# --- Create New Test Directories ---
TEST_IMG_DIR = TEST_DIR / "images"
TEST_MASK_DIR = TEST_DIR / "masks"
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
TEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

# --- Get list of validation images and shuffle them ---
val_images_list = os.listdir(VAL_DIR / "images")
random.shuffle(val_images_list)

# --- Calculate split point (approx. 50%) ---
split_idx = len(val_images_list) // 2
files_to_move = val_images_list[split_idx:]

print(f"Moving {len(files_to_move)} files from 'val' to 'test'...")

# --- Move the files ---
for filename in files_to_move:
    # Move the image
    shutil.move(
        src=VAL_DIR / "images" / filename,
        dst=TEST_IMG_DIR / filename
    )
    # Move the corresponding mask
    shutil.move(
        src=VAL_DIR / "masks" / filename,
        dst=TEST_MASK_DIR / filename
    )

print("Test set created successfully.")
print(f"New validation set size: {len(os.listdir(VAL_DIR / 'images'))}")
print(f"New test set size: {len(os.listdir(TEST_IMG_DIR))}")