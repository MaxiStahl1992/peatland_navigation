import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_binary_class_mapping(mapping_csv_path: Path) -> list:
    """
    Identifies the GOOSE label IDs that correspond to a walkable path.

    Args:
        mapping_csv_path: Path to the goose_label_mapping.csv file.

    Returns:
        A list of integer IDs corresponding to path-related classes.
    """
    # Define the GOOSE classes that we consider to be a "path"
    path_classes = ["asphalt", "gravel", "sidewalk", "bikeway", "cobble"]
    
    df = pd.read_csv(mapping_csv_path)
    
    # Get the numeric 'label_key' for each path class name
    path_ids = df[df["class_name"].isin(path_classes)]["label_key"].tolist()
    
    print(f"Identified Path IDs: {path_ids}")
    return path_ids

def process_dataset_split_binary(source_dir: Path, dest_dir: Path, path_ids: list):
    """
    Processes a single dataset split (train, val, or test) for binary segmentation.
    """
    dest_images_dir = dest_dir / "images"
    dest_masks_dir = dest_dir / "masks"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_masks_dir.mkdir(parents=True, exist_ok=True)

    source_labels_dir = source_dir / "labels"
    source_images_dir = source_dir / "images"
    
    if not source_labels_dir.exists():
        print(f"No 'labels' directory found in {source_dir}, skipping split.")
        return

    label_paths = sorted(list(source_labels_dir.rglob("*_labelids.png")))
    print(f"\nFound {len(label_paths)} labels in {source_dir.name}. Processing...")

    for label_path in tqdm(label_paths, desc=f"Processing {source_dir.name}"):
        original_mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if original_mask is None:
            continue

        # --- Create the new binary mask ---
        # Initialize with 0 (Background)
        new_mask = np.zeros(original_mask.shape, dtype=np.uint8)
        
        # Find all pixels that match our path IDs and set them to 1 (Path)
        path_pixel_mask = np.isin(original_mask, path_ids)
        new_mask[path_pixel_mask] = 1 # Class 1 is our 'Path' class

        # --- Find the corresponding image file ---
        image_name_base = label_path.name.replace("_labelids.png", "")
        relative_dir = label_path.parent.relative_to(source_labels_dir)
        image_search_dir = source_images_dir / relative_dir
        
        try:
            image_path = next(image_search_dir.glob(f"{image_name_base}*.png"))
        except StopIteration:
            print(f"Warning: No matching image found for label {label_path}, skipping.")
            continue

        # --- Save the new mask and copy the original image ---
        output_basename = image_path.stem
        # Save the new binary mask
        cv2.imwrite(str(dest_masks_dir / f"{output_basename}.png"), new_mask)
        # Copy the original image
        shutil.copy(image_path, dest_images_dir / f"{output_basename}.png")

def main():
    """Main function to run the binary data preparation pipeline."""
    base_source_dir = Path("data/segmentation")
    # Save to a new directory to keep experiments separate
    base_dest_dir = Path("data/processed/binary_segmentation")

    mapping_csv = base_source_dir / "gooseEx_2d_train/goose_label_mapping.csv"
    if not mapping_csv.exists():
        print(f"Error: Could not find mapping file at {mapping_csv}")
        return

    path_ids = create_binary_class_mapping(mapping_csv)
    
    splits = ["gooseEx_2d_train", "gooseEx_2d_val"] # Test set has no labels

    for split_name in splits:
        source_directory = base_source_dir / split_name
        dest_name = split_name.split("_")[-1]
        dest_directory = base_dest_dir / dest_name
        if source_directory.exists():
            process_dataset_split_binary(source_directory, dest_directory, path_ids)
        else:
            print(f"Warning: Source directory not found, skipping: {source_directory}")

    print("\nBinary data preparation complete!")
    print(f"Processed data saved in: {base_dest_dir}")

if __name__ == "__main__":
    main()
