"""
This script prepares multi-class segmentation data by converting the GOOSE dataset's
detailed class structure into a simplified set of classes relevant for navigation:
PATH, NATURAL_GROUND, TREE, VEGETATION, and IGNORE. The script maintains the spatial
relationships while reducing the complexity of the segmentation task.
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_class_mapping(mapping_csv_path: Path) -> tuple[dict, dict]:
    """
    Creates the mapping from GOOSE class names to simplified navigation classes.
    
    This function defines a simplified class structure for navigation purposes
    by grouping the detailed GOOSE classes into broader categories:
    - PATH: surfaces suitable for navigation
    - NATURAL_GROUND: unpaved but potentially traversable surfaces
    - TREE: tree-related objects that should be avoided
    - VEGETATION: various types of vegetation
    - IGNORE: any other classes
    
    Args:
        mapping_csv_path (Path): Path to the GOOSE dataset's class mapping CSV

    Returns:
        tuple[dict, dict]:
            - First dict: Maps class names to new class IDs
            - Second dict: Maps new class IDs to lists of original GOOSE class IDs
    """
    target_class_definitions = {
        "PATH": ["asphalt", "gravel", "sidewalk", "bikeway", "cobble"],
        "NATURAL_GROUND": ["soil", "moss", "leaves", "debris"],
        "TREE": ["tree_trunk", "tree_crown", "tree_root"],
        "VEGETATION": ["low_grass", "high_grass", "bush", "hedge", "scenery_vegetation", "forest", "crops"],
    }
    new_class_names = {
        "PATH": 0,
        "NATURAL_GROUND": 1,
        "TREE": 2,
        "VEGETATION": 3,
        "IGNORE": 4,
    }
    df = pd.read_csv(mapping_csv_path)
    id_to_goose_ids_mapping = {}
    for new_name, old_names_list in target_class_definitions.items():
        new_id = new_class_names[new_name]
        goose_ids = df[df["class_name"].isin(old_names_list)]["label_key"].tolist()
        id_to_goose_ids_mapping[new_id] = goose_ids
    print("Successfully created class mapping.")
    return new_class_names, id_to_goose_ids_mapping

def process_dataset_split(source_dir: Path, dest_dir: Path, new_classes: dict, id_mapping: dict):
    """
    Processes a single dataset split (train, val, or test) to create simplified segmentation masks.
    
    This function converts the detailed GOOSE segmentation masks into simplified
    versions using the new class mapping. It maintains the directory structure
    and handles image-mask pairs appropriately.

    Args:
        source_dir (Path): Directory containing the source dataset split
        dest_dir (Path): Directory where the processed dataset will be saved
        new_classes (dict): Mapping of class names to new class IDs
        id_mapping (dict): Mapping of new class IDs to lists of original GOOSE class IDs

    Note:
        The function expects a specific directory structure with 'images' and 'labels'
        subdirectories in the source path. The destination will mirror this structure
        but with masks instead of labels.
    """
    dest_images_dir = dest_dir / "images"
    dest_masks_dir = dest_dir / "masks"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_masks_dir.mkdir(parents=True, exist_ok=True)

    # Correctly locate the source labels and images directories within the split
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

        new_mask = np.full(original_mask.shape, new_classes["IGNORE"], dtype=np.uint8)
        for new_id, goose_ids in id_mapping.items():
            mask_for_class = np.isin(original_mask, goose_ids)
            new_mask[mask_for_class] = new_id

        # --- CORRECTED LOGIC for finding the image file ---
        # 1. Get the unique base name of the label file (e.g., 'alice_scenario02_..._0000_...')
        image_name_base = label_path.name.replace("_labelids.png", "")

        # 2. Determine the relative path of the label file (e.g., 'val/alice_scenario02')
        relative_dir = label_path.parent.relative_to(source_labels_dir)
        
        # 3. Construct the search directory for the corresponding image
        image_search_dir = source_images_dir / relative_dir
        
        # 4. Find the image by its base name. We use a glob (*) because the suffix can vary
        #    (e.g., _camera_left.png, _front.png)
        try:
            image_path = next(image_search_dir.glob(f"{image_name_base}*.png"))
        except StopIteration:
            print(f"Warning: No matching image found for label {label_path}, skipping.")
            continue

        # --- Save the new mask and copy the original image ---
        output_basename = image_path.stem
        cv2.imwrite(str(dest_masks_dir / f"{output_basename}.png"), new_mask)
        shutil.copy(image_path, dest_images_dir / f"{output_basename}.png")

def main():
    """
    Main function to run the entire data preparation pipeline.
    
    This function orchestrates the complete process of creating a simplified
    segmentation dataset:
    1. Loads and processes the GOOSE class mapping
    2. Creates new class definitions for navigation purposes
    3. Processes each dataset split to create new segmentation masks
    4. Saves the processed dataset in a new directory structure
    
    Note:
        Currently processes only the training split as validation and test
        sets may have different requirements or structures.
    """
    base_source_dir = Path("data/segmentation")
    base_dest_dir = Path("data/processed/segmentation")

    # The label mapping CSV is present in the training set directory.
    mapping_csv = base_source_dir / "gooseEx_2d_train/goose_label_mapping.csv"

    if not mapping_csv.exists():
        print(f"Error: Could not find mapping file at {mapping_csv}")
        return

    new_class_names, id_to_goose_ids_mapping = create_class_mapping(mapping_csv)
    
    # Process only train and val, as test has no labels.
    splits = ["gooseEx_2d_train"] #"gooseEx_2d_val"

    for split_name in splits:
        source_directory = base_source_dir / split_name
        dest_name = split_name.split("_")[-1]
        dest_directory = base_dest_dir / dest_name
        if source_directory.exists():
            process_dataset_split(source_directory, dest_directory, new_class_names, id_to_goose_ids_mapping)
        else:
            print(f"Warning: Source directory not found, skipping: {source_directory}")

    print("\nData preparation complete!")
    print(f"Processed data saved in: {base_dest_dir}")

if __name__ == "__main__":
    main()
