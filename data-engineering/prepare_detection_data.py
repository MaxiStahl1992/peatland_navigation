"""
This script prepares a unified YOLO-format detection dataset by combining multiple individual
datasets (benches, cones, and signs) into a single dataset with consistent class IDs.
The script handles merging the datasets, remapping class IDs, and creating appropriate
train/validation/test splits while maintaining proper YOLO directory structure.
"""

import os
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import yaml

# --- Source Data Paths (Update these to match your new folder names) ---
BASE_SOURCE_DIR = Path("./data/detection") # Assuming a new parent folder for the new data
BENCH_DIR = BASE_SOURCE_DIR / "Bench.v1i.yolov11" 
CONES_DIR = BASE_SOURCE_DIR / "Cones.v1i.yolov11"
SIGNS_DIR = BASE_SOURCE_DIR / "Signs.v1i.yolov11"

# --- Destination Path ---
BASE_DEST_DIR = Path("data/processed/detection")

# --- Final Class Mapping ---
CLASS_MAPPING = {
    'bench': 0,
    'cone': 1,
    'sign': 2,
}

# --- Split Ratios ---
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def create_yolo_structure(base_path: Path):
    """
    Creates the necessary folder structure for the final YOLO dataset.
    
    This function sets up a clean directory structure required for YOLO training:
    - images/train
    - images/val
    - images/test
    - labels/train
    - labels/val
    - labels/test
    
    If the destination directory already exists, it will be removed to ensure
    a clean build.

    Args:
        base_path (Path): The root directory where the YOLO structure will be created
    """
    if base_path.exists():
        print(f"Destination folder {base_path} already exists. Deleting it for a clean build.")
        shutil.rmtree(base_path)
        
    (base_path / "images/train").mkdir(parents=True, exist_ok=True)
    (base_path / "images/val").mkdir(parents=True, exist_ok=True)
    (base_path / "images/test").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/test").mkdir(parents=True, exist_ok=True)
    print("Created clean YOLO directory structure.")

def collect_all_files(source_dir: Path, dataset_name: str, target_class_id: int):
    """
    Collects all image and label pairs from a dataset, ignoring original splits.
    
    This function recursively searches through a dataset directory to find all
    matching image-label pairs, regardless of their original split (train/val/test).
    The function ensures that each label file has a corresponding image file.

    Args:
        source_dir (Path): The root directory of the source dataset
        dataset_name (str): Name of the dataset (e.g., 'bench', 'cone', 'sign')
        target_class_id (int): The new class ID to assign to all instances

    Returns:
        list[tuple]: List of tuples containing (image_path, label_path, target_class_id, dataset_name)
        
    Note:
        The function silently skips label files that don't have matching image files.
    """
    print(f"Collecting files from {dataset_name.capitalize()}...")
    all_pairs = []
    
    # Use rglob to find all label files, regardless of which split folder they are in
    for lbl_path in tqdm(list(source_dir.rglob("labels/*.txt")), desc=f"  - Finding pairs for {dataset_name}"):
        # Find corresponding image (could be .jpg, .png, etc.)
        img_stem = lbl_path.stem
        img_dir = lbl_path.parent.parent / "images"
        
        try:
            img_path = next(img_dir.glob(f"{img_stem}.*"))
            all_pairs.append((img_path, lbl_path, target_class_id, dataset_name))
        except StopIteration:
            # print(f"Warning: No image found for label {lbl_path}")
            continue
            
    return all_pairs

def create_yaml_file():
    """
    Creates the data.yaml file with absolute paths for YOLO training.
    
    This function generates a YAML configuration file that specifies:
    - Paths to train/val/test image directories
    - Number of classes
    - Class names
    
    The paths are stored as absolute paths to ensure compatibility
    across different working directories.

    Note:
        The file is saved as 'data.yaml' in the base destination directory.
    """
    print("\nCreating data.yaml file...")
    yaml_data = {
        'train': str(BASE_DEST_DIR.resolve() / "images/train"),
        'val': str(BASE_DEST_DIR.resolve() / "images/val"),
        'test': str(BASE_DEST_DIR.resolve() / "images/test"),
        'nc': len(CLASS_MAPPING),
        'names': list(CLASS_MAPPING.keys())
    }
    
    with open(BASE_DEST_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print("data.yaml created successfully with absolute paths.")


if __name__ == "__main__":
    create_yolo_structure(BASE_DEST_DIR)
    
    # --- Step 1: Collect all files from all datasets into one master list ---
    master_file_list = []
    master_file_list.extend(collect_all_files(BENCH_DIR, "bench", CLASS_MAPPING['bench']))
    master_file_list.extend(collect_all_files(CONES_DIR, "cone", CLASS_MAPPING['cone']))
    master_file_list.extend(collect_all_files(SIGNS_DIR, "sign", CLASS_MAPPING['sign']))
    
    print(f"\nTotal image/label pairs collected: {len(master_file_list)}")
    
    # --- Step 2: Shuffle and split the master list ---
    random.shuffle(master_file_list)
    
    train_split_idx = int(len(master_file_list) * TRAIN_RATIO)
    valid_split_idx = int(len(master_file_list) * (TRAIN_RATIO + VALID_RATIO))
    
    splits = {
        "train": master_file_list[:train_split_idx],
        "val": master_file_list[train_split_idx:valid_split_idx],
        "test": master_file_list[valid_split_idx:],
    }
    
    print(f"Train split size: {len(splits['train'])}")
    print(f"Validation split size: {len(splits['val'])}")
    print(f"Test split size: {len(splits['test'])}")

    # --- Step 3: Copy files and remap labels for each split ---
    print("\nCopying files and remapping labels...")
    for split_name, file_list in splits.items():
        for img_path, lbl_path, target_id, dataset_name in tqdm(file_list, desc=f"Processing {split_name} split"):
            
            # Prepend dataset name to avoid filename collisions
            new_stem = f"{dataset_name}_{img_path.stem}"
            
            # Copy image
            shutil.copy(img_path, BASE_DEST_DIR / f"images/{split_name}/{new_stem}{img_path.suffix}")
            
            # Read, remap class ID, and write new label file
            with open(lbl_path, 'r') as f_in, open(BASE_DEST_DIR / f"labels/{split_name}/{new_stem}.txt", 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    parts[0] = str(target_id)
                    f_out.write(" ".join(parts) + "\n")
                    
    create_yaml_file()
    
    print("\nAll datasets processed and combined successfully!")
    print(f"New dataset ready at: {BASE_DEST_DIR}")