import os
import json
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import requests # For downloading images
import yaml

# =============================================================================
# ## 1. Configuration
# =============================================================================
# --- Source File ---
NDJSON_FILE = Path("./data/video/fintune_labels.ndjson")

# --- Destination Path ---
BASE_DEST_DIR = Path("data/processed/finetuning_v1")

# --- Final Class Mapping (must match our main dataset) ---
CLASS_MAPPING = {
    'bench': 0,
    'cone': 1,
    'sign': 2,
}

# --- Split Ratios ---
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1

# =============================================================================
# ## 2. Helper Functions
# =============================================================================
def create_yolo_structure(base_path: Path):
    """Creates the necessary folder structure for the fine-tuning dataset."""
    if base_path.exists():
        print(f"Destination folder {base_path} already exists. Deleting it for a clean build.")
        shutil.rmtree(base_path)
        
    # Create train, val, and test directories for both images and labels
    (base_path / "images/train").mkdir(parents=True, exist_ok=True)
    (base_path / "images/val").mkdir(parents=True, exist_ok=True)
    (base_path / "images/test").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/test").mkdir(parents=True, exist_ok=True)
    print("Created clean YOLO directory structure for fine-tuning.")

def convert_labelbox_to_yolo(json_data):
    """
    Parses a single line of the Labelbox NDJSON export and converts it to
    YOLO format.
    """
    try:
        image_url = json_data['data_row']['row_data']
        image_filename = json_data['data_row']['external_id']
        img_w = json_data['media_attributes']['width']
        img_h = json_data['media_attributes']['height']
        
        project_key = list(json_data['projects'].keys())[0]
        annotations = json_data['projects'][project_key]['labels'][0]['annotations']['objects']
        
        yolo_labels = []
        for obj in annotations:
            class_name = obj['name'].lower()
            if class_name not in CLASS_MAPPING:
                continue
            
            class_id = CLASS_MAPPING[class_name]
            bbox = obj['bounding_box']
            x_min, y_min = bbox['left'], bbox['top']
            box_w, box_h = bbox['width'], bbox['height']
            
            center_x = (x_min + box_w / 2) / img_w
            center_y = (y_min + box_h / 2) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h
            
            yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        return image_url, image_filename, yolo_labels

    except (KeyError, IndexError) as e:
        # print(f"Skipping line due to missing data: {e}")
        return None, None, None

def create_yaml_file(base_dir):
    """Creates the data.yaml file with absolute paths."""
    print("\nCreating data.yaml file...")
    yaml_data = {
        'train': str(base_dir.resolve() / "images/train"),
        'val': str(base_dir.resolve() / "images/val"),
        'test': str(base_dir.resolve() / "images/test"),
        'nc': len(CLASS_MAPPING),
        'names': list(CLASS_MAPPING.keys())
    }
    
    with open(base_dir / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print("data.yaml created successfully.")

# =============================================================================
# ## 3. Main Execution
# =============================================================================
if __name__ == "__main__":
    create_yolo_structure(BASE_DEST_DIR)
    
    with open(NDJSON_FILE, 'r') as f:
        lines = f.readlines()
        
    print(f"Found {len(lines)} annotations to process.")
    
    # --- Create temporary folders for initial processing ---
    temp_img_dir = BASE_DEST_DIR / "temp_images"
    temp_lbl_dir = BASE_DEST_DIR / "temp_labels"
    temp_img_dir.mkdir()
    temp_lbl_dir.mkdir()

    for line in tqdm(lines, desc="Processing annotations and downloading images"):
        json_data = json.loads(line)
        image_url, image_filename, yolo_labels = convert_labelbox_to_yolo(json_data)
        
        if not image_url or not yolo_labels:
            continue
            
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            with open(temp_img_dir / image_filename, 'wb') as f_out:
                shutil.copyfileobj(response.raw, f_out)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {image_url}: {e}")
            continue
            
        label_filename = Path(image_filename).stem + ".txt"
        with open(temp_lbl_dir / label_filename, 'w') as f_out:
            f_out.write("\n".join(yolo_labels))
    
    # --- Split the processed data into train, val, test ---
    print("\nSplitting data into train, validation, and test sets...")
    all_files = [p.stem for p in temp_img_dir.glob("*.*")]
    random.shuffle(all_files)
    
    train_split_idx = int(len(all_files) * TRAIN_RATIO)
    valid_split_idx = int(len(all_files) * (TRAIN_RATIO + VALID_RATIO))
    
    splits = {
        "train": all_files[:train_split_idx],
        "val": all_files[train_split_idx:valid_split_idx],
        "test": all_files[valid_split_idx:],
    }
    
    for split_name, file_list in splits.items():
        for filename_stem in tqdm(file_list, desc=f"Moving {split_name} files"):
            img_extension = next(temp_img_dir.glob(f"{filename_stem}.*")).suffix
            shutil.move(
                temp_img_dir / f"{filename_stem}{img_extension}",
                BASE_DEST_DIR / f"images/{split_name}/{filename_stem}{img_extension}"
            )
            shutil.move(
                temp_lbl_dir / f"{filename_stem}.txt",
                BASE_DEST_DIR / f"labels/{split_name}/{filename_stem}.txt"
            )

    # --- Clean up temporary folders ---
    shutil.rmtree(temp_img_dir)
    shutil.rmtree(temp_lbl_dir)

    # --- Create the final YAML file ---
    create_yaml_file(BASE_DEST_DIR)
            
    print("\nFine-tuning dataset created successfully!")
    print(f"Data is ready in: {BASE_DEST_DIR}")