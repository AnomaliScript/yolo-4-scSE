"""
Auto-Convert Segmentation Labels to YOLO Format

Takes your segmentation labels and:
1. Randomly selects 29 cases for training, 4 for validation
2. Extracts 2D slices from CT and labels
3. Converts vertebra masks to bounding boxes
4. Saves in YOLO format

Author: Brandon's Cervical Spine Project
Date: 2024-11-20
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import cv2
import random
from tqdm import tqdm
import shutil


def window_ct_image(image):
    """Apply bone windowing to CT"""
    img = np.clip(image, -1000, 1000)
    img = ((img + 1000) / 2000 * 255).astype(np.uint8)
    return img


def mask_to_yolo_bbox(mask_2d, class_id, img_width=640, img_height=640):
    """
    Convert 2D segmentation mask to YOLO bounding box.
    
    Args:
        mask_2d: 2D binary mask for one vertebra
        class_id: YOLO class ID (0=C1, 1=C2, ..., 6=C7)
        img_width, img_height: Image dimensions
    
    Returns:
        str: YOLO format line (class x_center y_center width height)
        None if mask too small
    """
    coords = np.argwhere(mask_2d > 0)
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Filter tiny boxes (noise from edge slices)
    width = x_max - x_min
    height = y_max - y_min
    
    if width < 20 or height < 20:  # Minimum 20 pixels
        return None
    
    # Convert to YOLO format (normalized 0-1)
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    box_width = width / img_width
    box_height = height / img_height
    
    # YOLO format: class x_center y_center width height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def process_case(ct_path, label_path, output_images_dir, output_labels_dir, num_slices=200):
    """
    Process one case: extract slices and convert to YOLO format.
    
    Args:
        ct_path: Path to CT scan
        label_path: Path to segmentation label
        output_images_dir: Where to save slice images
        output_labels_dir: Where to save YOLO label files
        num_slices: Number of slices per orientation
    
    Returns:
        int: Number of slices with valid annotations
    """
    # Load CT and label
    ct_nii = nib.load(str(ct_path))
    ct_volume = ct_nii.get_fdata()
    
    label_nii = nib.load(str(label_path))
    label_volume = label_nii.get_fdata().astype(np.int32)
    
    case_id = Path(ct_path).stem.replace('_0000', '')
    
    # Window CT
    ct_windowed = window_ct_image(ct_volume)
    
    valid_slices = 0
    
    # Process SAGITTAL slices
    sagittal_indices = np.linspace(
        int(ct_volume.shape[0] * 0.2),
        int(ct_volume.shape[0] * 0.8),
        num_slices,
        dtype=int
    )
    
    for idx in sagittal_indices:
        # Extract image slice
        img_slice = ct_windowed[idx, :, :]
        img_slice = np.rot90(img_slice, k=1)
        img_slice = cv2.resize(img_slice, (640, 640))
        
        # Extract label slice
        label_slice = label_volume[idx, :, :]
        label_slice = np.rot90(label_slice, k=1)
        label_slice = cv2.resize(label_slice, (640, 640), interpolation=cv2.INTER_NEAREST)
        
        # Convert each vertebra to YOLO bbox
        yolo_lines = []
        for vert_label in range(1, 8):  # C1-C7
            mask = (label_slice == vert_label)
            bbox_line = mask_to_yolo_bbox(mask, vert_label - 1, 640, 640)  # class_id = label - 1
            
            if bbox_line:
                yolo_lines.append(bbox_line)
        
        # Only save if we have at least one valid bbox
        if yolo_lines:
            slice_name = f"{case_id}_sag_{idx:03d}"
            
            # Save image
            cv2.imwrite(str(output_images_dir / f"{slice_name}.jpg"), img_slice)
            
            # Save YOLO label
            with open(output_labels_dir / f"{slice_name}.txt", 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            valid_slices += 1
    
    # Process CORONAL slices
    coronal_indices = np.linspace(
        int(ct_volume.shape[1] * 0.2),
        int(ct_volume.shape[1] * 0.8),
        num_slices,
        dtype=int
    )
    
    for idx in coronal_indices:
        img_slice = ct_windowed[:, idx, :]
        img_slice = np.rot90(img_slice, k=1)
        img_slice = cv2.resize(img_slice, (640, 640))
        
        label_slice = label_volume[:, idx, :]
        label_slice = np.rot90(label_slice, k=1)
        label_slice = cv2.resize(label_slice, (640, 640), interpolation=cv2.INTER_NEAREST)
        
        yolo_lines = []
        for vert_label in range(1, 8):
            mask = (label_slice == vert_label)
            bbox_line = mask_to_yolo_bbox(mask, vert_label - 1, 640, 640)
            
            if bbox_line:
                yolo_lines.append(bbox_line)
        
        if yolo_lines:
            slice_name = f"{case_id}_cor_{idx:03d}"
            
            cv2.imwrite(str(output_images_dir / f"{slice_name}.jpg"), img_slice)
            
            with open(output_labels_dir / f"{slice_name}.txt", 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            valid_slices += 1
    
    return valid_slices


def main():
    # ====== PATHS ======
    CT_DIR = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\v3\\imagesTr")
    LABEL_DIR = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\v3\\labelsTr")
    OUTPUT_DIR = Path(r"C:\\Users\\anoma\\Downloads\\yolo-4-scSE")
    
    NUM_TRAIN = 29
    NUM_VAL = 4
    
    print(f"\n{'='*70}")
    print(f"AUTO-CONVERT SEGMENTATIONS TO YOLO FORMAT")
    print(f"{'='*70}")
    print(f"CT directory: {CT_DIR}")
    print(f"Label directory: {LABEL_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train cases: {NUM_TRAIN}")
    print(f"Val cases: {NUM_VAL}")
    
    # Create output structure
    train_images_dir = OUTPUT_DIR / 'images' / 'train'
    train_labels_dir = OUTPUT_DIR / 'labels' / 'train'
    val_images_dir = OUTPUT_DIR / 'images' / 'val'
    val_labels_dir = OUTPUT_DIR / 'labels' / 'val'
    
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    ct_files = sorted(CT_DIR.glob('*_0000.nii.gz'))
    
    if len(ct_files) < NUM_TRAIN + NUM_VAL:
        print(f"\n❌ Not enough cases! Found {len(ct_files)}, need {NUM_TRAIN + NUM_VAL}")
        return
    
    print(f"\nFound {len(ct_files)} cases total")
    
    # Randomly select cases
    random.seed(42)  # Reproducible split
    selected_cases = random.sample(ct_files, NUM_TRAIN + NUM_VAL)
    
    train_cases = selected_cases[:NUM_TRAIN]
    val_cases = selected_cases[NUM_TRAIN:]
    
    print(f"\nSelected {NUM_TRAIN} training cases")
    print(f"Selected {NUM_VAL} validation cases")
    
    # Process training cases
    print(f"\n{'='*70}")
    print("PROCESSING TRAINING CASES")
    print(f"{'='*70}")
    
    train_slices = 0
    for ct_file in tqdm(train_cases, desc="Train"):
        case_id = ct_file.stem.replace('_0000.nii', '')
        # Label files don't have _0000 suffix
        label_file = LABEL_DIR / f"{case_id}.nii.gz"
        
        slices = process_case(ct_file, label_file, train_images_dir, train_labels_dir)
        train_slices += slices
    
    # Process validation cases
    print(f"\n{'='*70}")
    print("PROCESSING VALIDATION CASES")
    print(f"{'='*70}")
    
    val_slices = 0
    for ct_file in tqdm(val_cases, desc="Val"):
        case_id = ct_file.stem.replace('_0000.nii', '')
        # Label files don't have _0000 suffix
        label_file = LABEL_DIR / f"{case_id}.nii.gz"
        
        slices = process_case(ct_file, label_file, val_images_dir, val_labels_dir)
        val_slices += slices
    
    # Create data.yaml
    print(f"\n{'='*70}")
    print("CREATING YOLO CONFIG")
    print(f"{'='*70}")
    
    data_yaml = OUTPUT_DIR / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"train: {train_images_dir.absolute()}\n")
        f.write(f"val: {val_images_dir.absolute()}\n")
        f.write(f"nc: 7\n")
        f.write(f"names: ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']\n")
    
    print(f"✓ Created: {data_yaml}")
    
    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Training slices: {train_slices}")
    print(f"Validation slices: {val_slices}")
    print(f"Total slices: {train_slices + val_slices}")
    print(f"\nOutput structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  ({train_slices} images)")
    print(f"  │   └── val/    ({val_slices} images)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/  ({train_slices} labels)")
    print(f"  │   └── val/    ({val_slices} labels)")
    print(f"  └── data.yaml")
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Clone YOLOv7:")
    print(f"   cd {OUTPUT_DIR}")
    print("   git clone https://github.com/WongKinYiu/yolov7.git")
    print("\n2. Download pretrained weights:")
    print("   cd yolov7")
    print("   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-x.pt")
    print("\n3. Train:")
    print("   python train.py --batch 64 --epochs 500 --data ../data.yaml --weights yolov7-x.pt --name cervical_detector")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()