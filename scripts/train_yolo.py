"""
Step 2: Train YOLOv8 Vertebra Detector
======================================

Trains YOLOv8 on your annotated 2D slices.

Author: Brandon's Cervical Spine Project
Date: 2024-11-19
"""

from ultralytics import YOLO
from pathlib import Path
import yaml


def create_dataset_yaml(output_path):
    """
    Create YOLOv8 dataset configuration file.
    
    YOLOv8 expects a data.yaml file that specifies:
    - Path to training images
    - Path to validation images  
    - Number of classes
    - Class names
    """
    
    # ========== UPDATE THESE PATHS ==========
    BASE_DIR = r"C:\Users\anoma\Downloads\cervical-yolo\data"
    # ========================================
    
    config = {
        'path': BASE_DIR,  # Dataset root
        'train': 'images/train',  # Training images (relative to 'path')
        'val': 'images/val',      # Validation images (relative to 'path')
        
        # Number of classes
        'nc': 7,
        
        # Class names (must match your LabelImg annotations)
        'names': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created dataset config: {output_path}")
    return output_path


def train_yolo_detector(data_yaml, output_dir, epochs=500):
    """
    Train YOLOv8 detector.
    
    Args:
        data_yaml: Path to data.yaml config file
        output_dir: Where to save training results
        epochs: Number of training epochs (SpineCLUE used 500)
    """
    
    print("\n" + "="*70)
    print("TRAINING YOLOV8 VERTEBRA DETECTOR")
    print("="*70)
    
    # Load pretrained YOLOv8x model
    # 'x' = extra large model (most accurate, slower)
    # Other options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    print("\nLoading YOLOv8x pretrained weights...")
    model = YOLO('yolov8x.pt')
    
    print(f"\nStarting training:")
    print(f"  Epochs: {epochs}")
    print(f"  Data config: {data_yaml}")
    print(f"  Output: {output_dir}")
    print(f"  GPU: {'Available' if model.device.type == 'cuda' else 'Not available (will use CPU)'}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,              # Image size (640x640)
        batch=16,               # Batch size (adjust based on your GPU memory)
        device=0,               # GPU device (0 = first GPU, 'cpu' for CPU)
        workers=8,              # Number of data loading workers
        project=output_dir,     # Where to save results
        name='vertebra_detector',
        
        # Training hyperparameters (SpineCLUE settings)
        lr0=0.001,              # Initial learning rate
        patience=50,            # Early stopping patience
        
        # Data augmentation
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation
        hsv_v=0.4,              # HSV-Value
        degrees=10.0,           # Rotation (+/- degrees)
        translate=0.1,          # Translation (+/- fraction)
        scale=0.5,              # Scale (+/- gain)
        fliplr=0.5,             # Flip left-right probability
        
        # Model settings
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        
        # Save settings
        save=True,
        save_period=50,         # Save checkpoint every N epochs
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}/vertebra_detector")
    print(f"Best weights: {output_dir}/vertebra_detector/weights/best.pt")
    print("="*70)
    
    return results


def validate_model(weights_path, data_yaml):
    """
    Validate trained model on validation set.
    
    Args:
        weights_path: Path to best.pt weights
        data_yaml: Path to data.yaml
    """
    print("\n" + "="*70)
    print("VALIDATING MODEL")
    print("="*70)
    
    model = YOLO(weights_path)
    results = model.val(data=data_yaml)
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def main():
    """Main training workflow"""
    
    # ========== UPDATE THESE PATHS ==========
    DATA_DIR = r"C:\Users\anoma\Downloads\cervical-yolo\data"
    OUTPUT_DIR = r"C:\Users\anoma\Downloads\cervical-yolo\runs"
    # ========================================
    
    # Create dataset config
    data_yaml = Path(DATA_DIR) / "data.yaml"
    create_dataset_yaml(data_yaml)
    
    # Check if annotations exist
    train_labels = Path(DATA_DIR) / "labels" / "train"
    if not train_labels.exists() or not list(train_labels.glob("*.txt")):
        print("\n❌ ERROR: No annotations found!")
        print(f"   Expected location: {train_labels}")
        print("\n📋 YOU NEED TO ANNOTATE IMAGES FIRST:")
        print("   1. Download LabelImg: https://github.com/HumanSignal/labelImg")
        print("   2. Open LabelImg")
        print("   3. Click 'Open Dir' → select data/images/train")
        print("   4. Click 'Change Save Dir' → select data/labels/train")
        print("   5. Set format to 'YOLO' (not PascalVOC)")
        print("   6. Draw boxes around vertebrae, label as C1-C7")
        print("   7. Save (Ctrl+S) and move to next image (D key)")
        print("\n   Annotate at least 400 images (mix of sagittal and coronal)")
        return
    
    num_annotations = len(list(train_labels.glob("*.txt")))
    print(f"\n✓ Found {num_annotations} annotated images")
    
    if num_annotations < 100:
        print("\n⚠️  WARNING: Only {num_annotations} annotations found.")
        print("   SpineCLUE used ~8000 images. Recommend at least 400 for good results.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Train model
    print("\nStarting training (this will take several hours)...")
    train_yolo_detector(
        data_yaml=str(data_yaml),
        output_dir=OUTPUT_DIR,
        epochs=500  # Can reduce to 100 for faster testing
    )
    
    # Validate
    best_weights = Path(OUTPUT_DIR) / "vertebra_detector" / "weights" / "best.pt"
    if best_weights.exists():
        validate_model(str(best_weights), str(data_yaml))
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"\n📦 Trained model saved at:")
    print(f"   {best_weights}")
    print(f"\n📊 Training metrics and graphs:")
    print(f"   {OUTPUT_DIR}/vertebra_detector/")
    print(f"\n🎯 NEXT STEP: Run detection on all 426 cases")
    print(f"   python 02_run_detection.py")


if __name__ == "__main__":
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
        print("✓ ultralytics package found")
    except ImportError:
        print("❌ ultralytics not installed!")
        print("\nInstall with: pip install ultralytics")
        print("Then run this script again.")
        exit(1)
    
    main()