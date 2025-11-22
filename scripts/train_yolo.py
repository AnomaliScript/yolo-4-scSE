"""
Train YOLOv8 Vertebra Detector
Uses auto-generated YOLO annotations from segmentation labels

REQUIREMENTS:
- Run auto_convert_to_yolo.py FIRST to create YOLO dataset
- Install: pip install ultralytics

Author: Brandon's Cervical Spine Project
Date: 2024-11-20
"""

from ultralytics import YOLO
from pathlib import Path


def train_vertebra_detector():
    """Train YOLOv8 on auto-converted vertebra annotations"""
    
    # ========== PATHS (should match auto_convert_to_yolo.py) ==========
    YOLO_DATA_DIR = Path(r"C:\Users\anoma\Downloads\yolo-4-scSE")
    DATA_YAML = YOLO_DATA_DIR / "data.yaml"
    OUTPUT_DIR = YOLO_DATA_DIR / "runs"
    # ==================================================================
    
    print(f"\n{'='*70}")
    print(f"YOLOV8 VERTEBRA DETECTOR TRAINING")
    print(f"{'='*70}")
    
    # Check if data.yaml exists (created by auto_convert_to_yolo.py)
    if not DATA_YAML.exists():
        print(f"\n❌ ERROR: {DATA_YAML} not found!")
        print(f"\nYou need to run auto_convert_to_yolo.py first!")
        print(f"That script creates:")
        print(f"  - {YOLO_DATA_DIR / 'images'}")
        print(f"  - {YOLO_DATA_DIR / 'labels'}")
        print(f"  - {DATA_YAML}")
        return
    
    # Check for training data
    train_images = YOLO_DATA_DIR / "images" / "train"
    train_labels = YOLO_DATA_DIR / "labels" / "train"
    
    if not train_images.exists() or not train_labels.exists():
        print(f"\n❌ ERROR: Training data not found!")
        print(f"Expected:")
        print(f"  {train_images}")
        print(f"  {train_labels}")
        print(f"\nRun auto_convert_to_yolo.py first!")
        return
    
    num_images = len(list(train_images.glob("*.jpg")))
    num_labels = len(list(train_labels.glob("*.txt")))
    
    print(f"\n✓ Dataset found:")
    print(f"  Training images: {num_images}")
    print(f"  Training labels: {num_labels}")
    print(f"  Data config: {DATA_YAML}")
    
    if num_images == 0 or num_labels == 0:
        print(f"\n❌ No training data! Run auto_convert_to_yolo.py")
        return
    
    if num_images != num_labels:
        print(f"\n⚠️  Warning: Image count != Label count")
        print(f"   Some images may not have annotations (okay if filtering was applied)")
    
    # Load pretrained YOLOv8x model
    print(f"\n{'='*70}")
    print("LOADING PRETRAINED MODEL")
    print(f"{'='*70}")
    print("Downloading YOLOv8x COCO weights (if not cached)...")
    
    model = YOLO('yolov8x.pt')
    
    print("✓ Model loaded")
    print(f"  Device: {model.device}")
    
    # Training configuration
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    
    EPOCHS = 300  # Can reduce to 100 for testing
    BATCH_SIZE = 8  # Adjust based on GPU memory (64 if you have 24GB+ GPU)
    IMG_SIZE = 640
    
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Learning rate: 0.001 (initial)")
    print(f"  Optimizer: Adam")
    
    # Start training
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print("This will take several hours...")
    print("Press Ctrl+C to stop early (model will save current progress)\n")
    
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=0,  # GPU 0 (use 'cpu' if no GPU)
            workers=8,
            project=str(OUTPUT_DIR),
            name='vertebra_detector',
            
            # Learning
            lr0=0.001,
            optimizer='Adam',
            patience=50,  # Early stopping
            
            # Medical image augmentation (conservative)
            hsv_h=0.01,      # Minimal hue variation
            hsv_s=0.3,       # Moderate saturation
            hsv_v=0.2,       # Moderate value
            degrees=5.0,     # Small rotation only
            translate=0.05,  # Minimal translation
            scale=0.3,       # Moderate scaling
            fliplr=0.0,      # NO left-right flip (spine is asymmetric!)
            flipud=0.0,      # NO up-down flip
            
            # Saving
            save=True,
            save_period=50,  # Save checkpoint every 50 epochs
            
            # Display
            verbose=True,
            plots=True,  # Generate training plots
        )
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        
        best_weights = OUTPUT_DIR / "vertebra_detector" / "weights" / "best.pt"
        last_weights = OUTPUT_DIR / "vertebra_detector" / "weights" / "last.pt"
        
        print(f"\n✓ Model saved:")
        print(f"  Best weights: {best_weights}")
        print(f"  Last weights: {last_weights}")
        print(f"\n✓ Training plots saved:")
        print(f"  {OUTPUT_DIR / 'vertebra_detector'}")
        
        # Validation
        print(f"\n{'='*70}")
        print("RUNNING VALIDATION")
        print(f"{'='*70}")
        
        val_results = model.val(data=str(DATA_YAML))
        
        print(f"\nValidation Metrics:")
        print(f"  mAP50: {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
        print(f"  Precision: {val_results.box.mp:.4f}")
        print(f"  Recall: {val_results.box.mr:.4f}")
        
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}")
        print(f"1. Check training plots in:")
        print(f"   {OUTPUT_DIR / 'vertebra_detector'}")
        print(f"\n2. Test detector on a few images:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('{best_weights}')")
        print(f"   results = model('path/to/test/image.jpg')")
        print(f"   results[0].show()")
        print(f"\n3. Integrate detector into nnUNet attention module")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted!")
        print(f"Latest weights saved to:")
        print(f"  {OUTPUT_DIR / 'vertebra_detector' / 'weights' / 'last.pt'}")
        print(f"\nYou can resume training later or use these weights.")


def main():
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ ultralytics package not found!")
        print("\nInstall with:")
        print("  pip install ultralytics")
        print("\nThen run this script again.")
        return
    
    train_vertebra_detector()


if __name__ == "__main__":
    main()