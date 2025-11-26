"""
Retrain/Continue Training YOLOv8 Vertebra Detector

This script continues training from your existing model.

Author: Brandon's Cervical Spine Project
Date: 2025-11-20
"""

from ultralytics import YOLO
from pathlib import Path


def continue_training():
    """Continue training from existing weights"""
    
    # Path to your trained model
    WEIGHTS = Path('runs/vertebra_detector_82/weights/best.pt')  # Use BEST performing model (37.2% mAP50)
    DATA_YAML = Path('data.yaml')
    
    print(f"\n{'='*70}")
    print("CONTINUE TRAINING YOLO DETECTOR")
    print(f"{'='*70}")
    print(f"Loading model from: {WEIGHTS}")
    print(f"Data config: {DATA_YAML}")
    
    if not WEIGHTS.exists():
        print(f"\n❌ Weights not found: {WEIGHTS}")
        print("Make sure you have a trained model first!")
        return
    
    if not DATA_YAML.exists():
        print(f"\n❌ Data config not found: {DATA_YAML}")
        return
    
    # Load existing model
    model = YOLO(str(WEIGHTS))
    
    print(f"\n✓ Model loaded")
    print(f"Starting additional training...\n")
    
    # Continue training
    results = model.train(
        data=str(DATA_YAML),
        epochs=200,          # Additional 100 epochs
        imgsz=640,
        batch=8,            # Reasonable for 42 training images (~5 batches/epoch)
        device=0,        # Use 'cpu' or 0 for GPU
        workers=0,           # IMPORTANT: 0 for Windows to avoid multiprocessing issues
        project='runs',
        cache='disk',       # Cache images in RAM for much faster training
        amp=True,          # Automatic mixed precision (saves memory)
        name='vertebra_detector_newest',  # New name to avoid overwriting
        
        # Training settings
        patience=30,        # Increased to tolerate oscillation
        optimizer='SGD',    # More stable than Adam for tiny datasets
        lr0=0.001,          # Lower learning rate for stability
        lrf=0.01,           # Reduce LR to 1% by end
        momentum=0.937,     # SGD momentum
        weight_decay=0.0005,
        dropout=0.1,        # Add dropout to prevent overfitting on tiny dataset

        # Keep same augmentation as before
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        
        verbose=True,
        plots=True,
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")

# =========================================================
# Optional: Function to start training from scratch
# =========================================================

def train_from_scratch():
    """Start fresh training from COCO pretrained weights"""
    
    DATA_YAML = Path('data.yaml')
    
    print(f"\n{'='*70}")
    print("TRAIN FROM SCRATCH")
    print(f"{'='*70}")
    print("Loading YOLOv8x COCO pretrained weights...")
    
    # Load COCO pretrained model
    model = YOLO('yolov8x.pt')
    
    print(f"Starting training...\n")
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=300,
        imgsz=640,
        batch=8,
        device=0,        # Use 'cpu' or 0 for GPU
        workers=0,           # IMPORTANT: 0 for Windows
        project='runs',
        name='vertebra_detector_new',
        
        lr0=0.001,
        patience=50,
        optimizer='Adam',
        
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        
        verbose=True,
        plots=True,
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    # Choose one:

    # Option 1: Continue from existing model (faster, builds on what you have)
    continue_training()

    # Option 2: Start completely fresh (uncomment to use)
    # train_from_scratch()