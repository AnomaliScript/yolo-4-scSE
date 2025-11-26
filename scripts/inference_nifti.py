"""
Inference on NIfTI volumes with trained YOLO vertebra detector

Takes a raw NIfTI volume, runs YOLO detection on each slice,
and returns bounding box annotations.

Author: Brandon's Cervical Spine Project
Date: 2025-11-25
"""

import nibabel as nib
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Tuple
import cv2


def load_yolo_model(weights_path: str = 'runs/vertebra_detector_114/weights/best.pt'):
    """Load trained YOLO model"""
    model = YOLO(weights_path)
    return model


def nifti_to_slices(nifti_path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI file and convert to slices

    Returns:
        slices: numpy array of image slices (H, W, num_slices)
        nifti_img: original NIfTI image object (for header/affine)
    """
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()

    # Normalize to 0-255 range for YOLO
    volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)

    return volume, nifti_img


def predict_on_volume(model, volume: np.ndarray, conf_threshold: float = 0.25) -> Dict[int, List[Dict]]:
    """
    Run YOLO inference on all slices of a volume

    Args:
        model: Loaded YOLO model
        volume: numpy array (H, W, num_slices) or (H, W, D)
        conf_threshold: confidence threshold for detections

    Returns:
        Dictionary mapping slice_index -> list of detections
        Each detection is a dict with: {class, conf, bbox: [x1, y1, x2, y2]}
    """
    detections = {}

    # Handle different volume orientations
    if volume.ndim == 3:
        num_slices = volume.shape[2]
    else:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    print(f"Processing {num_slices} slices...")

    for slice_idx in range(num_slices):
        # Extract slice
        slice_2d = volume[:, :, slice_idx]

        # Convert to RGB (YOLO expects 3 channels)
        if slice_2d.ndim == 2:
            slice_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
        else:
            slice_rgb = slice_2d

        # Run inference
        results = model(slice_rgb, conf=conf_threshold, verbose=False)

        # Parse detections
        slice_detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    detection = {
                        'class': int(box.cls[0].cpu().numpy()),
                        'class_name': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    }
                    slice_detections.append(detection)

        if slice_detections:
            detections[slice_idx] = slice_detections
            print(f"  Slice {slice_idx}: Found {len(slice_detections)} vertebrae")

    return detections


def annotate_volume(volume: np.ndarray, detections: Dict[int, List[Dict]],
                   draw_labels: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on volume slices

    Args:
        volume: Original volume (H, W, D)
        detections: Detection dictionary from predict_on_volume
        draw_labels: Whether to draw class labels

    Returns:
        Annotated volume as RGB (H, W, D, 3)
    """
    # Convert to RGB volume
    annotated = np.stack([volume] * 3, axis=-1)  # (H, W, D, 3)

    # Color map for vertebrae (C1-C7)
    colors = [
        (255, 0, 0),    # C1 - Red
        (255, 127, 0),  # C2 - Orange
        (255, 255, 0),  # C3 - Yellow
        (0, 255, 0),    # C4 - Green
        (0, 255, 255),  # C5 - Cyan
        (0, 0, 255),    # C6 - Blue
        (255, 0, 255),  # C7 - Magenta
    ]

    for slice_idx, slice_dets in detections.items():
        slice_img = annotated[:, :, slice_idx, :].copy()

        for det in slice_dets:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_idx = det['class']
            conf = det['confidence']
            class_name = det['class_name']

            # Get color for this class
            color = colors[class_idx % len(colors)]

            # Draw bounding box
            cv2.rectangle(slice_img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if draw_labels:
                label = f"{class_name} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(slice_img, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(slice_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)

        annotated[:, :, slice_idx, :] = slice_img

    return annotated


def save_annotated_nifti(annotated_volume: np.ndarray, original_nifti: nib.Nifti1Image,
                        output_path: str):
    """
    Save annotated volume as NIfTI file

    Args:
        annotated_volume: Annotated volume (H, W, D) or (H, W, D, 3)
        original_nifti: Original NIfTI image (for header/affine)
        output_path: Path to save annotated NIfTI
    """
    # If RGB, convert to grayscale or save first channel
    if annotated_volume.ndim == 4:
        annotated_volume = annotated_volume[:, :, :, 0]

    # Create new NIfTI with same affine/header as original
    annotated_nifti = nib.Nifti1Image(annotated_volume, original_nifti.affine, original_nifti.header)
    nib.save(annotated_nifti, output_path)
    print(f"Saved annotated NIfTI to: {output_path}")


def save_detections_json(detections: Dict[int, List[Dict]], output_path: str):
    """Save detection results as JSON"""
    import json
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"Saved detections JSON to: {output_path}")


def process_nifti(nifti_path: str,
                 weights_path: str = 'runs/vertebra_detector_114/weights/best.pt',
                 output_dir: str = None,
                 conf_threshold: float = 0.25,
                 save_annotated: bool = True,
                 save_json: bool = True) -> Tuple[Dict, np.ndarray]:
    """
    Main function: Process a NIfTI file and return annotated results

    Args:
        nifti_path: Path to input NIfTI file
        weights_path: Path to trained YOLO weights
        output_dir: Directory to save outputs (if None, uses same dir as input)
        conf_threshold: Detection confidence threshold
        save_annotated: Whether to save annotated NIfTI
        save_json: Whether to save detections as JSON

    Returns:
        detections: Dictionary of detections per slice
        annotated_volume: RGB volume with drawn bounding boxes
    """
    print(f"\n{'='*70}")
    print("YOLO VERTEBRA DETECTION ON NIFTI")
    print(f"{'='*70}")
    print(f"Input: {nifti_path}")
    print(f"Model: {weights_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(nifti_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("\nLoading YOLO model...")
    model = load_yolo_model(weights_path)

    # Load NIfTI
    print("Loading NIfTI file...")
    volume, nifti_img = nifti_to_slices(nifti_path)
    print(f"Volume shape: {volume.shape}")

    # Run inference
    print("\nRunning inference...")
    detections = predict_on_volume(model, volume, conf_threshold)
    print(f"\nTotal detections: {sum(len(dets) for dets in detections.values())} across {len(detections)} slices")

    # Annotate volume
    print("\nAnnotating volume...")
    annotated_volume = annotate_volume(volume, detections)

    # Save outputs
    input_name = Path(nifti_path).stem

    if save_annotated:
        output_nifti = output_dir / f"{input_name}_annotated.nii.gz"
        save_annotated_nifti(annotated_volume, nifti_img, str(output_nifti))

    if save_json:
        output_json = output_dir / f"{input_name}_detections.json"
        save_detections_json(detections, str(output_json))

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}\n")

    return detections, annotated_volume


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference_nifti.py <path_to_nifti_file> [weights_path] [conf_threshold]")
        print("\nExample:")
        print("  python inference_nifti.py data.nii.gz")
        print("  python inference_nifti.py data.nii.gz runs/vertebra_detector_newest/weights/best.pt 0.3")
        sys.exit(1)

    nifti_path = sys.argv[1]
    weights_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/vertebra_detector114/weights/best.pt'
    conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

    # Process
    detections, annotated = process_nifti(nifti_path, weights_path, conf_threshold=conf_threshold)
