"""
Step 3: Run Detection on All Cases
===================================

Uses trained YOLO detector to find vertebrae in all 426 cases,
then converts 2D detections to 3D bounding boxes with clustering.

Author: Brandon's Cervical Spine Project
Date: 2024-11-19
"""

from ultralytics import YOLO
import nibabel as nib
import numpy as np
from pathlib import Path
import cv2
import json
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict


class VertebraDetector:
    """
    Run YOLO detection on CT scans and convert to 3D coordinates.
    
    Follows SpineCLUE methodology:
    1. Extract 2D slices (sagittal + coronal)
    2. Run YOLO detection on each slice
    3. Cluster 2D detections into 3D vertebra locations
    """
    
    def __init__(self, model_path):
        """
        Args:
            model_path: Path to trained YOLO weights (best.pt)
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Class names (must match training)
        self.class_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    def normalize_ct_for_display(self, ct_volume):
        """Normalize CT to 0-255 for YOLO"""
        windowed = np.clip(ct_volume, -1000, 1000)
        normalized = ((windowed + 1000) / 2000 * 255).astype(np.uint8)
        return normalized
    
    def extract_and_detect_slices(self, ct_path, num_slices=200):
        """
        Extract slices and run YOLO detection on each.
        
        Args:
            ct_path: Path to CT NIfTI file
            num_slices: Number of slices per orientation
        
        Returns:
            dict: Detections organized by vertebra class
        """
        # Load CT
        ct_nii = nib.load(str(ct_path))
        ct_volume = ct_nii.get_fdata()
        spacing = ct_nii.header.get_zooms()
        
        # Normalize
        ct_normalized = self.normalize_ct_for_display(ct_volume)
        
        # Store all detections
        detections = defaultdict(list)  # {vertebra_name: [detection_dicts]}
        
        # ===== SAGITTAL SLICES =====
        X_dim = ct_volume.shape[0]
        sagittal_indices = np.linspace(
            int(X_dim * 0.2), int(X_dim * 0.8), num_slices, dtype=int
        )
        
        for x_idx in sagittal_indices:
            slice_img = ct_normalized[x_idx, :, :]
            slice_img = cv2.resize(slice_img, (640, 640))
            
            # Convert to RGB (YOLO expects 3 channels)
            slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)
            
            # Run YOLO detection
            results = self.model.predict(slice_rgb, verbose=False, conf=0.25)
            
            # Parse detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    # Convert back to original volume coordinates
                    # Scale from 640x640 back to original slice size
                    scale_y = ct_volume.shape[1] / 640
                    scale_z = ct_volume.shape[2] / 640
                    
                    y1, y2 = xyxy[0] * scale_y, xyxy[2] * scale_y
                    z1, z2 = xyxy[1] * scale_z, xyxy[3] * scale_z
                    
                    vertebra_name = self.class_names[cls_id]
                    
                    detections[vertebra_name].append({
                        'orientation': 'sagittal',
                        'slice_index': x_idx,
                        'dimension': 'X',
                        'bbox_2d': [y1, y2, z1, z2],  # In original coords
                        'center_2d': [(y1+y2)/2, (z1+z2)/2],
                        'confidence': conf,
                        'spacing': spacing
                    })
        
        # ===== CORONAL SLICES =====
        Y_dim = ct_volume.shape[1]
        coronal_indices = np.linspace(
            int(Y_dim * 0.2), int(Y_dim * 0.8), num_slices, dtype=int
        )
        
        for y_idx in coronal_indices:
            slice_img = ct_normalized[:, y_idx, :]
            slice_img = cv2.resize(slice_img, (640, 640))
            slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)
            
            results = self.model.predict(slice_rgb, verbose=False, conf=0.25)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    scale_x = ct_volume.shape[0] / 640
                    scale_z = ct_volume.shape[2] / 640
                    
                    x1, x2 = xyxy[0] * scale_x, xyxy[2] * scale_x
                    z1, z2 = xyxy[1] * scale_z, xyxy[3] * scale_z
                    
                    vertebra_name = self.class_names[cls_id]
                    
                    detections[vertebra_name].append({
                        'orientation': 'coronal',
                        'slice_index': y_idx,
                        'dimension': 'Y',
                        'bbox_2d': [x1, x2, z1, z2],
                        'center_2d': [(x1+x2)/2, (z1+z2)/2],
                        'confidence': conf,
                        'spacing': spacing
                    })
        
        return detections, ct_volume.shape, spacing
    
    def cluster_to_3d(self, detections_2d, volume_shape, spacing):
        """
        Cluster 2D detections into 3D vertebra locations.
        
        Uses DBSCAN clustering (SpineCLUE's dual-factor density clustering).
        
        Args:
            detections_2d: Dict of 2D detections per vertebra
            volume_shape: Original volume shape (X, Y, Z)
            spacing: Voxel spacing (mm)
        
        Returns:
            dict: 3D vertebra information for nnUNet
        """
        vertebrae_3d = {}
        
        for vertebra_name, detections_list in detections_2d.items():
            if not detections_list:
                vertebrae_3d[vertebra_name] = {'present': False}
                continue
            
            # Collect all detection centers
            centers_3d = []
            
            for det in detections_list:
                if det['orientation'] == 'sagittal':
                    # Sagittal: x is fixed (slice_index), y and z from bbox
                    x = det['slice_index']
                    y, z = det['center_2d']
                    centers_3d.append([x, y, z, det['confidence']])
                
                elif det['orientation'] == 'coronal':
                    # Coronal: y is fixed (slice_index), x and z from bbox
                    y = det['slice_index']
                    x, z = det['center_2d']
                    centers_3d.append([x, y, z, det['confidence']])
            
            centers_3d = np.array(centers_3d)
            
            # Cluster using DBSCAN
            # eps = spatial proximity threshold (in voxels)
            # min_samples = minimum detections to form a cluster
            clustering = DBSCAN(eps=15, min_samples=5).fit(centers_3d[:, :3])
            
            # Find the largest cluster (most detections)
            labels = clustering.labels_
            if len(set(labels)) == 1 and labels[0] == -1:
                # No valid clusters (all noise)
                vertebrae_3d[vertebra_name] = {'present': False}
                continue
            
            # Get largest cluster
            label_counts = {}
            for label in labels:
                if label != -1:  # Ignore noise
                    label_counts[label] = (labels == label).sum()
            
            if not label_counts:
                vertebrae_3d[vertebra_name] = {'present': False}
                continue
            
            largest_cluster_label = max(label_counts, key=label_counts.get)
            cluster_mask = labels == largest_cluster_label
            cluster_centers = centers_3d[cluster_mask]
            
            # Compute 3D centroid
            centroid = cluster_centers[:, :3].mean(axis=0)
            
            # Compute 3D bounding box (rough estimate)
            # Vertebrae are roughly 30mm tall, 30mm wide, 20mm deep
            half_size_voxels = np.array([15, 15, 10]) / np.array(spacing)
            
            bbox_3d = [
                int(max(0, centroid[0] - half_size_voxels[0])),  # x_min
                int(min(volume_shape[0], centroid[0] + half_size_voxels[0])),  # x_max
                int(max(0, centroid[1] - half_size_voxels[1])),  # y_min
                int(min(volume_shape[1], centroid[1] + half_size_voxels[1])),  # y_max
                int(max(0, centroid[2] - half_size_voxels[2])),  # z_min
                int(min(volume_shape[2], centroid[2] + half_size_voxels[2])),  # z_max
            ]
            
            # Average confidence
            avg_confidence = cluster_centers[:, 3].mean()
            
            vertebrae_3d[vertebra_name] = {
                'present': True,
                'center': [int(centroid[0]), int(centroid[1]), int(centroid[2])],
                'bbox': bbox_3d,
                'confidence': float(avg_confidence),
                'num_detections': int(cluster_mask.sum())
            }
        
        return vertebrae_3d
    
    def detect_case(self, ct_path, output_json):
        """
        Run full detection pipeline on one case.
        
        Args:
            ct_path: Path to CT scan
            output_json: Where to save detection results
        """
        case_id = Path(ct_path).stem.replace('_0000', '')
        
        # Step 1: Detect on 2D slices
        detections_2d, volume_shape, spacing = self.extract_and_detect_slices(ct_path)
        
        # Step 2: Cluster to 3D
        vertebrae_3d = self.cluster_to_3d(detections_2d, volume_shape, spacing)
        
        # Step 3: Save as JSON (for nnUNet)
        output_data = {
            'case_id': case_id,
            'vertebrae_detected': vertebrae_3d
        }
        
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return vertebrae_3d


def main():
    """Run detection on all 426 cases"""
    
    # ========== UPDATE THESE PATHS ==========
    MODEL_WEIGHTS = r"C:\Users\anoma\Downloads\cervical-yolo\runs\vertebra_detector\weights\best.pt"
    IMAGES_DIR = r"C:\Users\anoma\Downloads\spine-segmentation-data-cleaning\v3\imagesTr"
    OUTPUT_DIR = r"C:\Users\anoma\Downloads\cervical-yolo\outputs\detections"
    # ========================================
    
    print("="*70)
    print("RUNNING VERTEBRA DETECTION ON ALL CASES")
    print("="*70)
    
    # Check model exists
    if not Path(MODEL_WEIGHTS).exists():
        print(f"\n❌ ERROR: Model weights not found at {MODEL_WEIGHTS}")
        print("   Train the model first: python 02_train_yolo.py")
        return
    
    # Initialize detector
    detector = VertebraDetector(MODEL_WEIGHTS)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CT scans
    images_dir = Path(IMAGES_DIR)
    ct_files = sorted(images_dir.glob("*_0000.nii.gz"))
    
    print(f"\nFound {len(ct_files)} CT scans")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nStarting detection (this will take a while)...\n")
    
    # Process each case
    successful = 0
    failed = []
    
    for ct_path in tqdm(ct_files, desc="Detecting vertebrae"):
        case_id = ct_path.stem.replace('_0000', '')
        output_json = output_dir / f"{case_id}.json"
        
        try:
            vertebrae = detector.detect_case(ct_path, output_json)
            successful += 1
            
            # Quick summary
            present = [v for v, data in vertebrae.items() if data.get('present', False)]
            tqdm.write(f"  {case_id}: Detected {len(present)} vertebrae - {', '.join(present)}")
            
        except Exception as e:
            failed.append((case_id, str(e)))
            tqdm.write(f"  ❌ {case_id}: FAILED - {e}")
    
    # Summary
    print("\n" + "="*70)
    print("DETECTION COMPLETE")
    print("="*70)
    print(f"Successful: {successful}/{len(ct_files)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed cases:")
        for case_id, error in failed:
            print(f"  - {case_id}: {error}")
    
    print(f"\nDetection results saved to: {OUTPUT_DIR}")
    print(f"\nNEXT STEP: Upload {OUTPUT_DIR} to Google Drive")
    print(f"   Then integrate with nnUNet in Colab")


if __name__ == "__main__":
    main()