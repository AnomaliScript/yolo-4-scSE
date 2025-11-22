"""
Step 1: Extract 2D Slices from 3D CT Scans
===========================================

Extracts sagittal and coronal slices from CT volumes for YOLO training.
Based on SpineCLUE methodology.

Author: Brandon's Cervical Spine Project
Date: 2024-11-19
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import json


class SliceExtractor:
    """Extract 2D slices from 3D CT volumes for YOLO training"""
    
    def __init__(self, output_dir, num_slices_per_orientation=200):
        """
        Args:
            output_dir: Where to save extracted slices
            num_slices_per_orientation: Number of slices to extract (sagittal and coronal)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_slices = num_slices_per_orientation
        
        # Track which slices we extracted for later annotation
        self.slice_metadata = []
    
    def normalize_ct_for_display(self, ct_volume):
        """
        Normalize CT HU values for display.
        
        CT scans have Hounsfield Units (HU):
        - Air: -1000
        - Water: 0
        - Bone: +1000 to +3000
        
        We window to [-1000, 1000] HU for bone/soft tissue visualization.
        """
        # Clip to bone window
        windowed = np.clip(ct_volume, -1000, 1000)
        
        # Normalize to 0-255 for image display
        normalized = ((windowed + 1000) / 2000 * 255).astype(np.uint8)
        
        return normalized
    
    def extract_slices_from_case(self, ct_path, case_id):
        """
        Extract sagittal and coronal slices from one CT scan.
        
        Args:
            ct_path: Path to NIfTI CT file
            case_id: Case identifier (e.g., 'RSNA_010')
        
        Returns:
            list of extracted slice paths
        """
        print(f"\nProcessing {case_id}...")
        
        # Load CT
        try:
            ct_nii = nib.load(str(ct_path))
            ct_volume = ct_nii.get_fdata()
            spacing = ct_nii.header.get_zooms()
        except Exception as e:
            print(f"  ❌ Failed to load {ct_path}: {e}")
            return []
        
        print(f"  Volume shape: {ct_volume.shape}")
        print(f"  Spacing: {spacing}")
        
        # Normalize for display
        ct_normalized = self.normalize_ct_for_display(ct_volume)
        
        extracted_files = []
        
        # ===== SAGITTAL SLICES (YZ plane) =====
        # Looking from the side - can see vertebrae stacked vertically
        print(f"  Extracting {self.num_slices} sagittal slices...")
        X_dim = ct_volume.shape[0]
        
        # Sample evenly across X dimension
        sagittal_indices = np.linspace(
            int(X_dim * 0.2),  # Skip first 20% (far from spine)
            int(X_dim * 0.8),  # Skip last 20%
            self.num_slices, 
            dtype=int
        )
        
        for idx in sagittal_indices:
            slice_img = ct_normalized[idx, :, :]
            
            # Resize to YOLO standard size
            slice_img = cv2.resize(slice_img, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # Save
            filename = f"{case_id}_sagittal_{idx:03d}.jpg"
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), slice_img)
            
            # Track metadata
            self.slice_metadata.append({
                'case_id': case_id,
                'filename': filename,
                'orientation': 'sagittal',
                'slice_index': int(idx),
                'dimension': 'X',
                'spacing_mm': float(spacing[0])
            })
            
            extracted_files.append(output_path)
        
        # ===== CORONAL SLICES (XZ plane) =====
        # Looking from front/back - can see vertebrae from anterior/posterior view
        print(f"  Extracting {self.num_slices} coronal slices...")
        Y_dim = ct_volume.shape[1]
        
        coronal_indices = np.linspace(
            int(Y_dim * 0.2),
            int(Y_dim * 0.8),
            self.num_slices,
            dtype=int
        )
        
        for idx in coronal_indices:
            slice_img = ct_normalized[:, idx, :]
            slice_img = cv2.resize(slice_img, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            filename = f"{case_id}_coronal_{idx:03d}.jpg"
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), slice_img)
            
            self.slice_metadata.append({
                'case_id': case_id,
                'filename': filename,
                'orientation': 'coronal',
                'slice_index': int(idx),
                'dimension': 'Y',
                'spacing_mm': float(spacing[1])
            })
            
            extracted_files.append(output_path)
        
        print(f"  ✓ Extracted {len(extracted_files)} slices")
        return extracted_files
    
    def save_metadata(self, output_file):
        """Save metadata about extracted slices"""
        with open(output_file, 'w') as f:
            json.dump(self.slice_metadata, f, indent=2)
        print(f"\n✓ Saved metadata to {output_file}")


def main():
    """
    Main extraction workflow.
    
    YOU NEED TO UPDATE THESE PATHS!
    """
    
    # ========== UPDATE THESE PATHS ==========
    IMAGES_DIR = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\v3\\imagesTr"
    OUTPUT_DIR = r"C:\\Users\\anoma\\Downloads\\yolo-4-scse\\data\\images\\train"
    
    # Select cases to extract slices from
    # SpineCLUE used 20 cases - select full-spine cases (C1-C7)
    # Based on your dataset inspection: 264 cases have full C1-C7
    SELECTED_CASES = [
        "RSNA_010", "RSNA_015", "RSNA_020", "RSNA_025", "RSNA_030",
        "RSNA_035", "RSNA_040", "RSNA_045", "RSNA_050", "RSNA_055",
        "CTS1K_012", "CTS1K_015", "CTS1K_018", "CTS1K_022", "CTS1K_025",
        "VerSe_001", "VerSe_005", "VerSe_010", "VerSe_015", "VerSe_020"
    ]
    # ========================================
    
    print("="*70)
    print("2D SLICE EXTRACTION FOR YOLO TRAINING")
    print("="*70)
    print(f"Source directory: {IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cases to process: {len(SELECTED_CASES)}")
    print(f"Expected output: ~{len(SELECTED_CASES) * 400} images")
    print("="*70)
    
    # Initialize extractor
    extractor = SliceExtractor(
        output_dir=OUTPUT_DIR,
        num_slices_per_orientation=200
    )
    
    # Process each case
    images_dir = Path(IMAGES_DIR)
    total_slices = 0
    
    for case_id in tqdm(SELECTED_CASES, desc="Processing cases"):
        # Find the CT file
        # Common naming: RSNA_010_0000.nii.gz
        ct_files = list(images_dir.glob(f"{case_id}*_0000.nii.gz"))
        
        if not ct_files:
            print(f"⚠️  Warning: No CT file found for {case_id}")
            continue
        
        ct_path = ct_files[0]
        slices = extractor.extract_slices_from_case(ct_path, case_id)
        total_slices += len(slices)
    
    # Save metadata
    metadata_path = Path(OUTPUT_DIR).parent / "slice_metadata.json"
    extractor.save_metadata(metadata_path)
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total slices extracted: {total_slices}")
    print(f"Location: {OUTPUT_DIR}")
    print(f"Metadata: {metadata_path}")
    print("\n📋 NEXT STEP: Annotate these images with LabelImg")
    print("   Download: https://github.com/HumanSignal/labelImg")
    print("="*70)


if __name__ == "__main__":
    main()