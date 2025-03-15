#!/usr/bin/env python3
"""
run_bibsnet_standalone.py - Standalone script to run the BIBSNet deep learning model 
for infant brain segmentation with optimized memory usage.

This script can process either raw T2w images or preprocessed data and runs 
the nnUNet deep learning model with BIBSNet pre-trained weights to generate
brain segmentations.

Modified for memory optimization: 2025-03-15
"""

import sys
import argparse
import subprocess
import logging
import time
import os
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparse():
    """Set up argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run BIBSNet deep learning model with memory optimization'
    )
    parser.add_argument(
        '--input_file',
        help='Path to input T2w file'
    )
    parser.add_argument(
        '--input_dir',
        help='Directory containing preprocessed data'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory for output results'
    )
    parser.add_argument(
        '--subject',
        required=True,
        help='Subject ID (with or without sub- prefix)'
    )
    parser.add_argument(
        '--session',
        required=True,
        help='Session ID (with or without ses- prefix)'
    )
    parser.add_argument(
        '--model',
        type=int,
        choices=[540, 541, 542],
        default=542,
        help='BIBSNet model ID (542 for T2w, 541 for T1w, 540 for both)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--downsample',
        action='store_true',
        help='Downsample input image to reduce memory usage'
    )
    parser.add_argument(
        '--downsample_size',
        type=int,
        default=160,
        help='Target size for downsampling (default: 160)'
    )
    return parser


def clean_id(id_str, prefix):
    """Remove prefix from ID if present."""
    return id_str[len(prefix):] if id_str.startswith(prefix) else id_str


def downsample_image(input_path, target_size=160):
    """
    Downsample an image to reduce memory requirements.
    
    Args:
        input_path: Path to the input image
        target_size: Target size for the largest dimension
        
    Returns:
        Tuple of (downsampled_path, original_path)
    """
    logger.info(f"Downsampling image {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Calculate zoom factors to get target size
    max_dim = max(data.shape)
    if max_dim <= target_size:
        logger.info(f"Image is already small enough (shape: {data.shape})")
        return input_path, None
    
    zoom_factors = np.array([target_size/data.shape[0], 
                              target_size/data.shape[1], 
                              target_size/data.shape[2]])
    
    logger.info(f"Original size: {data.shape}")
    downsampled = zoom(data, zoom_factors, order=1)
    logger.info(f"Downsampled size: {downsampled.shape}")
    
    # Create new image with adjusted affine
    new_img = nib.Nifti1Image(
        downsampled, 
        img.affine * np.diag(list(1.0/zoom_factors) + [1])
    )
    
    # Save downsampled version
    backup_path = input_path + '.original'
    os.rename(input_path, backup_path)
    nib.save(new_img, input_path)
    logger.info(f"Saved downsampled version to {input_path}")
    
    return input_path, backup_path


def upsample_segmentation(seg_path, original_img_path, output_path):
    """
    Upsample a segmentation image to match the original resolution.
    
    Args:
        seg_path: Path to the segmentation image
        original_img_path: Path to the original image
        output_path: Path to save the upsampled segmentation
    """
    logger.info(f"Upsampling segmentation {seg_path} to match original resolution")
    
    # Load original and segmentation
    orig_img = nib.load(original_img_path)
    seg_img = nib.load(seg_path)
    
    # Get shapes
    orig_shape = orig_img.get_fdata().shape
    seg_data = seg_img.get_fdata()
    
    # Calculate zoom factors
    zoom_factors = [float(o)/float(s) for o, s in zip(orig_shape, seg_data.shape)]
    
    logger.info(f"Original shape: {orig_shape}, segmentation shape: {seg_data.shape}")
    logger.info(f"Zoom factors: {zoom_factors}")
    
    # Use nearest neighbor interpolation for label maps
    upsampled = zoom(seg_data, zoom_factors, order=0)
    
    # Save upsampled segmentation
    upsampled_img = nib.Nifti1Image(upsampled, orig_img.affine)
    nib.save(upsampled_img, output_path)
    logger.info(f"Saved upsampled segmentation to {output_path}")


def run_nnunet(input_dir, output_dir, model_id, num_threads=1):
    """
    Run nnUNet with memory optimizations.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output segmentations
        model_id: BIBSNet model ID
        num_threads: Number of threads for preprocessing
    
    Returns:
        Exit code from nnUNet
    """
    # Prepare nnUNet command with memory optimization flags
    cmd = [
        "nnUNet_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-t", str(model_id),
        "-m", "3d_fullres",
        "--disable_tta",           # Disable test time augmentation to save memory
        "--mode", "fast",          # Use fast mode with reduced memory usage
        "--step_size", "0.75",     # Increase step size to reduce memory usage
        "--disable_mixed_precision",  # Can help with certain hardware
        "--num_threads_preprocessing", str(num_threads),  # Limit preprocessing threads
        "--all_in_gpu", "False"    # Process on CPU when GPU memory is full
    ]
    
    # Run nnUNet
    logger.info(f"Running nnUNet command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Show output in real-time
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
            
        # Get return code
        return_code = process.wait()
        if return_code == 0:
            logger.info("nnUNet completed successfully")
        else:
            logger.error(f"nnUNet failed with return code {return_code}")
        
        return return_code
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nnUNet: {e}")
        return e.returncode


def run_bibsnet(args):
    """Run the BIBSNet deep learning model."""
    # Clean subject and session IDs
    subject_clean = clean_id(args.subject, 'sub-')
    session_clean = clean_id(args.session, 'ses-')
    
    # Prepare paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up directories
    if args.input_file:
        # We're using a raw input file
        bids_dir = output_dir / "bids_input"
        anat_dir = bids_dir / f"sub-{subject_clean}" / f"ses-{session_clean}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input file to BIDS directory with proper naming
        t2w_file = anat_dir / f"sub-{subject_clean}_ses-{session_clean}_T2w.nii.gz"
        if not t2w_file.exists():
            logger.info(f"Copying input file to BIDS directory: {t2w_file}")
            shutil.copy2(args.input_file, t2w_file)
    else:
        # We're using an existing BIDS directory
        bids_dir = Path(args.input_dir) if args.input_dir else Path(".")
        t2w_file = (bids_dir / f"sub-{subject_clean}" / f"ses-{session_clean}" / 
                   "anat" / f"sub-{subject_clean}_ses-{session_clean}_T2w.nii.gz")
    
    # Verify input T2w file exists
    if not t2w_file.exists():
        # Try with run-1 in the filename
        t2w_file = (bids_dir / f"sub-{subject_clean}" / f"ses-{session_clean}" / 
                   "anat" / f"sub-{subject_clean}_ses-{session_clean}_run-1_T2w.nii.gz")
        if not t2w_file.exists():
            logger.error(f"T2w file not found: {t2w_file}")
            sys.exit(1)
    
    # Create derivatives and work directories
    derivatives_dir = output_dir / "derivatives" / "bibsnet"
    work_dir = output_dir / "work"
    for dir_path in [derivatives_dir, work_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create bibsnet input and output directories
    bibsnet_input_dir = (
        work_dir / "bibsnet" / f"sub-{subject_clean}" / 
        f"ses-{session_clean}" / "input"
    )
    bibsnet_output_dir = (
        work_dir / "bibsnet" / f"sub-{subject_clean}" / 
        f"ses-{session_clean}" / "output"
    )
    bibsnet_input_dir.mkdir(parents=True, exist_ok=True)
    bibsnet_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy T2w file to nnUNet input directory with correct naming
    bibsnet_input = (
        bibsnet_input_dir / 
        f"sub-{subject_clean}_ses-{session_clean}_optimal_resized_0000.nii.gz"
    )
    
    # Preprocess the input file - copy it to the proper location
    logger.info(f"Preprocessing input file: {t2w_file}")
    shutil.copy2(t2w_file, bibsnet_input)
    
    # Optionally downsample the input image
    original_path = None
    if args.downsample:
        _, original_path = downsample_image(
            str(bibsnet_input), 
            target_size=args.downsample_size
        )
    
    # Run nnUNet for segmentation
    logger.info("Running nnUNet for segmentation")
    exit_code = run_nnunet(
        str(bibsnet_input_dir),
        str(bibsnet_output_dir),
        args.model,
        num_threads=1  # Use single thread for preprocessing to save memory
    )
    
    if exit_code != 0:
        logger.error(f"nnUNet failed with exit code {exit_code}")
        sys.exit(exit_code)
    
    # Check if segmentation was created
    segmentation_file = (
        bibsnet_output_dir / 
        f"sub-{subject_clean}_ses-{session_clean}_optimal_resized.nii.gz"
    )
    
    if not segmentation_file.exists():
        logger.error(f"Segmentation file not created: {segmentation_file}")
        sys.exit(1)
    
    # Create final output file
    final_output = derivatives_dir / f"sub-{subject_clean}_ses-{session_clean}_seg-bibs_dseg.nii.gz"
    
    # If we downsampled, we need to upsample the segmentation back to original resolution
    if original_path:
        logger.info("Upsampling segmentation to original resolution")
        upsample_segmentation(
            str(segmentation_file),
            original_path,
            str(final_output)
        )
        # Restore original file
        os.rename(original_path, str(bibsnet_input))
    else:
        # Copy segmentation to derivatives directory
        logger.info(f"Copying segmentation to output: {final_output}")
        shutil.copy2(segmentation_file, final_output)
    
    logger.info(f"BIBSNet processing completed successfully. Output at: {final_output}")
    return 0


def main():
    """Main function."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Run BIBSNet
    exit_code = run_bibsnet(args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()