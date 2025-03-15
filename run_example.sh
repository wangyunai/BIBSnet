#!/bin/bash
# Example script for running BIBSNet standalone implementation
# This script demonstrates various ways to run the BIBSNet model

# Activate your Python environment here
# source activate bibsnet
# OR
# source bibsnet-env/bin/activate

# Set nnUNet environment variables if not already done in your .bashrc/.profile
export nnUNet_raw_data_base="$HOME/nnUNet/nnUNet_raw_data"
export nnUNet_preprocessed="$HOME/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="$HOME/nnUNet/nnUNet_trained_models"

echo "======================================================="
echo "BIBSNet Standalone Implementation - Example Usage"
echo "======================================================="

# Check if input arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file> <output_dir> [subject_id] [session_id]"
    echo ""
    echo "Arguments:"
    echo "  input_file   : Path to T2w input file (.nii.gz)"
    echo "  output_dir   : Directory for results"
    echo "  subject_id   : Subject ID (default: sub-test)"
    echo "  session_id   : Session ID (default: ses-01)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/subject_T2w.nii.gz ./results sub-123456 ses-V01"
    exit 1
fi

# Inputs
INPUT_FILE=$1
OUTPUT_DIR=$2
SUBJECT_ID=${3:-sub-test}   # Default to sub-test if not provided
SESSION_ID=${4:-ses-01}     # Default to ses-01 if not provided

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Input file:  $INPUT_FILE"
echo "Output dir:  $OUTPUT_DIR"
echo "Subject ID:  $SUBJECT_ID"
echo "Session ID:  $SESSION_ID"
echo "======================================================="

# Example 1: Basic usage
echo "Running BIBSNet (basic usage)..."
python run_bibsnet_standalone.py \
  --input_file "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --subject "$SUBJECT_ID" \
  --session "$SESSION_ID" \
  --model 542 \
  --debug

# Example 2: With memory optimization (downsampling)
# Uncomment to use
# echo "Running BIBSNet with memory optimization..."
# python run_bibsnet_standalone.py \
#   --input_file "$INPUT_FILE" \
#   --output_dir "${OUTPUT_DIR}_downsampled" \
#   --subject "$SUBJECT_ID" \
#   --session "$SESSION_ID" \
#   --model 542 \
#   --debug \
#   --downsample \
#   --downsample_size 128

# Print output location
echo "======================================================="
echo "Processing complete!"
echo "Output segmentation file should be at:"
echo "$OUTPUT_DIR/derivatives/bibsnet/${SUBJECT_ID}_${SESSION_ID}_seg-bibs_dseg.nii.gz"
echo "======================================================="