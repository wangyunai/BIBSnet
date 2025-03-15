# BIBSNet Standalone Installation and Usage Guide

This guide explains how to set up and run BIBSNet without Docker for direct server installation.

## System Requirements

- Linux server (preferably with CUDA-capable GPU)
- Python 3.7+ with pip
- At least 32GB RAM (48GB+ recommended)
- 5GB+ free disk space for model weights

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/wangyunai/BIBSnet.git
cd BIBSnet
```

### 2. Set up Python environment

Using conda (recommended):
```bash
conda create -n bibsnet python=3.8
conda activate bibsnet
```

Or using venv:
```bash
python -m venv bibsnet-env
source bibsnet-env/bin/activate
```

### 3. Install nnUNet and dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install nnUNet and other dependencies
pip install nnunet==1.7.0
pip install nibabel SimpleITK scikit-image scipy numpy
```

### 4. Set up nnUNet environment variables

Add these to your `.bashrc` or environment setup:

```bash
export nnUNet_raw_data_base="$HOME/nnUNet/nnUNet_raw_data"
export nnUNet_preprocessed="$HOME/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="$HOME/nnUNet/nnUNet_trained_models"
```

Create necessary directories:
```bash
mkdir -p $nnUNet_raw_data_base $nnUNet_preprocessed $RESULTS_FOLDER
```

### 5. Download pre-trained BIBSNet models

```bash
# These links are for illustration - use the actual model links provided by the BIBSNet team
mkdir -p $RESULTS_FOLDER/nnUNet/3d_fullres

# Download and extract model 542 (T2w)
wget https://app.globus.org/file-manager?origin_id=a386b552-6086-11ea-9688-0e56c063f437&origin_path=/nnUNet/3d_fullres/Task542_dHCPT2wSS/nnUNetTrainerV2__nnUNetPlansv2.1/ -O T2w_model.zip
unzip T2w_model.zip -d $RESULTS_FOLDER/nnUNet/3d_fullres/

# Download other models if needed
# wget https://path/to/model541.zip -O T1w_model.zip
# unzip T1w_model.zip -d $RESULTS_FOLDER/nnUNet/3d_fullres/
```

## Running BIBSNet Standalone

The `run_bibsnet_standalone.py` script provides an optimized way to run BIBSNet directly on your server.

### Basic Usage

```bash
python run_bibsnet_standalone.py \
  --input_file /path/to/input_T2w.nii.gz \
  --output_dir /path/to/output \
  --subject sub-123456 \
  --session ses-V01 \
  --model 542
```

### Memory Optimization Options

If you're experiencing memory issues, try these options:

```bash
python run_bibsnet_standalone.py \
  --input_file /path/to/input_T2w.nii.gz \
  --output_dir /path/to/output \
  --subject sub-123456 \
  --session ses-V01 \
  --model 542 \
  --downsample \
  --downsample_size 128
```

The `--downsample` flag will temporarily reduce the input image size to save memory during processing, then upsample the results back to the original resolution.

### Using Existing Preprocessed Data

If you've already set up a BIDS-compliant dataset:

```bash
python run_bibsnet_standalone.py \
  --input_dir /path/to/bids_dataset \
  --output_dir /path/to/output \
  --subject sub-123456 \
  --session ses-V01 \
  --model 542
```

### Debug Mode

For detailed logging:

```bash
python run_bibsnet_standalone.py \
  --input_file /path/to/input_T2w.nii.gz \
  --output_dir /path/to/output \
  --subject sub-123456 \
  --session ses-V01 \
  --model 542 \
  --debug
```

## Output

The script produces a segmentation file at:
```
/path/to/output/derivatives/bibsnet/sub-123456_ses-V01_seg-bibs_dseg.nii.gz
```

## Troubleshooting

### Memory Issues

1. Try the `--downsample` option to reduce memory requirements
2. Increase server swap space
3. Ensure no other memory-intensive processes are running

### Missing Dependencies

If you get errors about missing modules, install them with:
```bash
pip install [module_name]
```

### Model Not Found

Ensure the environment variables are correctly set and the model files are in the expected location.

## Differences from Docker Version

This standalone version:
- Runs directly on your system without Docker overhead
- Includes memory optimization options not available in the Docker version
- Allows easier customization of parameters and preprocessing
- May require more manual setup of dependencies

## License

This code is provided under the same license as the original BIBSNet repository.