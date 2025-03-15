#!/bin/bash
# BIBSNet Standalone Installation Script
# This script sets up a Python environment and installs all required dependencies
# for running BIBSNet directly on a server without Docker

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/nnUNet"
PYTHON_VERSION="3.8"
ENV_NAME="bibsnet"
USE_CONDA=true  # Set to false to use venv instead

# Model URLs - replace with actual URLs if needed
MODEL_542_URL="https://app.globus.org/file-manager?origin_id=a386b552-6086-11ea-9688-0e56c063f437&origin_path=/nnUNet/3d_fullres/Task542_dHCPT2wSS/nnUNetTrainerV2__nnUNetPlansv2.1/"
MODEL_541_URL="https://app.globus.org/file-manager?origin_id=a386b552-6086-11ea-9688-0e56c063f437&origin_path=/nnUNet/3d_fullres/Task541_dHCPT1wSS/nnUNetTrainerV2__nnUNetPlansv2.1/"
MODEL_540_URL="https://app.globus.org/file-manager?origin_id=a386b552-6086-11ea-9688-0e56c063f437&origin_path=/nnUNet/3d_fullres/Task540_dHCPT1T2wSS/nnUNetTrainerV2__nnUNetPlansv2.1/"

# CUDA version - adjust as needed
CUDA_VERSION="cu113"  # For PyTorch 1.10.1

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is required but not installed. Please install Python 3.7+ and try again."
        exit 1
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        echo "Error: pip3 is required but not installed. Please install pip and try again."
        exit 1
    fi
    
    # Check for conda if using conda
    if [ "$USE_CONDA" = true ] && ! command -v conda &> /dev/null; then
        echo "Error: conda is required but not installed. Please install conda or set USE_CONDA=false to use venv instead."
        exit 1
    fi
    
    # Check for wget
    if ! command -v wget &> /dev/null; then
        echo "Error: wget is required but not installed. Please install wget and try again."
        exit 1
    fi
    
    echo "All prerequisites satisfied."
}

# Create Python environment
create_environment() {
    echo "Setting up Python environment..."
    
    if [ "$USE_CONDA" = true ]; then
        echo "Creating conda environment: $ENV_NAME"
        conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
        
        # Activate conda environment
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
    else
        echo "Creating virtual environment: $ENV_NAME"
        python3 -m venv "$ENV_NAME-env"
        
        # Activate virtual environment
        source "$ENV_NAME-env/bin/activate"
    fi
    
    echo "Python environment created and activated."
}

# Install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    
    # Install PyTorch with appropriate CUDA version
    echo "Installing PyTorch (CUDA version: $CUDA_VERSION)..."
    pip install torch==1.10.1+${CUDA_VERSION} torchvision==0.11.2+${CUDA_VERSION} -f https://download.pytorch.org/whl/${CUDA_VERSION}/torch_stable.html
    
    # Install nnUNet
    echo "Installing nnUNet..."
    pip install nnunet==1.7.0
    
    # Install other dependencies
    echo "Installing other dependencies..."
    pip install nibabel SimpleITK scikit-image scipy numpy tqdm
    
    echo "All dependencies installed successfully."
}

# Set up nnUNet directories and environment variables
setup_nnunet() {
    echo "Setting up nnUNet directories and environment variables..."
    
    # Create directories
    mkdir -p "$INSTALL_DIR/nnUNet_raw_data"
    mkdir -p "$INSTALL_DIR/nnUNet_preprocessed"
    mkdir -p "$INSTALL_DIR/nnUNet_trained_models"
    
    # Set environment variables
    export nnUNet_raw_data_base="$INSTALL_DIR/nnUNet_raw_data"
    export nnUNet_preprocessed="$INSTALL_DIR/nnUNet_preprocessed"
    export RESULTS_FOLDER="$INSTALL_DIR/nnUNet_trained_models"
    
    # Add to .bashrc if not already there
    if ! grep -q "nnUNet_raw_data_base" "$HOME/.bashrc"; then
        echo "# nnUNet environment variables" >> "$HOME/.bashrc"
        echo "export nnUNet_raw_data_base=\"$INSTALL_DIR/nnUNet_raw_data\"" >> "$HOME/.bashrc"
        echo "export nnUNet_preprocessed=\"$INSTALL_DIR/nnUNet_preprocessed\"" >> "$HOME/.bashrc"
        echo "export RESULTS_FOLDER=\"$INSTALL_DIR/nnUNet_trained_models\"" >> "$HOME/.bashrc"
    fi
    
    echo "nnUNet setup complete."
    echo "Environment variables have been added to your .bashrc file."
}

# Download pre-trained models
download_models() {
    echo "Downloading pre-trained BIBSNet models..."
    echo "NOTE: The models are large and may take some time to download."
    
    mkdir -p "$INSTALL_DIR/nnUNet_trained_models/nnUNet/3d_fullres"
    
    # Display instructions for downloading models manually
    echo "Please manually download the BIBSNet models from:"
    echo "T2w model (542): $MODEL_542_URL"
    echo "T1w model (541): $MODEL_541_URL"
    echo "T1w+T2w model (540): $MODEL_540_URL"
    echo ""
    echo "After downloading, extract them to: $INSTALL_DIR/nnUNet_trained_models/nnUNet/3d_fullres/"
    echo ""
    echo "For automated downloads, this script would need to be updated with direct download links."
}

# Main installation function
install_bibsnet() {
    echo "======================================================="
    echo "BIBSNet Standalone Installation"
    echo "======================================================="
    
    check_prerequisites
    create_environment
    install_dependencies
    setup_nnunet
    download_models
    
    echo "======================================================="
    echo "BIBSNet installation completed successfully!"
    echo "To use BIBSNet, activate the environment with:"
    if [ "$USE_CONDA" = true ]; then
        echo "  conda activate $ENV_NAME"
    else
        echo "  source $ENV_NAME-env/bin/activate"
    fi
    echo ""
    echo "Then run the example script:"
    echo "  bash $SCRIPT_DIR/run_example.sh /path/to/input.nii.gz /path/to/output [subject_id] [session_id]"
    echo "======================================================="
}

# Run the installation
install_bibsnet