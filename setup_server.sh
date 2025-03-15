#!/bin/bash
# Script to set up BIBSNet on a server without Docker
# Created: 2025-03-15

set -e  # Exit on error

# Create directories
BASEDIR="$HOME/bibsnet"
mkdir -p $BASEDIR/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
mkdir -p $BASEDIR/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed
mkdir -p $BASEDIR/data
cd $BASEDIR

# Set up Python environment
echo "Setting up Python environment..."
if command -v conda &> /dev/null; then
    conda create -y -n bibsnet python=3.8
    conda activate bibsnet
else
    python -m venv bibsnet_env
    source bibsnet_env/bin/activate
fi

# Install nnUNet and dependencies
echo "Installing nnUNet and dependencies..."
pip install torch torchvision nibabel SimpleITK scikit-image
# Install a specific version of nnUNet (1.7.1) that matches BIBSNet's requirement
pip install git+https://github.com/MIC-DKFZ/nnUNet.git@v1.7.1

# Set environment variables
echo "Setting environment variables..."
echo "export nnUNet_raw_data_base=\"$BASEDIR/nnUNet/nnUNet_raw_data_base\"" >> $HOME/.bashrc
echo "export nnUNet_preprocessed=\"$BASEDIR/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed\"" >> $HOME/.bashrc
echo "export RESULTS_FOLDER=\"$BASEDIR/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models\"" >> $HOME/.bashrc

# Source the updated .bashrc
source $HOME/.bashrc

# Download pre-trained models (you'll need to replace with actual download links)
echo "Downloading pre-trained BIBSNet models..."
MODEL_URLS=(
    "https://s3.msi.umn.edu/bibsnet-data/Task540_BIBSnet_Production_T1T2_model.tar.gz"
    "https://s3.msi.umn.edu/bibsnet-data/Task541_BIBSnet_Production_T1only_model.tar.gz"
    "https://s3.msi.umn.edu/bibsnet-data/Task542_BIBSnet_Production_T2only_model.tar.gz"
)

for URL in "${MODEL_URLS[@]}"; do
    FILENAME=$(basename $URL)
    wget $URL -O $FILENAME
    tar -xzf $FILENAME -C $BASEDIR/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet --strip-components 1
    rm $FILENAME
done

echo "BIBSNet setup complete!"
echo "To use BIBSNet, activate the environment: source bibsnet_env/bin/activate"
echo "Or if using conda: conda activate bibsnet"