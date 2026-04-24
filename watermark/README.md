# Watermarking LlamaGen — Setup Guide

## 1. Clone the repo
git clone https://github.com/iwu16/LlamaGen_6S976.git
cd LlamaGen_6S976

## 2. Create conda environment
conda create -n llamagen python=3.10 -y
conda activate llamagen

## 3. Install PyTorch (pick the right CUDA version)
# CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

## 4. Install dependencies
pip install -r requirements.txt
pip install huggingface_hub lpips clean-fid

## 5. Download pretrained weights
python watermark/download_weights.py

## 6. Verify everything works
python watermark/verify.py
