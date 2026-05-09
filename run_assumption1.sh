#!/bin/bash
#SBATCH --job-name=assumption1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/assumption1_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llamagen

cd $HOME/LlamaGen_6S976

PYTHONPATH=$HOME/LlamaGen_6S976 python watermark/reporting/assumption1_verification.py \
    --root aggregated_samples \
    --n-images 200
