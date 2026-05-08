#!/bin/bash
#SBATCH --job-name=attack2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/attack2_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llamagen

cd $HOME/LlamaGen_6S976
python /home/isawu888/LlamaGen_6S976/watermark/attack/vqvae_roundtrip/run_attack2.py
