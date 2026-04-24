#!/bin/bash
#SBATCH --job-name=cgz_generate
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/generate_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llamagen

cd /orcd/home/002/isawu888/LlamaGen_6S976
python watermark/generate_dataset.py --start $START --end $END