#!/bin/bash
#SBATCH --job-name=cgz_generate
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/generate_%j.out

conda activate llamagen
python watermark/generate_dataset.py --start $START --end $END
