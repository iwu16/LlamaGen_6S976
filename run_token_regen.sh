#!/bin/bash
#SBATCH --job-name=cgz_attack1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/attack1_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llamagen
cd $HOME/LlamaGen_6S976

python -m watermark.attack.token_regeneration --start $START --end $END