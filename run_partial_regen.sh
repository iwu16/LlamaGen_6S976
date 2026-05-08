#!/bin/bash
#SBATCH --job-name=partial_regen
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/partial_regen_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llamagen

cd $HOME/LlamaGen_6S976

python -m watermark.attack.token_regeneration.partial_regen \
    --root aggregated_samples \
    --output-dir watermark/attack/token_regeneration/partial_regen_outputs
