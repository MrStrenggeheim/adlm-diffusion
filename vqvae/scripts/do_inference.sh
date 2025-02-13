#!/bin/bash
#SBATCH --job-name=vqvae-inf
#SBATCH --output=logs/vqvae-inf.out
#SBATCH --error=logs/vqvae-inf.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

ml python/anaconda3

source deactivate
source activate py312

python3 infer.py