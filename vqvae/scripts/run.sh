#!/bin/bash
#SBATCH --job-name=vqvae
#SBATCH --output=logs/vqvae.out
#SBATCH --error=logs/vqvae.err
#SBATCH --time=48:00:00
#SBATCH --mem=10000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
# gpu 
#SBATCH --gres=gpu:1


ml python/anaconda3

source deactivate
source activate py312

python3 main.py --resume
