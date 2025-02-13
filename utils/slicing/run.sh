#!/bin/bash
#SBATCH --job-name=slicing
#SBATCH --output=slicing.out
#SBATCH --error=slicing.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G

ml python/anaconda3

source deactivate
source activate py312

python slicing_script.py
