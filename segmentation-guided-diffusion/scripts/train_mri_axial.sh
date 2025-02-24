#!/bin/bash
#SBATCH --job-name=sgd-amos_mri_axial
#SBATCH --mail-user=florian.hunecke@tum.de
#SBATCH --mail-type=ALL
#SBATCH --output=logs/amos_mri_axial.out
#SBATCH --error=logs/amos_mri_axial.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
# make sure to not get interrupted
#SBATCH --qos=master-queuesave
##SBATCH --partition=universe,asteroids

ml python/anaconda3

nvidia-smi

source deactivate
source activate py312

# strace -f python -v main.py \
python main.py \
    --mode train \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset amos_mri_axial \
    --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/images_axial \
    --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labels_axial \
    --model_type DDIM \
    --img_type MRI \
    --segmentation_guided \
    --segmentation_ingestion_mode concat \
    --segmentation_channel_mode single \
    --num_segmentation_classes 73 \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --num_epochs 100 \
    --transforms "['ToTensor', 'Resize', 'CenterCrop', 'Normalize']" \
    --resume \