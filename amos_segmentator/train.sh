#!/bin/bash
#SBATCH --job-name=amos_segmentator
#SBATCH --mail-user=florian.hunecke@tum.de
#SBATCH --mail-type=ALL
#SBATCH --output=logs/amos_segmentator.out
#SBATCH --error=logs/amos_segmentator.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=master-queuesave

# Load the conda environment
ml python/anaconda3

source deactivate
source activate py312

# Run the main.py script
python main.py \
    --exp_name test \
    --log_dir logs \
    --batch_size 32 \
    --img_size 256 \
    --transforms "['ToTensor', 'RandomRotation', 'RandomResizedCrop', 'Normalize']" \
    --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/images_all_axis \
    --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labels_all_axis \
    --include_bg True \
    --model_type unet \
    --lr 0.01 \
    --weight_decay 0.001 \
    --tv_weight 0.01 \
    --focal_weight 10 \
    --num_workers 32 \
    --max_epochs 100 \
    --resume \

    # --transforms "['ToTensor', 'Resize', 'CenterCrop', 'Normalize']" \
    # --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/test_data/images \
    # --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/test_data/labels \
    # --ce_weight "/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/amos_segmentator/utils/label_weights.npy" \