ml python/anaconda3

source deactivate
source activate py312

python eval.py \
    --dir /vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_mri_all_axis/single_nopad/ \
    --img_size 256 \
    --num_classes 73 \
