ml python/anaconda3

source deactivate
source activate py312

python inference.py \
    --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/images_axial \
    --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labels_axial \
    --output_dir /vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_mri_axial/single_nopad \
    --ckpt_path /vol/miltank/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/output/ddim-amos_mri_all_axis-256-1-concat-segguided/ep65/unet \
    --num_eval_batches 64 \
    --num_preds_per_seg 1 \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset amos_mri_axial \
    --model_type DDIM \
    --img_type MRI \
    --segmentation_guided \
    --segmentation_ingestion_mode concat \
    --segmentation_channel_mode single \
    --num_segmentation_classes 73 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 100 \
    --transforms "['ToTensor', 'Resize', 'CenterCrop', 'Normalize']" \
    --resume \
    # --ckpt_path /vol/miltank/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/output/ddim-amos_mri_axial-256-1-concat-segguided/ep82/unet \