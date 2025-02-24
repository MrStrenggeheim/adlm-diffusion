# gen
python3 inference.py \
    --input_dir /vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_mri_all_axis/single_nopad/img_gen \
    --output_dir /vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_mri_all_axis/single_nopad/seg_pred_gen \
    --ckpt_path /vol/miltank/projects/practical_WS2425/diffusion/code/amos_segmentator/logs/final/version_6/checkpoints/amos_segmentator_checkpoint-epoch=04-val_loss=0.17.ckpt \
    --img_size 256 \
# orig
python3 inference.py \
    --input_dir /vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_mri_all_axis/single_nopad/img \
    --output_dir /vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_mri_all_axis/single_nopad/seg_pred \
    --ckpt_path /vol/miltank/projects/practical_WS2425/diffusion/code/amos_segmentator/logs/final/version_6/checkpoints/amos_segmentator_checkpoint-epoch=04-val_loss=0.17.ckpt \
    --img_size 256 \