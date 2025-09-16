#!/bin/bash --login
#SBATCH --job-name ZS-Eval
#SBATCH --time=0:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gpus=1
#SBATCH --constraint=v100

#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96GB

#SBATCH -o logs/%x.%A.%a.out
#SBATCH -e logs/%x.%A.%a.err

#SBATCH --array=2,3,4,6,11

# Setup working environment:
cd #PATH_TO_THE_REPO
conda activate sm


FPS=3.0

# Launch experiment
DATASET=charades_sta
SMA_WINDOW=15
eval_batch_size=64
WATERSHED_THRESHOLD=1.0

ENCODER_WITH_DROP=openclip_vit_b_32
N=$SLURM_ARRAY_TASK_ID

# Model parameters
EPOCH=5
P_DROP=0.85
RESIDUAL_FEAT=True
DROP_MODE=motion
USE_MOTION_VECTORS=True

case $ENCODER_WITH_DROP in
    openclip_vit_b_32)
        ENCODER_NO_DROP=ViT-B-32  
        ;;
    openclip_vit_b_16)
        ENCODER_NO_DROP=ViT-B-16  
        ;;
    *)
        ENCODER_NO_DROP=ViT-L-14
        ;;
esac

case $ENCODER_NO_DROP in
    ViT-L-14)
        # Set values specific for ViT-L-14
        BSIZE=384  
        RESIDUAL_FEAT_DIM=768  
        ;;
    *)
        BSIZE=512
        RESIDUAL_FEAT_DIM=512
        ;;
esac

MODEL_PATH= #PATH_TO_MODEL_FOLDER

case $N in
    2)
        eval_batch_size=256
        ;;
    3 | 4)
        eval_batch_size=128
        ;;
    6 | 11)
        eval_batch_size=64
        ;;
    *)
        # Default value if N doesn't match any condition
        eval_batch_size=128
        ;;
esac


python -m aligner  \
    command=evaluate  \
    data=$DATASET   \
    output_dir=./out/$DATASET \
    data.num_workers=8 \
    data.eval_batch_size=$eval_batch_size \
    data.clip_length_in_frames=$N \
    data.frames_between_clips=$N \
    data.frame_rate=$FPS \
    data.use_motion_vectors=$USE_MOTION_VECTORS \
    encoder=$ENCODER_WITH_DROP   \
    encoder.model.force_patch_dropout=$P_DROP \
    encoder.model.force_inference_patch_dropout=True \
    encoder.model.inference_patch_dropout_mode=$DROP_MODE \
    encoder.use_residual_feat=$RESIDUAL_FEAT \
    encoder.model.use_residual_feat=$RESIDUAL_FEAT \
    encoder.model.residual_token_dim=$RESIDUAL_FEAT_DIM \
    encoder.model.pretrained=$MODEL_PATH/checkpoints/epoch_${EPOCH}.pt \
    encoder.model.distill_model=$ENCODER_NO_DROP \
    encoder.model.distill_pretrained=openai \
    encoder.model.force_quick_gelu=True \
    post_processing.smoothing_function='SMA' \
    post_processing.smoothing_window_size=$SMA_WINDOW \
    post_processing.watershed_threshold=$WATERSHED_THRESHOLD