#!/bin/bash --login
#SBATCH --job-name ResidualViT
#SBATCH --time=23:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint=v100

#SBATCH --cpus-per-gpu=8
#SBATCH --mem=384GB

#SBATCH -o logs/%x.%A.%a.out
#SBATCH -e logs/%x.%A.%a.err

unset KUBERNETES_PORT

# Setup working environment:
cd #PATH_TO_REPOSITORY
source .venv/bin/activate 
cd ./src    

#------------------------------------ DATA ------------------------------------
NUM_FRAMES=4
TARGET_FPS=1.0
ROOT_SHARDS= #PATH_TO_WEBVID

DATA_PARAMS="--train-data "${ROOT_SHARDS}/webvid-2M-{000000..001596}.tar" \
            --train-num-samples 2483998 \
            --val-data "${ROOT_SHARDS}/webvid-2M-001597.tar" \
            --val-num-samples 1414 \
            --dataset-type webdataset \
            --video \
            --num-frames $NUM_FRAMES \
            --video-target-fps $TARGET_FPS"

#----------------------------------- MODEL ------------------------------------
STUDENT=ViT-B-32
TEACHER=ViT-B-32

RESIDUAL_TOKEN_DIM=512  # 768 if L14 is teacher, otherwise 512
PRETRAINED_STUDENT=openai
PRETRAINED_TEACHER=openai
NUM_RES_LAYERS=1

STUDENT_IMAGE_SIZE=96
TEACHER_IMAGE_SIZE=224

MODEL_PARAMS="--model $STUDENT \
            --pretrained $PRETRAINED_STUDENT \
            --distill-model $TEACHER \
            --distill-pretrained $PRETRAINED_TEACHER \
            --distill-no-contrastive-loss \
            --distill-vision-tower-only \
            --distill-anchor-target \
            --use-residual-token True \
            --residual-token-dim $RESIDUAL_TOKEN_DIM \
            --residual-token-projection-num-layers $NUM_RES_LAYERS \
            --lock-image \
            --force-image-size $STUDENT_IMAGE_SIZE \
            --force-dist-image-size $TEACHER_IMAGE_SIZE"

#---------------------------------- TRAINING ----------------------------------
LOSS=ce
BSIZE=512
EPOCHS=5
LR=0.0005
WD=0.0

TRAINING_PARAMS="--precision amp \
                --workers 6 \
                --epochs $EPOCHS \
                --loss-version $LOSS \
                --batch-size $BSIZE \
                --wd $WD  \
                --warmup 0 \
                --lr-scheduler const \
                --lr $LR "
                
#---------------------------------- LOGGING -----------------------------------
FOLDER=lower_res
EXP=${FOLDER}/${STUDENT}__res_${STUDENT_IMAGE_SIZE}
LOGGING_PARAMS="--logs-path ../log_models/ \
                --report-to wandb \
                --name $EXP "     

#----------------------------- Launch experiemnt ------------------------------
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    -m training.main \
    $DATA_PARAMS \
    $MODEL_PARAMS \
    $TRAINING_PARAMS \
    $LOGGING_PARAMS