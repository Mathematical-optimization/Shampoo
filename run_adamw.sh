#!/bin/bash

# --- 스크립트 설정 ---
set -e
set -x

# --- 사용자 설정 변수 ---
# 이 부분의 값들을 필요에 맞게 수정하여 사용하세요.
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 시스템에 맞게 GPU 설정
N_GPUS=4 # 사용할 GPU 개수

# 데이터셋을 캐시할 경로
DATA_PATH="$HOME/.cache/huggingface/datasets"

# [변경] AdamW 실험 결과를 저장할 기본 경로
OUTPUT_DIR="./training_output_adamw"

# [변경] 실행할 Python 스크립트 파일 이름
SCRIPT_NAME="vit_adamw.py"

# 학습 기본 하이퍼파라미터
EPOCHS=90
BATCH_SIZE_PER_GPU=256
WORKERS=4

# [변경] AdamW 옵티마이저 및 스케줄러 하이퍼파라미터
BASE_LR=0.001
WARMUP_STEPS=7500
WEIGHT_DECAY=0.001
BETA1=0.95
BETA2=0.999 # AdamW를 위한 Beta2 추가

# 데이터 증강 하이퍼파라미터
MIXUP=0.2
LABEL_SMOOTHING=0.1
RESUME_FROM="" # 체크포인트에서 재개 (필요시 경로 설정)

# --- 실행 설정 ---
# [변경] RUN_NAME에 AdamW와 관련 하이퍼파라미터 포함
RUN_NAME="vit_adamw_LR${BASE_LR}_WD${WEIGHT_DECAY}_B1${BETA1}_B2${BETA2}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$OUTPUT_DIR/$RUN_NAME/logs"
SAVE_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"
mkdir -p $LOG_PATH
mkdir -p $SAVE_DIR

RESUME_OPTION=""
if [ ! -z "$RESUME_FROM" ]; then
    RESUME_OPTION="--resume $RESUME_FROM"
    echo "Resuming training from: $RESUME_FROM"
fi

# --- 분산 학습 실행 ---
echo "========================================================"
echo "Vision Transformer on ImageNet-1k Training (Algoperf Spec)"
echo "Optimizer: AdamW" # [변경] 옵티마이저 이름 변경
echo "GPUs: $N_GPUS"
echo "Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "LR: $BASE_LR, WD: $WEIGHT_DECAY, Beta1: $BETA1, Beta2: $BETA2" # [변경] Beta2 정보 추가
echo "Augmentations: RandAugment(m15-n2), Mixup($MIXUP), LS($LABEL_SMOOTHING)"
echo "Log Path: $LOG_PATH"
echo "========================================================"

# [변경] torchrun 명령어에 --beta2 인자 추가 및 SCRIPT_NAME 변경
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS $SCRIPT_NAME \
    --data-path $DATA_PATH \
    --log-dir $LOG_PATH \
    --save-dir $SAVE_DIR \
    $RESUME_OPTION \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --base-lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --mixup $MIXUP \
    --label-smoothing $LABEL_SMOOTHING \
    --log-interval 300 \
    --save-interval 15

echo "Training finished successfully."