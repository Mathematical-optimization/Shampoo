#!/bin/bash

# --- 스크립트 설정 ---
# set -e: 명령어가 실패하면 즉시 스크립트를 중단합니다.
# set -x: 실행되는 명령어를 터미널에 출력하여 디버깅을 돕습니다.
set -e
set -x

# --- 사용자 설정 변수 ---
# GPU 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4

# 데이터셋을 캐시할 경로 (절대 경로 권장)
DATA_PATH="$HOME/.cache/huggingface/datasets"

# TensorBoard 로그 및 모델 체크포인트를 저장할 기본 경로
OUTPUT_DIR="./training_output_transformer"

# Python 스크립트 파일 이름
SCRIPT_NAME="Transformer.py"

# === Transformer 모델 하이퍼파라미터 (AlgoPerf 사양) ===
D_MODEL=256           # Model dimension ()
N_HEADS=4            # Number of attention heads
N_LAYERS=4            # Number of encoder/decoder layers
D_FF=1024             # Feed-forward dimension
MAX_SEQ_LEN=128       # Maximum sequence length
DROPOUT=0.1           # Dropout rate
LABEL_SMOOTHING=0.1   # Label smoothing (AlgoPerf default)

# === 학습 하이퍼파라미터 ===
EPOCHS=90
BATCH_SIZE_PER_GPU=64  # GPU 메모리에 맞춰 조절 (V100 16GB 기준)
WORKERS=4               # 데이터 로딩에 사용할 CPU 워커 수

# === Optimizer 하이퍼파라미터 (튜닝 대상) ===
BASE_LR=0.0012            # Base learning rate (scaled by schedule)
WARMUP_STEPS=103700      # Warmup steps (Transformer paper default)
WEIGHT_DECAY=0.0       # Weight decay (AlgoPerf uses 0.0 for Transformer)
BETA1=0.9              # Adam beta1
BETA2=0.98             # Adam beta2 (AlgoPerf uses 0.98)
GRAD_CLIP=1.0          # Gradient clipping

# === Distributed Shampoo 하이퍼파라미터 ===
MAX_PRECONDITIONER_DIM=1024    # Maximum preconditioner dimension
PRECONDITION_FREQUENCY=50     # Preconditioning frequency
START_PRECONDITIONING=50     # Step to start preconditioning

# === 평가 설정 ===
SAVE_INTERVAL=10        # 체크포인트 저장 간격
BLEU_INTERVAL=5        # BLEU 평가 간격

# === 디버깅/개발용 옵션 ===
MAX_TRAIN_SAMPLES=""   # 빈 값이면 전체 데이터 사용, 숫자 입력시 샘플 제한
RESUME_FROM=""         # 체크포인트에서 재개 (필요시 설정)

# --- 실행 설정 ---
# 실험 이름 생성 (주요 하이퍼파라미터 포함)
RUN_NAME="transformer_shampoo_LR${BASE_LR}_WS${WARMUP_STEPS}_B2${BETA2}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$OUTPUT_DIR/$RUN_NAME/logs"
CHECKPOINT_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"

# 디렉토리 생성
mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_DIR

# 재개 옵션 설정
RESUME_OPTION=""
if [ ! -z "$RESUME_FROM" ]; then
    RESUME_OPTION="--resume $RESUME_FROM"
    echo "Resuming training from: $RESUME_FROM"
fi

# 샘플 제한 옵션 설정 (디버깅용)
SAMPLE_OPTION=""
if [ ! -z "$MAX_TRAIN_SAMPLES" ]; then
    SAMPLE_OPTION="--max-train-samples $MAX_TRAIN_SAMPLES"
    echo "Training with limited samples: $MAX_TRAIN_SAMPLES"
fi

# --- 분산 학습 실행 ---
echo "========================================================"
echo "AlgoPerf Transformer on WMT17 DE-EN Training"
echo "Optimizer: Distributed Shampoo"
echo "========================================================"
echo "Model Configuration:"
echo "  - d_model: $D_MODEL"
echo "  - n_heads: $N_HEADS"
echo "  - n_layers: $N_LAYERS"
echo "  - d_ff: $D_FF"
echo "  - max_seq_len: $MAX_SEQ_LEN"
echo "========================================================"
echo "Training Configuration:"
echo "  - GPUs: $N_GPUS"
echo "  - Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "  - Epochs: $EPOCHS"
echo "  - LR: $BASE_LR (with warmup: $WARMUP_STEPS steps)"
echo "  - Beta1: $BETA1, Beta2: $BETA2"
echo "  - Label Smoothing: $LABEL_SMOOTHING"
echo "  - Gradient Clipping: $GRAD_CLIP"
echo "========================================================"
echo "Shampoo Configuration:"
echo "  - Max Preconditioner Dim: $MAX_PRECONDITIONER_DIM"
echo "  - Precondition Frequency: $PRECONDITION_FREQUENCY"
echo "  - Start Preconditioning: $START_PRECONDITIONING"
echo "========================================================"
echo "Output:"
echo "  - Log Path: $LOG_PATH"
echo "  - Checkpoint Path: $CHECKPOINT_DIR"
echo "========================================================"

# torchrun으로 분산 학습 실행
torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$N_GPUS \
    $SCRIPT_NAME \
    --data-path $DATA_PATH \
    --log-dir $LOG_PATH \
    --checkpoint-dir $CHECKPOINT_DIR \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --n-layers $N_LAYERS \
    --d-ff $D_FF \
    --max-seq-len $MAX_SEQ_LEN \
    --dropout $DROPOUT \
    --label-smoothing $LABEL_SMOOTHING \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --grad-clip $GRAD_CLIP \
    --max-preconditioner-dim $MAX_PRECONDITIONER_DIM \
    --precondition-frequency $PRECONDITION_FREQUENCY \
    --start-preconditioning-step $START_PRECONDITIONING \
    --save-interval $SAVE_INTERVAL \
    --bleu-interval $BLEU_INTERVAL \
    $SAMPLE_OPTION \
    $RESUME_OPTION

echo "========================================================"
echo "Training finished successfully!"
echo "Results saved to: $OUTPUT_DIR/$RUN_NAME"
echo "========================================================"

# 선택적: 최종 BLEU 점수 출력
if [ -f "$CHECKPOINT_DIR/best_model.pth" ]; then
    echo "Best model checkpoint saved at: $CHECKPOINT_DIR/best_model.pth"
fi
