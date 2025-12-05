#!/bin/sh
export CUDA_VISIBLE_DEVICES=0  # Single GPU setup

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

export GPUNUM=1  # Single GPU
export MASTER_PORT=16900

export MODEL="1b"

export TGT="gemma-7b"

MODEL_NAME="./data/pythia2${TGT}/TokAlign-Init-1B"

export DATASET_PATH="./data/pretrain-dataset/pile00-${TGT}-tokenized"
# export DATASET_PATH="./data/pretrain-dataset/pile00-sample-${TGT}-tokenized"

export CONFIG_FILE="./data/Deepspeed-Configs/zero3.yaml"

export TRAIN_BS=8
export EVAL_BS=1
export GRADIENT_ACC=16

export BLOCK_SIZE=2048

export SEED=0

export LR=6.4e-4
export NUM_STEPS=2500
export NUM_SAVE_STEPS=2500
export EVAL_STEP=10000
export NUM_WORKERS=0
export LOGGING_STEPS=1

export RESUME=False

export TRAIN_START_IDX=0

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${SEED}_${TGT}_S1"

if [ "${RESUME}" != "False" ];
then
PREFIX="${PREFIX}_resume"
ADD_PARAMETERS="${ADD_PARAMETERS} --resume_from_checkpoint ${RESUME}"
fi

MODEL_DIR="${MAIN_DIR}/log/$PREFIX"
LOG_FILE="${MAIN_DIR}/log/${PREFIX}.log"

mkdir -p $MODEL_DIR


echo "Starting STAGE-1 training..."
echo "Log file: ${LOG_FILE}"

accelerate launch \
    --config_file ${CONFIG_FILE} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPUNUM} \
    --num_machines 1 src/clm_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ignore_data_skip True \
    --train_start_idx ${TRAIN_START_IDX} \
    ${ADD_PARAMETERS} \
    --warmup_ratio 0.03 \
    --finetune_embed_only True \
    --use_flash_attn True 2>&1 >$LOG_FILE

STAGE1_EXIT_CODE=$?

if [ ${STAGE1_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: STAGE-1 training failed with exit code ${STAGE1_EXIT_CODE}"
    echo "=========================================="
    echo "Please check the log file for details: ${LOG_FILE}"
    echo "STAGE-2 will not run until STAGE-1 completes successfully."
    exit 1
fi

# STAGE-2 (only run if STAGE-1 checkpoint exists)
STAGE1_CHECKPOINT="${MAIN_DIR}/log/${MODEL}/${SEED}_${TGT}_S1/checkpoint-${NUM_STEPS}"
if [ ! -d "${STAGE1_CHECKPOINT}" ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: STAGE-1 checkpoint not found at ${STAGE1_CHECKPOINT}"
    echo "=========================================="
    echo "STAGE-1 training may have completed but the checkpoint was not saved."
    echo "Please check:"
    echo "  1. The log file: ${LOG_FILE}"
    echo "  2. The output directory: ${MODEL_DIR}"
    echo "  3. Whether training completed all ${NUM_STEPS} steps"
    echo ""
    echo "STAGE-2 will not run until STAGE-1 checkpoint exists."
    exit 1
fi

MODEL_NAME="${STAGE1_CHECKPOINT}"
LR=5e-5
export TRAIN_START_IDX=2560000

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${SEED}_${TGT}_S2"

MODEL_DIR="${MAIN_DIR}/log/$PREFIX"
LOG_FILE="${MAIN_DIR}/log/${PREFIX}.log"

mkdir -p $MODEL_DIR

echo "Starting STAGE-2 training from checkpoint: ${MODEL_NAME}"

accelerate launch \
    --config_file ${CONFIG_FILE} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPUNUM} \
    --num_machines 1 src/clm_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ignore_data_skip True \
    --train_start_idx ${TRAIN_START_IDX} \
    ${ADD_PARAMETERS} \
    --warmup_ratio 0.03 \
    --use_flash_attn True 2>&1 >$LOG_FILE
  
