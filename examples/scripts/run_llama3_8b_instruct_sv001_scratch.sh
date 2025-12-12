#!/bin/bash

# BIPOST Training Script for meta-llama/Meta-Llama-3-8B-Instruct on H100 NVL
# Server: sv001 - Using scratch space for caches
# GPU: NVIDIA H100 NVL (95.8GB VRAM) - Single GPU

set -e

# Set cache directories to use /scratch (1TB available)
export HF_HOME=/scratch/qx232/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/qx232/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/qx232/.cache/datasets
export DATASETS_CACHE_DIR=/scratch/qx232/.cache/datasets

# Create cache directories
mkdir -p /scratch/qx232/.cache/huggingface
mkdir -p /scratch/qx232/.cache/datasets

# Configuration
PRETRAIN="meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_PATH="./checkpoint/llama3_8b_instruct_selector_sv001"
WANDB_PROJECT="bipost-sv001"
WANDB_RUN_NAME="llama3_8b_sv001_scratch"

# Training parameters optimized for single H100 NVL with 8B model
TRAIN_BATCH_SIZE=32
MICRO_TRAIN_BATCH_SIZE=8
LEARNING_RATE=5e-6
SELECTOR_LEARNING_RATE=1e-4
MAX_EPOCHS=3
MAX_SAMPLES_1=56000
MAX_SAMPLES_2=56000
MAX_LEN=2048

# LoRA configuration
LORA_RANK=16
LORA_ALPHA=16
TARGET_MODULES="q_proj v_proj"

# DeepSpeed configuration
ZERO_STAGE=2
BF16=true
GRADIENT_CHECKPOINTING=true

# Dataset configuration
OBJ_1="SFT"
DATASET_1="Open-Orca/OpenOrca"
INPUT_KEY="question"
OUTPUT_KEY="response"

OBJ_2="SFT"
DATASET_2="yahma/alpaca-cleaned"
INPUT_KEY_2="input"
OUTPUT_KEY_2="output"

# BIPOST specific parameters
LAMBD=0.5
SELECTOR_MODE="BDR"
SELECTOR_ACTIVATION="softmax"

# BDR_Online specific parameters
ONLINE_SAMPLE_RATIO=0.1
ONLINE_GENERATION_FREQ=1000
ONLINE_SAMPLE_SIZE=10
ONLINE_TEMPERATURE=1.0

# Generation batch size for BDR_Online
GENERATION_BATCH_SIZE=64
# Logging and checkpointing
SAVE_STEPS=-1
LOGGING_STEPS=100
EVAL_STEPS=500
MAX_CKPT_NUM=2
MAX_CKPT_MEM=2

# Wandb configuration
USE_WANDB=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --selector_mode)
            SELECTOR_MODE="$2"
            shift 2
            ;;
        --online_sample_ratio)
            ONLINE_SAMPLE_RATIO="$2"
            shift 2
            ;;
        --online_generation_freq)
            ONLINE_GENERATION_FREQ="$2"
            shift 2
            ;;
        --online_sample_size)
            ONLINE_SAMPLE_SIZE="$2"
            shift 2
            ;;
        --online_temperature)
            ONLINE_TEMPERATURE="$2"
            shift 2
            ;;
        --generation_batch_size)
            GENERATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --micro_train_batch_size)
            MICRO_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --selector_learning_rate)
            SELECTOR_LEARNING_RATE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --max_samples_1)
            MAX_SAMPLES_1="$2"
            shift 2
            ;;
        --max_samples_2)
            MAX_SAMPLES_2="$2"
            shift 2
            ;;
        --max_len_1)
            MAX_LEN="$2"
            shift 2
            ;;
        --max_len_2)
            MAX_LEN="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --target_modules)
            TARGET_MODULES="$2"
            shift 2
            ;;
        --zero_stage)
            ZERO_STAGE="$2"
            shift 2
            ;;
        --bf16)
            BF16=true
            shift
            ;;
        --no_bf16)
            BF16=false
            shift
            ;;
        --gradient_checkpointing)
            GRADIENT_CHECKPOINTING=true
            shift
            ;;
        --no_gradient_checkpointing)
            GRADIENT_CHECKPOINTING=false
            shift
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --eval_steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --max_ckpt_num)
            MAX_CKPT_NUM="$2"
            shift 2
            ;;
        --max_ckpt_mem)
            MAX_CKPT_MEM="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --no_wandb)
            USE_WANDB=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --selector_mode MODE           Selector mode: BDR, BDR_Online (default: BDR)"
            echo "  --online_sample_ratio RATIO    Online sample ratio for BDR_Online (default: 0.1)"
            echo "  --online_generation_freq FREQ  Online generation frequency (default: 1000)"
            echo "  --online_sample_size SIZE      Online sample size (default: 10)"
            echo "  --online_temperature TEMP      Online temperature (default: 1.0)"
            echo "  --generation_batch_size SIZE   Generation batch size for BDR_Online (default: 64)"
            echo "  --wandb_run_name NAME         Wandb run name (default: llama3_8b_sv001_scratch)"
            echo "  --train_batch_size SIZE        Training batch size (default: 32)"
            echo "  --micro_train_batch_size SIZE  Micro training batch size (default: 8)"
            echo "  --learning_rate LR             Learning rate (default: 5e-6)"
            echo "  --selector_learning_rate LR    Selector learning rate (default: 1e-4)"
            echo "  --max_epochs EPOCHS            Maximum epochs (default: 3)"
            echo "  --max_samples_1 SAMPLES        Max samples for dataset 1 (default: 56000)"
            echo "  --max_samples_2 SAMPLES        Max samples for dataset 2 (default: 56000)"
            echo "  --max_len_1 LENGTH            Max length for dataset 1 (default: 2048)"
            echo "  --max_len_2 LENGTH            Max length for dataset 2 (default: 2048)"
            echo "  --lora_rank RANK              LoRA rank (default: 16)"
            echo "  --lora_alpha ALPHA            LoRA alpha (default: 16)"
            echo "  --target_modules MODULES       Target modules (default: q_proj v_proj)"
            echo "  --zero_stage STAGE             DeepSpeed ZeRO stage (default: 2)"
            echo "  --bf16                        Enable bf16 (default: true)"
            echo "  --no_bf16                     Disable bf16"
            echo "  --gradient_checkpointing       Enable gradient checkpointing (default: true)"
            echo "  --no_gradient_checkpointing    Disable gradient checkpointing"
            echo "  --save_steps STEPS             Save steps (default: -1)"
            echo "  --logging_steps STEPS          Logging steps (default: 100)"
            echo "  --eval_steps STEPS             Evaluation steps (default: 500)"
            echo "  --max_ckpt_num NUM             Max checkpoint number (default: 2)"
            echo "  --max_ckpt_mem MEM             Max checkpoint memory (default: 2)"
            echo "  --wandb_project PROJECT        Wandb project (default: bipost-sv001)"
            echo "  --no_wandb                     Disable Wandb logging"
            echo "  --help                         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --selector_mode BDR_Online --online_sample_size 5"
            echo "  $0 --selector_mode BDR_Online --online_sample_ratio 0.2 --online_generation_freq 500"
            echo "  $0 --train_batch_size 16 --learning_rate 1e-5"
            echo ""
            echo "Save Path Structure:"
            echo "  BDR: $SAVE_PATH/BDR/"
            echo "  BDR_Online: $SAVE_PATH/BDR_Online/ratio_<ONLINE_SAMPLE_RATIO>/size_<ONLINE_SAMPLE_SIZE>/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Update wandb run name to include algorithm (after argument parsing)
WANDB_RUN_NAME="${SELECTOR_MODE,,}_${WANDB_RUN_NAME}"

# Build save path based on selector mode
if [ "$SELECTOR_MODE" = "BDR_Online" ]; then
    # For BDR_Online, include both online_sample_ratio and online_sample_size in the path
    FINAL_SAVE_PATH="$SAVE_PATH/$SELECTOR_MODE/ratio_$ONLINE_SAMPLE_RATIO/size_$ONLINE_SAMPLE_SIZE"
else
    # For BDR, use the standard path
    FINAL_SAVE_PATH="$SAVE_PATH/$SELECTOR_MODE"
fi

echo "Starting BIPOST training on H100 NVL server sv001..."
echo "Model: $PRETRAIN"
echo "Save path: $FINAL_SAVE_PATH"
echo "HF Cache: $HF_HOME"
echo "Datasets Cache: $HF_DATASETS_CACHE"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "LoRA rank: $LORA_RANK"
echo "Max length: $MAX_LEN"
echo "Selector mode: $SELECTOR_MODE"

# Show save path structure
if [ "$SELECTOR_MODE" = "BDR_Online" ]; then
    echo "BDR_Online save structure: $SAVE_PATH/BDR_Online/ratio_$ONLINE_SAMPLE_RATIO/size_$ONLINE_SAMPLE_SIZE/"
    echo "Online sample ratio: $ONLINE_SAMPLE_RATIO"
    echo "Online generation freq: $ONLINE_GENERATION_FREQ"
    echo "Online sample size: $ONLINE_SAMPLE_SIZE"
fi

# Verify cache directories exist
echo "Cache directories:"
ls -la /scratch/qx232/.cache/

# Create save directory
mkdir -p "$FINAL_SAVE_PATH"

# Build command
cmd="deepspeed --module bipost.cli.train_selector"
cmd="$cmd --pretrain $PRETRAIN"
cmd="$cmd --lora_alpha $LORA_ALPHA"
cmd="$cmd --lora_rank $LORA_RANK"
cmd="$cmd --target_modules $TARGET_MODULES"
cmd="$cmd --lambd $LAMBD"
cmd="$cmd --train_batch_size $TRAIN_BATCH_SIZE"
cmd="$cmd --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE"
cmd="$cmd --learning_rate $LEARNING_RATE"
cmd="$cmd --obj_1 $OBJ_1"
cmd="$cmd --dataset_1 $DATASET_1"
cmd="$cmd --input_key $INPUT_KEY"
cmd="$cmd --output_key $OUTPUT_KEY"
cmd="$cmd --max_samples_1 $MAX_SAMPLES_1"
cmd="$cmd --obj_2 $OBJ_2"
cmd="$cmd --dataset_2 $DATASET_2"
cmd="$cmd --input_key_2 $INPUT_KEY_2"
cmd="$cmd --output_key_2 $OUTPUT_KEY_2"
cmd="$cmd --selector_learning_rate $SELECTOR_LEARNING_RATE"
cmd="$cmd --selector_activation $SELECTOR_ACTIVATION"
cmd="$cmd --max_samples_2 $MAX_SAMPLES_2"
cmd="$cmd --max_len_1 $MAX_LEN"
cmd="$cmd --max_len_2 $MAX_LEN"
cmd="$cmd --save_path $FINAL_SAVE_PATH"
cmd="$cmd --max_epochs $MAX_EPOCHS"
cmd="$cmd --zero_stage $ZERO_STAGE"
cmd="$cmd --save_steps $SAVE_STEPS"
cmd="$cmd --logging_steps $LOGGING_STEPS"
cmd="$cmd --eval_steps $EVAL_STEPS"
cmd="$cmd --max_ckpt_num $MAX_CKPT_NUM"
cmd="$cmd --max_ckpt_mem $MAX_CKPT_MEM"

if [ "$BF16" = true ]; then
    cmd="$cmd --bf16"
fi

if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    cmd="$cmd --gradient_checkpointing"
fi

cmd="$cmd --selector_mode $SELECTOR_MODE"
cmd="$cmd --online_sample_ratio $ONLINE_SAMPLE_RATIO"
cmd="$cmd --online_generation_freq $ONLINE_GENERATION_FREQ"
cmd="$cmd --online_sample_size $ONLINE_SAMPLE_SIZE"
cmd="$cmd --online_temperature $ONLINE_TEMPERATURE"
cmd="$cmd --generation_batch_size $GENERATION_BATCH_SIZE"
if [ "$USE_WANDB" = true ]; then
    cmd="$cmd --use_wandb true"
    cmd="$cmd --wandb_project $WANDB_PROJECT"
    cmd="$cmd --wandb_run_name $WANDB_RUN_NAME"
fi

# Run training
echo "Executing command:"
echo "$cmd"
echo ""

eval $cmd

echo "Training completed!"
echo "Check the logs in $FINAL_SAVE_PATH/"
echo "Caches stored in: $HF_HOME and $HF_DATASETS_CACHE"

