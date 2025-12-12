#!/bin/bash

# BIPOST Training Script for meta-llama/Meta-Llama-3-8B-Instruct with BlueOrca and RedOrca datasets
# Server: sv001 - Using scratch space for caches
# GPU: NVIDIA H100 NVL (95.8GB VRAM) - Single GPU
# Algorithms: BDR and BDR_Online
# Datasets: BlueOrca (dataset_1) and RedOrca (dataset_2)

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
SAVE_PATH="./checkpoint/llama3_8b_blueorca_redorca"
WANDB_PROJECT="bipost-sv001"
WANDB_RUN_NAME="llama3_8b_blueorca_redorca"

# Training parameters optimized for single H100 NVL with 8B model
TRAIN_BATCH_SIZE=32
MICRO_TRAIN_BATCH_SIZE=8
LEARNING_RATE=5e-6
SELECTOR_LEARNING_RATE=1e-4
MAX_EPOCHS=2
MAX_SAMPLES_1=56000
MAX_SAMPLES_2=56000
MAX_LEN=2048

# LoRA configuration for Llama models
LORA_RANK=16
LORA_ALPHA=16
TARGET_MODULES="q_proj v_proj"

# DeepSpeed configuration
ZERO_STAGE=2
BF16=true
GRADIENT_CHECKPOINTING=true

# Dataset configuration - BlueOrca and RedOrca
OBJ_1="SFT"
DATASET_1="./examples/datasets/BlueOrca/train.jsonl"
INPUT_KEY="question"
OUTPUT_KEY="response"

OBJ_2="SFT"
DATASET_2="./examples/datasets/RedOrca/train.jsonl"
INPUT_KEY_2="question"
OUTPUT_KEY_2="response"

# BIPOST specific parameters
LAMBD=0.5
SELECTOR_MODE="BDR"
SELECTOR_ACTIVATION="softmax"

# BDR_Online specific parameters
ONLINE_SAMPLE_RATIO=0.1
ONLINE_GENERATION_FREQ=500
ONLINE_SAMPLE_SIZE=1
ONLINE_TEMPERATURE=1.0

# Generation batch size for BDR_Online
GENERATION_BATCH_SIZE=32

# Filter rate for gated loss (top-k selection for model updates)
FILTER_RATE=0.1

# Dynamic mode for BDR_Online
DYNAMIC_MODE=false

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
        --filter_rate)
            FILTER_RATE="$2"
            shift 2
            ;;
        --dynamic_mode)
            DYNAMIC_MODE=true
            shift
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
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --help)
            echo "BIPOST Training Script for Llama 8B with BlueOrca and RedOrca datasets"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Algorithm Options:"
            echo "  --selector_mode MODE     Selector mode: BDR, BDR_Online (default: BDR)"
            echo ""
            echo "BDR_Online Options:"
            echo "  --online_sample_ratio RATIO    Online sample ratio (default: 0.1)"
            echo "  --online_generation_freq FREQ   Online generation frequency (default: 500)"
            echo "  --online_sample_size SIZE       Online sample size (default: 1)"
            echo "  --online_temperature TEMP       Online temperature (default: 1.0)"
            echo "  --generation_batch_size SIZE    Generation batch size (default: 32)"
            echo "  --filter_rate RATE              Filter rate for gated loss (default: 0.1)"
            echo "  --dynamic_mode                  Enable dynamic mode for BDR_Online (select questions based on current weights)"
            echo ""
            echo "Training Options:"
            echo "  --train_batch_size SIZE        Training batch size (default: 32)"
            echo "  --micro_train_batch_size SIZE   Micro training batch size (default: 8)"
            echo "  --learning_rate RATE            Learning rate (default: 5e-6)"
            echo "  --selector_learning_rate RATE   Selector learning rate (default: 1e-4)"
            echo "  --max_epochs EPOCHS             Maximum epochs (default: 2)"
            echo "  --max_samples_1 SAMPLES         Max samples for dataset 1 (default: 56000)"
            echo "  --max_samples_2 SAMPLES         Max samples for dataset 2 (default: 56000)"
            echo "  --max_len_1 LENGTH              Max length for dataset 1 (default: 2048)"
            echo "  --max_len_2 LENGTH              Max length for dataset 2 (default: 2048)"
            echo ""
            echo "LoRA Options:"
            echo "  --lora_rank RANK               LoRA rank (default: 16)"
            echo "  --lora_alpha ALPHA             LoRA alpha (default: 16)"
            echo "  --target_modules MODULES        Target modules (default: q_proj v_proj)"
            echo ""
            echo "Other Options:"
            echo "  --zero_stage STAGE             DeepSpeed ZeRO stage (default: 2)"
            echo "  --save_path PATH               Save path (default: ./checkpoint/llama3_8b_blueorca_redorca)"
            echo "  --wandb_project PROJECT        Wandb project (default: bipost-sv001)"
            echo "  --wandb_run_name NAME          Wandb run name (default: llama3_8b_blueorca_redorca)"
            echo ""
            echo "Save Path Structure:"
            echo "  BDR: ./checkpoint/llama3_8b_blueorca_redorca/BDR_filter_<FILTER_RATE>/"
            echo "  BDR_Online: ./checkpoint/llama3_8b_blueorca_redorca/BDR_Online/ratio_<RATIO>/size_<SIZE>_filter_<FILTER_RATE>/"
            echo "  BDR_Online (dynamic): ./checkpoint/llama3_8b_blueorca_redorca/BDR_Online/ratio_<RATIO>/size_<SIZE>_dynamic_filter_<FILTER_RATE>/"
            echo ""
            echo "Examples:"
            echo "  $0 --selector_mode BDR --filter_rate 0.1"
            echo "  $0 --selector_mode BDR_Online --online_sample_ratio 0.2 --filter_rate 0.1"
            echo "  $0 --selector_mode BDR_Online --dynamic_mode --online_sample_ratio 0.1 --filter_rate 0.1"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "Starting BIPOST training on H100 NVL server sv001..."
echo "Model: $PRETRAIN"
echo "Algorithm: $SELECTOR_MODE"

if [ "$SELECTOR_MODE" = "BDR_Online" ]; then
    echo "  Online sample ratio: $ONLINE_SAMPLE_RATIO"
    echo "  Online generation freq: $ONLINE_GENERATION_FREQ"
    echo "  Online sample size: $ONLINE_SAMPLE_SIZE"
    echo "  Online temperature: $ONLINE_TEMPERATURE"
    echo "  Dynamic mode: $DYNAMIC_MODE"
fi

echo "Dataset 1 (BlueOrca): $DATASET_1"
echo "Dataset 2 (RedOrca): $DATASET_2"
echo "Save path: $SAVE_PATH"
echo "HF Cache: $HF_HOME"
echo "Datasets Cache: $HF_DATASETS_CACHE"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "LoRA rank: $LORA_RANK"
echo "Target modules: $TARGET_MODULES"
echo "Max length: $MAX_LEN"

# Create output directory
mkdir -p "$SAVE_PATH"

# Build the save path based on selector mode
if [ "$SELECTOR_MODE" = "BDR" ]; then
    # Build BDR path with filter rate
    FILTER_SUFFIX="_filter_$FILTER_RATE"
    FINAL_SAVE_PATH="$SAVE_PATH/BDR$FILTER_SUFFIX"
elif [ "$SELECTOR_MODE" = "BDR_Online" ]; then
    # Build BDR_Online path with dynamic mode and filter rate
    DYNAMIC_SUFFIX=""
    if [ "$DYNAMIC_MODE" = true ]; then
        DYNAMIC_SUFFIX="_dynamic"
    fi
    FILTER_SUFFIX="_filter_$FILTER_RATE"
    FINAL_SAVE_PATH="$SAVE_PATH/BDR_Online/ratio_$ONLINE_SAMPLE_RATIO/size_$ONLINE_SAMPLE_SIZE$DYNAMIC_SUFFIX$FILTER_SUFFIX"
else
    echo "Error: Unknown selector_mode: $SELECTOR_MODE. Choose from: BDR, BDR_Online"
    exit 1
fi

echo "Final save path: $FINAL_SAVE_PATH"

# Run training with deepspeed (using random port to avoid conflicts)
MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"
cmd="deepspeed --master_port $MASTER_PORT --module bipost.cli.train_selector \
    --pretrain $PRETRAIN \
    --save_path $FINAL_SAVE_PATH \
    --ckpt_path ./checkpoint \
    --dataset_1 $DATASET_1 \
    --obj_1 $OBJ_1 \
    --dataset_2 $DATASET_2 \
    --obj_2 $OBJ_2 \
    --input_key $INPUT_KEY \
    --output_key $OUTPUT_KEY \
    --input_key_2 $INPUT_KEY_2 \
    --output_key_2 $OUTPUT_KEY_2 \
    --max_samples_1 $MAX_SAMPLES_1 \
    --max_samples_2 $MAX_SAMPLES_2 \
    --max_len_1 $MAX_LEN \
    --max_len_2 $MAX_LEN \
    --max_epochs $MAX_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_alpha $LORA_ALPHA \
    --lora_rank $LORA_RANK \
    --target_modules $TARGET_MODULES \
    --selector_mode $SELECTOR_MODE \
    --selector_learning_rate $SELECTOR_LEARNING_RATE \
    --selector_activation $SELECTOR_ACTIVATION \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --max_ckpt_num $MAX_CKPT_NUM \
    --max_ckpt_mem $MAX_CKPT_MEM \
    --zero_stage $ZERO_STAGE \
    --bf16 \
    --gradient_checkpointing \
    --use_wandb $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME"

# Add BDR_Online specific parameters
if [ "$SELECTOR_MODE" = "BDR_Online" ]; then
    cmd="$cmd --online_sample_ratio $ONLINE_SAMPLE_RATIO \
    --online_generation_freq $ONLINE_GENERATION_FREQ \
    --online_sample_size $ONLINE_SAMPLE_SIZE \
    --online_temperature $ONLINE_TEMPERATURE \
    --generation_batch_size $GENERATION_BATCH_SIZE \
    --filter_rate $FILTER_RATE"
    
    if [ "$DYNAMIC_MODE" = true ]; then
        cmd="$cmd --dynamic_mode"
    fi
fi

# Add lambd parameter for BDR and BDR_Online
cmd="$cmd --lambd $LAMBD"

# Add filter_rate parameter for BDR and BDR_Online
if [ "$SELECTOR_MODE" = "BDR" ] || [ "$SELECTOR_MODE" = "BDR_Online" ]; then
    cmd="$cmd --filter_rate $FILTER_RATE"
fi

echo ""
echo "Executing command:"
echo "$cmd"
echo ""

# Execute the training command
eval $cmd

echo ""
echo "Training completed!"
echo "Results saved to: $FINAL_SAVE_PATH"
