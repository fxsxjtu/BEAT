#!/bin/bash

# BEAT Training Script
# This script trains the BEAT model on specified datasets

# Configuration
DATASET="amazon"  # Options: amazon, google, yelp
MODEL_NAME="llama_8b"  # Options: llama_8b, qwen_7b, deepseek_8b, llama_3.1_8b, etc.
TASK_NAME="beat_experiment"
GPUS="0,1,2,3"  # Comma-separated GPU IDs

# Hyperparameters
BATCH_SIZE=8
LEARNING_RATE=1e-4
EPOCHS=3
TOKEN_DIM=768
BETA=1.0
ZERO_RATE=0

# Optional: Enable W&B logging
# export WANDB_API_KEY="your_wandb_api_key"
# export WANDB_PROJECT="BEAT"
ENABLE_WANDB=""  # Set to experiment name to enable, empty to disable

echo "========================================"
echo "BEAT Training Script"
echo "========================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL_NAME"
echo "Task: $TASK_NAME"
echo "GPUs: $GPUS"
echo "========================================"

# Stage 1: Train Behavior Tokenization Model
echo "Stage 1: Training Behavior Tokenization Model..."
CUDA_VISIBLE_DEVICES=$GPUS python ../src/models/SemanticTokenAlignment.py \
    --dataset $DATASET \
    --token_dim 64 \
    --token_len 5 \
    --codebook_size 512 \
    --batch_size 10240 \
    --lr 0.001 \
    --epochs 201

if [ $? -ne 0 ]; then
    echo "Error: Stage 1 training failed!"
    exit 1
fi

echo "Stage 1 completed successfully!"
echo ""

# Stage 2: Train Explanation Generation Model
echo "Stage 2: Training Explanation Generation Model..."
CUDA_VISIBLE_DEVICES=$GPUS python ../src/training/train_token_ras.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --task_name $TASK_NAME \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --epochs $EPOCHS \
    --token_dim $TOKEN_DIM \
    --beta $BETA \
    --zero_rate $ZERO_RATE \
    ${ENABLE_WANDB:+--enable_wandb $ENABLE_WANDB}

if [ $? -ne 0 ]; then
    echo "Error: Stage 2 training failed!"
    exit 1
fi

echo "Stage 2 completed successfully!"
echo ""

# Stage 3: Generate and Evaluate Explanations
echo "Stage 3: Generating Explanations..."
CUDA_VISIBLE_DEVICES=$GPUS python ../src/evaluation/api_process.py \
    --task_name $TASK_NAME \
    --dataset $DATASET

if [ $? -ne 0 ]; then
    echo "Error: Explanation generation failed!"
    exit 1
fi

echo "========================================"
echo "Training pipeline completed successfully!"
echo "========================================"
echo "Results saved in: data/$DATASET/repre/$TASK_NAME/"
echo "Explanations saved in: generatedText/$DATASET/$TASK_NAME/"
