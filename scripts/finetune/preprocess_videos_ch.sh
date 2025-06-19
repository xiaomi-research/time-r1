#!/bin/bash

MODEL_PATH="./ckpts/Qwen2.5-VL-3B-Instruct_my"
DATASET="charades"
TRAIN_DATA="./dataset/finetune/charades/charades_annotation/train.json"
EVAL_DATA="./dataset/finetune/charades/charades_annotation/val.json"
VIDEO_FOLDER="./dataset/charades/Charades_v1"
MAX_PIX=3584
MIN_PIX=16
NUM_WORKERS=16
OUTPUT_DIR=./dataset/finetune/charades/Charades_preprocessed_data_maxpix_3584

python src/utils/preprocess_dataset.py \
  --model_name $MODEL_PATH \
  --dataset $DATASET \
  --train_data_path $TRAIN_DATA \
  --eval_data_path $EVAL_DATA \
  --video_folder $VIDEO_FOLDER \
  --max_pix_size $MAX_PIX \
  --min_pix_size $MIN_PIX \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR