
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Finetune_charades_3B

export PYTHONPATH=".:$PYTHONPATH"

OUTDIR=./logs/finetune/${WANDB_NAME}
export DEBUG_MODE="true"
export LOG_PATH="./logs/finetune/${WANDB_NAME}/${WANDB_NAME}.txt"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12370" \
    finetune.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path ./ckpts/Qwen2.5-VL-3B-Instruct_my \
    --train_data_path ./dataset/finetune/charades/charades_annotation/train.json \
    --video_folder xxx \
    --preprocessed_data_path ./dataset/finetune/charades/Charades_preprocessed_data_maxpix_3584 \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 2 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --reward_funcs iou format \
    --temperature 1.0 \
    --beta 0.0 \
    --prompt_type v1 \
    --is_curriculum_learning false \
    --logging_dir ./logs/finetune/${WANDB_NAME} \
    --save_steps 100 \
    --use_grpo false \
    --save_only_model true
