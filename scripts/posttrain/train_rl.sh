# training model without sample filtering per epoch

export WANDB_PROJECT=Video-GRPO
export EXP_NAME=3b_kl_cot_gaussian_03_iouv2_2500_ME
export PYTHONPATH=".:$PYTHONPATH"
export DEBUG_MODE="true"
export LOG_PATH="./logs/$EXP_NAME/$EXP_NAME.txt"

OUTDIR=./logs/$EXP_NAME
BASE_MODEL_NAME_OR_PATH="./ckpts/Qwen2.5-VL-3B-Instruct_my"

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12399" \
    main.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --train_data_path ./dataset/timer1/annotations/train_2k5.json \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 200 \
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
    --num_train_epochs 5 \
    --run_name $EXP_NAME \
    --report_to tensorboard \
    --reward_funcs iou_v2 format \
    --temperature 1.0 \
    --prompt_type v1 \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 50 \
    --save_only_model true
