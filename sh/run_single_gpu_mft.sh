#!/bin/bash
export base_model_name='/remote-home/yinzhitao/models--meta-llama--Llama-2-7b-hf/snapshots/model'

# 第一次运行
export exp_name='standard_mft_rank32_fixed_mr0.4'
echo "开始第一次运行: ${exp_name}"
python3 main.py \
    --do_train \
    --scene llama_generation \
    --report_to none \
    --seed 1 \
    --trainer trainer436 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --warmup_ratio 0.01 \
    --print_every 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --exp_name ${exp_name} \
    --train_dir data/gsm8k_train.json \
    --eval_dir data/gsm8k_train.json \
    --gradient_checkpointing False \
    --tok_max_length 512 \
    --tgt_max_length 512 \
    --pad_front False \
    --lr_scheduler_type "cosine" \
    --model ${base_model_name} \
    --model_name ${base_model_name} \
    --instruct_format True \
    --bf16 True \
    --tf32 True \
    --output_dir ./${exp_name} \
    --max_mask_rate 0.4 \
    --update_mask_rate False \
    --mask_rate_warmup 0.66 \
    --save_strategy "epoch" \
    --replace False \
    --replace_rate 0.4 \
    --load_in_4bit True \
    --use_peft True \
    --lora_rank 32 \
    --not_mask_source True

export exp_name='standard_mft_rank32_updated_mr0.3'
echo "开始第二次运行: ${exp_name}"
python3 main.py \
    --do_train \
    --scene llama_generation \
    --report_to none \
    --seed 1 \
    --trainer trainer436 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --warmup_ratio 0.01 \
    --print_every 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --exp_name ${exp_name} \
    --train_dir data/gsm8k_train.json \
    --eval_dir data/gsm8k_train.json \
    --gradient_checkpointing False \
    --tok_max_length 512 \
    --tgt_max_length 512 \
    --pad_front False \
    --lr_scheduler_type "cosine" \
    --model ${base_model_name} \
    --model_name ${base_model_name} \
    --instruct_format True \
    --bf16 True \
    --tf32 True \
    --output_dir ./${exp_name} \
    --max_mask_rate 0.3 \
    --update_mask_rate True \
    --mask_rate_warmup 0.66 \
    --save_strategy "epoch" \
    --replace False \
    --replace_rate 0.4 \
    --load_in_4bit True \
    --use_peft True \
    --lora_rank 32 \
    --not_mask_source True

export exp_name='standard_mft_rank32_updated_mr0.5'
echo "开始第三次运行: ${exp_name}"
python3 main.py \
    --do_train \
    --scene llama_generation \
    --report_to none \
    --seed 1 \
    --trainer trainer436 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --warmup_ratio 0.01 \
    --print_every 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --exp_name ${exp_name} \
    --train_dir data/gsm8k_train.json \
    --eval_dir data/gsm8k_train.json \
    --gradient_checkpointing False \
    --tok_max_length 512 \
    --tgt_max_length 512 \
    --pad_front False \
    --lr_scheduler_type "cosine" \
    --model ${base_model_name} \
    --model_name ${base_model_name} \
    --instruct_format True \
    --bf16 True \
    --tf32 True \
    --output_dir ./${exp_name} \
    --max_mask_rate 0.5 \
    --update_mask_rate True \
    --mask_rate_warmup 0.66 \
    --save_strategy "epoch" \
    --replace False \
    --replace_rate 0.4 \
    --load_in_4bit True \
    --use_peft True \
    --lora_rank 32 \
    --not_mask_source True


export exp_name='worst_try2'
echo "开始第四次运行: ${exp_name}"
python3 main.py \
    --do_train \
    --scene llama_generation \
    --report_to none \
    --seed 1 \
    --trainer trainer436 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --warmup_ratio 0.01 \
    --print_every 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --exp_name ${exp_name} \
    --train_dir data/gsm8k_train.json \
    --eval_dir data/gsm8k_train.json \
    --gradient_checkpointing False \
    --tok_max_length 512 \
    --tgt_max_length 512 \
    --pad_front False \
    --lr_scheduler_type "cosine" \
    --model ${base_model_name} \
    --model_name ${base_model_name} \
    --instruct_format True \
    --bf16 True \
    --tf32 True \
    --output_dir ./${exp_name} \
    --max_mask_rate 0.4 \
    --update_mask_rate True \
    --mask_rate_warmup 0.66 \
    --save_strategy "epoch" \
    --replace True \
    --replace_rate 1 \
    --load_in_4bit True \
    --use_peft True \
    --lora_rank 32 \
    --not_mask_source True

