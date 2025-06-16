#!/bin/bash

# --- 基本配置 ---
# 基础模型父目录
BASE_MODEL_PARENT_DIR="/root/models_res" # (根据原脚本信息推断)

# DPO 偏好数据集路径
# 注意：请确保此路径是正确的，并且适用于所有以下模型的DPO训练。
# 如果每个模型需要不同的DPO数据，您需要修改脚本以在循环中相应地设置此变量。
# 以下路径取自您原脚本中的注释行，请确认其适用性。
export DPO_TRAIN_DATA_PATH="/remote-home/yinzhitao/MaskedThought/MaskedThought-main/eval_output/llama/math/mft_rank16.json" #

# 定义要训练的模型名称数组 (根据您的需求)
MODEL_NAMES=(
    "mft_lora32_mr0.4_srcMask_tgtMask"
    "mft_lora32_mr0.4_srcMask_tgtNoMask"
    "mft_lora32_mr0.4_srcNoMask_tgtMask"
    "mft_lora32_mr0.4_srcNoMask_tgtNoMask"
)

# --- 循环遍历模型并运行 DPO 训练 ---
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    echo "-----------------------------------------------------"
    echo "开始为模型 ${MODEL_NAME} 进行DPO训练"
    echo "-----------------------------------------------------"

    # 1. MFT 合并模型的路径 (这是DPO的基础模型)
    export MFT_MERGED_MODEL_PATH="${BASE_MODEL_PARENT_DIR}/${MODEL_NAME}"

    # 2. DPO 输出目录
    export DPO_EXP_NAME="dpo_${MODEL_NAME}" # 实验名称 (根据模型名动态生成)
    export DPO_OUTPUT_DIR_ACTUAL="./dpo_models_res/${DPO_EXP_NAME}" # DPO LoRA adapter或完整模型保存路径
    mkdir -p ${DPO_OUTPUT_DIR_ACTUAL} #

    echo "DPO基础模型路径: ${MFT_MERGED_MODEL_PATH}"
    echo "DPO输出目录: ${DPO_OUTPUT_DIR_ACTUAL}"
    echo "DPO训练数据路径: ${DPO_TRAIN_DATA_PATH}"

    # --- 运行 DPO 训练 ---
    # 确保你的主脚本是 main_dpo.py
    python3 main_dpo.py \
        --model_name_or_path ${MFT_MERGED_MODEL_PATH} \
        \
        --dpo_train_dir ${DPO_TRAIN_DATA_PATH} \
        --dpo_eval_dir "None" \
        --dpo_output_dir ${DPO_OUTPUT_DIR_ACTUAL} \
        --output_dir ${DPO_OUTPUT_DIR_ACTUAL} \
        \
        --load_in_4bit True \
        --use_peft True \
        --lora_rank 16 \
        --lora_no_emb False \
        --modules_to_save True \
        \
        --dpo_beta 0.1 \
        --dpo_learning_rate 1e-5 \
        --dpo_num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        \
        --tok_max_length 1024 \
        --tgt_max_length 768 \
        --dpo_max_length 1024 \
        --dpo_max_prompt_length 256 \
        \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.05 \
        --logging_steps 10 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --evaluation_strategy "no" \
        \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing True \
        --seed 42 \
        --report_to "tensorboard" \
        --remove_unused_columns False # DPOTrainer 通常需要这个 (此部分参数取自原脚本)

    echo "模型 ${MODEL_NAME} 的DPO训练命令已提交。"
    echo "DPO LoRA adapter (如果使用PEFT) 或完整模型将保存在 ${DPO_OUTPUT_DIR_ACTUAL}" #
    echo "-----------------------------------------------------"
    echo ""
done

echo "所有模型的DPO训练已完成。"