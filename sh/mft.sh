#!/bin/bash
export base_model_name='/remote-home/yinzhitao/models--meta-llama--Llama-2-7b-hf/snapshots/model'
export NEW_OUTPUT_DIR_BASE="/remote-home/yinzhitao/MaskedThought/MaskedThought-main/models_res"

# Common settings
LEARNING_RATE=1e-5
NUM_TRAIN_EPOCHS=10
WARMUP_RATIO=0.01
PRINT_EVERY=200
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
TRAIN_DIR="data/gsm8k_train.json"
EVAL_DIR="data/gsm8k_train.json" # Consider using a separate validation set for evaluation
TOK_MAX_LENGTH=512
TGT_MAX_LENGTH=512
BF16=True
TF32=True
LORA_RANK=32       # <<--- 修改: LoRA rank 设置为 32
MASK_RATE=0.4
REPLACE=False # <<--- True
LOAD_IN_4BIT=True
USE_PEFT=True
SAVE_STRATEGY="epoch" # Saves checkpoints per epoch
SEED_VALUE=1

# Ensure the new base output directory exists
mkdir -p "${NEW_OUTPUT_DIR_BASE}"

# Function to run training and merging
run_experiment() {
    local exp_suffix=$1
    # local update_mask_rate_val=$2 # <<--- 修改: 此参数不再变化，固定为 True
    local not_mask_source_val=$2 # 参数位置调整
    local not_mask_tgt_val=$3    # <<--- 新增: not_mask_tgt 参数

    # update_mask_rate 固定为 True
    local update_mask_rate_val=True

    export exp_name="mft_${exp_suffix}"
    # Directory for Hugging Face Trainer to save checkpoints
    local output_dir_checkpoints="${NEW_OUTPUT_DIR_BASE}/${exp_name}_checkpoints"
    # Final directory for the merged model (will have the simple exp_name)
    local final_merged_model_dir="${NEW_OUTPUT_DIR_BASE}/${exp_name}"
    # Temporary directory for storing the merged model before renaming
    local temp_merged_output_dir="${NEW_OUTPUT_DIR_BASE}/${exp_name}_merged_temp"

    echo "--------------------------------------------------------------------------"
    echo "Starting Experiment: ${exp_name}"
    echo "Configuration:"
    echo "  LoRA Rank: ${LORA_RANK}"
    echo "  Mask Rate: ${MASK_RATE}"
    echo "  Update Mask Rate: ${update_mask_rate_val}" # <<--- 修改: 固定为 True
    echo "  Not Masking Source: ${not_mask_source_val}"
    echo "  Not Masking Target: ${not_mask_tgt_val}"   # <<--- 新增
    echo "  Output (Checkpoints): ${output_dir_checkpoints}"
    echo "  Output (Final Merged Model): ${final_merged_model_dir}"
    echo "--------------------------------------------------------------------------"

    # Clean up any previous temp or checkpoint directories for this experiment to avoid conflicts
    rm -rf "${output_dir_checkpoints}"
    rm -rf "${temp_merged_output_dir}"
    # If a previous final merged model exists, you might want to remove it or back it up
    # rm -rf "${final_merged_model_dir}"

    python3 main.py \
        --do_train \
        --scene llama_generation \
        --report_to none \
        --seed ${SEED_VALUE} \
        --trainer trainer436 \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --warmup_ratio ${WARMUP_RATIO} \
        --print_every ${PRINT_EVERY} \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --per_device_eval_batch_size 1 \
        --exp_name "${exp_name}" \
        --train_dir ${TRAIN_DIR} \
        --eval_dir ${EVAL_DIR} \
        --gradient_checkpointing False \
        --tok_max_length ${TOK_MAX_LENGTH} \
        --tgt_max_length ${TGT_MAX_LENGTH} \
        --pad_front False \
        --lr_scheduler_type "cosine" \
        --model "${base_model_name}" \
        --model_name "${base_model_name}" \
        --instruct_format True \
        --bf16 ${BF16} \
        --tf32 ${TF32} \
        --output_dir "${output_dir_checkpoints}" \
        --max_mask_rate ${MASK_RATE} \
        --update_mask_rate ${update_mask_rate_val} \
        --mask_rate_warmup 0.66 \
        --save_strategy "${SAVE_STRATEGY}" \
        --replace ${REPLACE} \
        --replace_rate 0.4\
        --load_in_4bit ${LOAD_IN_4BIT} \
        --use_peft ${USE_PEFT} \
        --lora_rank ${LORA_RANK} \
        --not_mask_source ${not_mask_source_val} \
        --not_mask_tgt ${not_mask_tgt_val} 

    echo "Training finished for ${exp_name}."
    echo "Searching for the latest checkpoint in ${output_dir_checkpoints}..."

    latest_checkpoint_dir=$(ls -d "${output_dir_checkpoints}/checkpoint-"* 2>/dev/null | sort -V | tail -n 1)

    if [ -z "${latest_checkpoint_dir}" ] || [ ! -d "${latest_checkpoint_dir}" ]; then
        echo "ERROR: No checkpoint directory found in ${output_dir_checkpoints} for ${exp_name}. Skipping merge."
        latest_checkpoint_dir=$(ls -d "${output_dir_checkpoints}/epoch-"* 2>/dev/null | sort -V | tail -n 1)
        if [ -z "${latest_checkpoint_dir}" ] || [ ! -d "${latest_checkpoint_dir}" ]; then
            echo "ERROR: Also no epoch-based checkpoint directory found. Cannot proceed with merge for ${exp_name}."
        else
             echo "Found epoch-based checkpoint: ${latest_checkpoint_dir}"
        fi
    else
        echo "Found step-based checkpoint: ${latest_checkpoint_dir}"
    fi

    if [ -n "${latest_checkpoint_dir}" ] && [ -d "${latest_checkpoint_dir}" ]; then
        echo "Latest checkpoint for merging: ${latest_checkpoint_dir}"
        echo "Merging LoRA adapter for ${exp_name}..."

        python3 merge.py \
            --base_model_path "${base_model_name}" \
            --lora_adapter_path "${latest_checkpoint_dir}" \
            --merged_model_output_path "${temp_merged_output_dir}"

        if [ $? -eq 0 ]; then
            echo "Merge successful for ${exp_name}."
            echo "Cleaning up checkpoint directory: ${output_dir_checkpoints}"
            rm -rf "${output_dir_checkpoints}"

            echo "Moving merged model from ${temp_merged_output_dir} to ${final_merged_model_dir}"
            rm -rf "${final_merged_model_dir}"
            mv "${temp_merged_output_dir}" "${final_merged_model_dir}"
            echo "Merged model for ${exp_name} is now available at: ${final_merged_model_dir}"
        else
            echo "ERROR: Merge script failed for ${exp_name}."
            echo "Checkpoints remain at: ${output_dir_checkpoints}"
            echo "Attempted merged output (if any) at: ${temp_merged_output_dir}"
        fi
    else
        echo "Skipping merge for ${exp_name} as no valid checkpoint was found."
    fi
    echo "Experiment ${exp_name} processing complete."
    echo "--------------------------------------------------------------------------"
    echo ""
}

# <<--- 修改: 更新实验组合和文件名 --- >>
# 默认 update_mask_rate=True

# 组合 1: not_mask_source=True, not_mask_tgt=True (都不 mask)
run_experiment "lora32_rp0.4_srcNoMask_tgtNoMask" True True

# 组合 2: not_mask_source=True, not_mask_tgt=False (mask target, 不 mask source)
run_experiment "lora32_rp0.4_srcNoMask_tgtMask" True False

# 组合 3: not_mask_source=False, not_mask_tgt=True (mask source, 不 mask target)
run_experiment "lora32_rp0.4_srcMask_tgtNoMask" False True

# 组合 4: not_mask_source=False, not_mask_tgt=False (都 mask)
run_experiment "lora32_rp0.4_srcMask_tgtMask" False False

echo "All experiments finished."
echo "Final merged models (if successful) are in subdirectories under: ${NEW_OUTPUT_DIR_BASE}"