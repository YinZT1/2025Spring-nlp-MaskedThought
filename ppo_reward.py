# ppo_reward_lora.py (用于训练奖励模型的脚本，集成LoRA)
import json
from datasets import Dataset, DatasetDict
import logging
import os # 确保导入os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments, # RewardConfig 基于这个
    HfArgumentParser # 如果需要更复杂的参数解析
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training # 导入PEFT相关
from trl import RewardConfig, RewardTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
    level=logging.INFO
)

def load_and_prepare_rm_dataset(json_file_path: str):
    """
    加载并转换您的JSON数据为奖励模型训练所需的格式。
    (此函数与您之前的版本基本一致，确保它能正确输出 'prompt', 'chosen', 'rejected' 列)
    """
    prompts = []
    chosens = []
    rejecteds = []
    count_processed = 0
    count_skipped = 0

    logger.info(f"开始处理奖励模型数据集: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                data = json.loads(line)
                
                source_list = data.get("source")
                target_list = data.get("target")
                kd_data_list = data.get("kd_data")
                # judge = data.get("judge") # judge 在这个基础版本中未直接用于筛选，但可以保留

                if source_list is None or not source_list:
                    logger.warning(f"跳过第 {line_idx+1} 行: 'source' 缺失或为空列表。")
                    count_skipped += 1
                    continue
                if target_list is None or not target_list:
                    logger.warning(f"跳过第 {line_idx+1} 行: 'target' 缺失或为空列表。")
                    count_skipped += 1
                    continue
                if kd_data_list is None or not kd_data_list:
                    logger.warning(f"跳过第 {line_idx+1} 行: 'kd_data' 缺失或为空列表。")
                    count_skipped += 1
                    continue
                
                prompt = source_list[0]
                target = target_list[0]
                kd_data = kd_data_list[0]

                if not isinstance(prompt, str):
                    logger.warning(f"跳过第 {line_idx+1} 行: 'source' 的内容不是字符串 (类型: {type(prompt)}).")
                    count_skipped += 1
                    continue
                if not isinstance(target, str):
                    logger.warning(f"跳过第 {line_idx+1} 行: 'target' 的内容不是字符串 (类型: {type(target)}).")
                    count_skipped += 1
                    continue
                if not isinstance(kd_data, str):
                    logger.warning(f"跳过第 {line_idx+1} 行: 'kd_data' 的内容不是字符串 (类型: {type(kd_data)}).")
                    count_skipped += 1
                    continue
                
                chosen_response = target # 核心假设: target 永远是 chosen
                rejected_response = kd_data

                prompts.append(prompt)
                chosens.append(chosen_response)
                rejecteds.append(rejected_response)
                count_processed += 1

            except json.JSONDecodeError:
                logger.warning(f"跳过第 {line_idx+1} 行: 无效的 JSON。")
                count_skipped += 1
            except Exception as e:
                logger.warning(f"跳过第 {line_idx+1} 行: 出现错误 {e}。")
                count_skipped += 1

    logger.info(f"处理完成: {count_processed} 条有效数据，跳过 {count_skipped} 条。")
    if count_processed == 0:
        raise ValueError("未能从文件中提取任何有效的偏好对。")

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })


# --- LoRA 和模型参数定义 (可以考虑使用 HfArgumentParser 或 dataclass 来管理这些参数) ---
# 脚本参数 (为了简化，直接在这里定义，实际项目中推荐用argparse或HfArgumentParser)
model_name_or_path = "/remote-home/yinzhitao/models--meta-llama--Llama-2-7b-hf/snapshots/model"
dataset_path = "/remote-home/yinzhitao/MaskedThought/MaskedThought-main/eval_output/llama/math/mft_rank16.json"
output_dir = "./reward_model_lora_output"

# LoRA 配置
use_peft = True # 设置为 True 来使用 LoRA
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
# 奖励模型通常是序列分类任务，目标模块可能与Causal LM不同。
# 对于Llama这样的模型，如果将其用作序列分类器，LoRA通常仍然应用于其Transformer块中的线性层。
# 如果基础模型是BERT/RoBERTa这类编码器模型，目标模块会是 attention 和 FFN 中的 query, key, value, dense 层。
# 对于Llama作为序列分类器，我们仍然可以尝试应用到其主要的线性层。
# 需要确认 AutoModelForSequenceClassification 加载Llama时，哪些是主要的权重层。
# 通常，即使是分类任务，LoRA仍然应用于原模型的注意力层和MLP层。
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 适用于Llama的典型设置


# --- 1. 加载分词器和基础模型 (用于奖励模型) ---
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=1, # 奖励模型输出1个标量
    torch_dtype=torch.bfloat16, # 或者 torch.float16，根据您的GPU调整
    # 如果基础模型很大，这里也可以考虑 device_map="auto" 或 low_cpu_mem_usage=True
    # 但对于LoRA，通常可以将整个基础模型（即使很大）加载到CPU，然后只将LoRA层和少量相关层放到GPU
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id # 非常重要，确保模型配置也更新
    logger.info(f"Tokenizer pad_token 设置为 eos_token: {tokenizer.eos_token} (ID: {tokenizer.pad_token_id})")


# --- 2. 配置和应用 PEFT (LoRA) ---
if use_peft:
    logger.info("为奖励模型配置LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # <--- 关键：奖励模型是序列分类任务
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none", # 通常LoRA不训练bias
    )
    # 如果模型本身加载时用了量化 (如 BitsAndBytesConfig)，这里可能需要 prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model) # 如果基础模型量化了
    
    model = get_peft_model(model, peft_config)
    logger.info("LoRA已应用于奖励模型。可训练参数:")
    model.print_trainable_parameters()
else:
    logger.info("不使用LoRA，进行全参数微调奖励模型。")


# --- 3. 加载和预处理数据集 ---
# (这里的 preprocess_function 与您之前的版本相同，它将文本转换为input_ids)
# 注意：RewardTrainer 的较新版本可以直接处理文本输入，如果您的版本支持，可以简化这里。
# 我们假设您仍使用 preprocess_function 的方式。
def preprocess_function_for_reward(examples, tokenizer_to_use, max_len):
    new_examples = {
        "input_ids_chosen": [], "attention_mask_chosen": [],
        "input_ids_rejected": [], "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        sep_token = tokenizer_to_use.sep_token if tokenizer_to_use.sep_token is not None else " "
        eos_token = tokenizer_to_use.eos_token if tokenizer_to_use.eos_token is not None else ""

        text_chosen = str(prompt) + sep_token + str(chosen) + eos_token
        text_rejected = str(prompt) + sep_token + str(rejected) + eos_token

        tokenized_chosen = tokenizer_to_use(text_chosen, truncation=True, max_length=max_len)
        tokenized_rejected = tokenizer_to_use(text_rejected, truncation=True, max_length=max_len)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
    return new_examples

rm_dataset_full = load_and_prepare_rm_dataset(dataset_path)
# 考虑数据量，如果数据量很大，可以先取一部分进行测试
# rm_dataset_full = rm_dataset_full.select(range(min(1000, len(rm_dataset_full))))

rm_dataset_split = rm_dataset_full.train_test_split(test_size=0.1, seed=42)
train_dataset = rm_dataset_split['train']
eval_dataset = rm_dataset_split['test']

# 在 RewardConfig 中定义 max_length
reward_config_max_length = 512 # 您可以调整这个值

# 应用预处理 (传递 tokenizer 和 max_length)
# 如果RewardTrainer可以直接处理文本，这一步可以省略，并在RewardTrainer中设置tokenizer和max_length
# 根据您之前的错误，RewardTrainer可能需要processing_class，但它也可能接受预处理好的数据。
# 我们先假设需要预处理。
train_dataset = train_dataset.map(
    preprocess_function_for_reward,
    batched=True,
    fn_kwargs={"tokenizer_to_use": tokenizer, "max_len": reward_config_max_length}
)
eval_dataset = eval_dataset.map(
    preprocess_function_for_reward,
    batched=True,
    fn_kwargs={"tokenizer_to_use": tokenizer, "max_len": reward_config_max_length}
)
logger.info("数据集预处理完成。")


# --- 4. 定义训练参数和初始化 RewardTrainer ---
training_args = RewardConfig(
    output_dir=output_dir,
    num_train_epochs=1, # LoRA微调奖励模型通常1-3个epoch就够了
    per_device_train_batch_size=2, # 使用LoRA后可以尝试比全参数稍大，但仍需根据显存调整
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, # 有效batch size = 2 * 4 = 8
    eval_strategy="epoch",       # 注意：是 eval_strategy 而非 evaluation_strategy (取决于transformers版本)
    save_strategy="epoch",
    learning_rate=1e-4, # LoRA 通常可以使用稍高的学习率
    weight_decay=0.001,
    warmup_ratio=0.1,
    logging_steps=10,
    report_to="tensorboard",
    remove_unused_columns=False, # 如果数据集已预处理，且包含所需列，通常设为False
    max_length=reward_config_max_length, # RewardConfig 也接受 max_length
    # fp16=True, # 如果GPU支持且效果好
    bf16=True, # 推荐用于较新GPU
    gradient_checkpointing=True, # 配合LoRA进一步节省显存
    # optim="adamw_bnb_8bit" # 如果安装了 bitsandbytes 并且想进一步节省优化器显存
)

# 检查 RewardTrainer 的签名，确认如何传递 tokenizer/processing_class
# 根据您之前的报错，它需要 processing_class
trainer = RewardTrainer(
    model=model, # 传入PEFT包装后的模型
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer, # <--- 传递 tokenizer 给 processing_class
    # max_length 参数已在 RewardConfig 中设置，RewardTrainer 会使用它
    # peft_config 参数通常在初始化时不需要再次传递，因为model已经是PeftModel了
    # 如果您的 TRL 版本期望在这里传入原始的 peft_config 对象，可以添加它，但通常不需要。
)

# --- 5. 开始训练 ---
logger.info("开始使用LoRA训练奖励模型...")
trainer.train()

# --- 6. 保存模型 ---
logger.info("训练完成，保存LoRA奖励模型。")
# trainer.save_model() 会自动保存LoRA adapter到training_args.output_dir
trainer.save_model() # 这会保存LoRA adapter
# 基础模型的分词器通常不需要重新保存，除非在LoRA训练中添加了新token。
# 但为了方便后续加载，可以将原始分词器也保存到输出目录。
# tokenizer.save_pretrained(output_dir)
# 如果想直接得到合并后的模型（不推荐在训练脚本中做，除非是最后一步）
# model = model.merge_and_unload()
# model.save_pretrained(os.path.join(output_dir, "final_merged_reward_model"))
# tokenizer.save_pretrained(os.path.join(output_dir, "final_merged_reward_model"))

logger.info(f"LoRA奖励模型适配器已保存到: {output_dir}")
logger.info(f"要使用此奖励模型，您需要先加载基础模型 '{model_name_or_path}'，然后加载此目录中的LoRA适配器。")

if __name__ == "__main__":
    # 可以在这里添加通过 HfArgumentParser 或 tyro 解析命令行参数的逻辑
    # 来覆盖上面定义的 model_name_or_path, dataset_path, output_dir, lora参数等
    pass