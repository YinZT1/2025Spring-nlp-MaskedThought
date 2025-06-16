# main.py
import random
import numpy as np
import os
import json # 用于加载和处理数据集
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments # DPOTrainer也用这个，或者我们定义的ExtendedTrainingArguments
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset # 修改这里
from trl import DPOTrainer, DPOConfig

# 从你的项目中导入参数类和工具函数
# (确保这些路径和类名与你的项目结构完全一致)
from config.parse_args import parse_args
from data.data_reader import *
# Python 标准库导入
from dataclasses import dataclass, field
from typing import Optional, List


# from data.data_reader import input_builder # DPO不需要这个，它有自己的数据处理
# import trainer.Trainer as Trainer # MFT的Trainer，DPO不需要

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"



def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# merge_lora_model 函数 (假设已在某处定义，DPO通常不需要在开始时合并，它作用于SFT/MFT模型)
# def merge_lora_model(...): ...

# 日志相关 (从你的main.py复制)
from transformers import logging
logging.set_verbosity_info()
logging.enable_explicit_format()
import logging as local_logging
logger = local_logging.getLogger(__name__) # 使用 transformers 的 logger
logger.setLevel(local_logging.INFO) # 或者 logging.INFO
local_logging.basicConfig(
    format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
    level=local_logging.INFO # 或者 logging.INFO
)

@dataclass
class DpoTrainArgs:
    """
    DPO训练专用参数类，整合了dpo.sh中指定且在main_dpo.py中使用的参数。
    """
    # --- 路径参数 ---
    model_name_or_path: str = field(metadata={"help": "作为DPO基础的MFT合并模型的路径。"})
    dpo_train_dir: str = field(metadata={"help": "DPO训练偏好数据集的路径。"})
    dpo_output_dir: str = field(metadata={"help": "DPO训练结果（LoRA适配器或完整模型）的输出目录。"})
    # output_dir 也是 TrainingArguments 的一个参数, dpo.sh 中设置其与 dpo_output_dir 相同
    output_dir: str = field(metadata={"help": "Hugging Face TrainingArguments 的主输出目录。"})
    dpo_eval_dir: Optional[str] = field(default=None, metadata={"help": "DPO评估偏好数据集的路径。"})

    # --- 模型加载与PEFT参数 ---
    load_in_4bit: bool = field(default=False, metadata={"help": "是否以4-bit模式加载模型。"})
    use_peft: bool = field(default=True, metadata={"help": "DPO阶段是否使用PEFT (LoRA)。"})
    lora_rank: int = field(default=16, metadata={"help": "LoRA的秩 (如果 use_peft 为 True)。"})
    lora_no_emb: bool = field(default=False, metadata={"help": "如果为True，LoRA将不应用于词嵌入层。"})
    # dpo.sh 中 --modules_to_save True, main_dpo.py 将其作为标志来决定是否保存 lm_head 和 embed_tokens
    modules_to_save_flag: bool = field(default=False, metadata={"help": "如果使用PEFT且此项为True，则尝试保存 'lm_head' 和 'embed_tokens'。"})

    # --- DPO特定超参数 ---
    dpo_beta: float = field(default=0.1, metadata={"help": "DPO损失函数中的beta参数。"})
    # dpo.sh 同时指定了 dpo_learning_rate 和 learning_rate。
    # main_dpo.py 的逻辑是将 dpo_learning_rate 赋给 TrainingArguments 的 learning_rate。
    # 因此，我们在这里只保留 dpo_learning_rate，并用它来设置训练器的学习率。
    dpo_learning_rate: float = field(default=1e-5, metadata={"help": "DPO训练的学习率。"})
    dpo_num_train_epochs: int = field(default=2, metadata={"help": "DPO训练的轮数。"})
    dpo_max_length: int = field(default=1024, metadata={"help": "DPO数据对 (prompt + chosen/rejected) 的最大总长度。"})
    dpo_max_prompt_length: int = field(default=256, metadata={"help": "DPO数据中prompt部分的最大长度。"})

    # --- Tokenizer 相关参数 (源自dpo.sh, 用于main_dpo.py的内部逻辑) ---
    tok_max_length: int = field(default=1024, metadata={"help": "Tokenizer处理的最大序列长度 (用作dpo_max_length的默认值)。"})
    tgt_max_length: int = field(default=768, metadata={"help": "目标序列的最大长度 (用于dpo_max_prompt_length的逻辑)。"})

    # --- 通用训练参数 (Hugging Face TrainingArguments所需) ---
    per_device_train_batch_size: int = field(default=1, metadata={"help": "每台设备上的训练批量大小。"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "梯度累积步数。"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "学习率调度器类型。"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "学习率预热比例。"})
    logging_steps: int = field(default=10, metadata={"help": "每X步记录一次日志。"})
    save_strategy: str = field(default="epoch", metadata={"help": "模型保存策略 ('steps', 'epoch', 'no')。"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "最多保存的checkpoint数量。"})
    evaluation_strategy: str = field(default="no", metadata={"help": "评估策略 ('steps', 'epoch', 'no')。"})
    bf16: bool = field(default=True, metadata={"help": "是否使用bf16混合精度训练。"})
    tf32: Optional[bool] = field(default=True, metadata={"help": "是否在Ampere GPU上启用TF32。"}) # dpo.sh设置为True
    gradient_checkpointing: bool = field(default=True, metadata={"help": "是否使用梯度检查点以节省显存。"})
    seed: int = field(default=42, metadata={"help": "随机种子。"})
    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"], metadata={"help": "日志和结果上报的目标 (例如 'tensorboard', 'wandb')。"})
    # DPOTrainer 通常需要 remove_unused_columns=False 以保留 'prompt', 'chosen', 'rejected' 列
    remove_unused_columns: bool = field(default=False, metadata={"help": "是否移除数据集中模型不需要的列。"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "从指定checkpoint恢复训练的路径。"})
    # dpo.sh 未指定 optim, 但原脚本检查了 optim=='adamw_bnb_8bit' 的情况
    optim: str = field(default="adamw_torch", metadata={"help": "使用的优化器。"})

def merge_lora_to_base_model(base_model_path_for_merge: str,
                             lora_adapter_path_to_merge: str,
                             merged_model_save_path: str,
                             tokenizer_path_for_saving: str, # 通常和lora_adapter_path一致
                             torch_dtype_str: str = "bfloat16"): # 合并时使用的精度
    """
    将LoRA adapter与基础模型合并，并保存合并后的完整模型和tokenizer。
    """
    logger.info(f"开始合并LoRA adapter...")
    logger.info(f"  基础模型路径 (用于合并): {base_model_path_for_merge}")
    logger.info(f"  LoRA adapter路径: {lora_adapter_path_to_merge}")
    logger.info(f"  合并后模型保存路径: {merged_model_save_path}")
    logger.info(f"  Tokenizer保存路径: {tokenizer_path_for_saving}")
    logger.info(f"  合并精度: {torch_dtype_str}")

    if torch_dtype_str == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # 1. 加载基础模型 (用于合并，最好是非量化版本以保证精度)
    #    注意：如果 base_model_path_for_merge 指向的是一个已量化的模型，
    #    合并后的模型精度可能会受影响。理想情况下，这里的 base_model_path_for_merge
    #    应该指向原始的、未量化的模型版本。
    #    在DPO场景下，这个base_model_path_for_merge就是MFT合并模型的路径。
    base_model_for_final_merge = AutoModelForCausalLM.from_pretrained(
        base_model_path_for_merge,
        torch_dtype=torch_dtype, # 以指定精度加载
        low_cpu_mem_usage=True,
        # device_map="auto" # 合并时可以先放CPU，如果显存紧张
    )

    # 2. 加载分词器 (从adapter路径或tokenizer_path_for_saving加载，确保一致性)
    tokenizer_to_save_with_merged = AutoTokenizer.from_pretrained(tokenizer_path_for_saving)

    # 3. 确保基础模型和分词器词表大小一致
    if base_model_for_final_merge.get_input_embeddings().weight.shape[0] != len(tokenizer_to_save_with_merged):
        logger.info(f"调整基础模型 (用于合并) 的词嵌入层大小，从 {base_model_for_final_merge.get_input_embeddings().weight.shape[0]} 到 {len(tokenizer_to_save_with_merged)}")
        base_model_for_final_merge.resize_token_embeddings(len(tokenizer_to_save_with_merged))
        # 通常，如果adapter包含了embedding的权重，PeftModel加载时会处理好；
        # 如果没有，且词表变了，这里resize后新token的embedding是随机初始化的，可能不是最优。

    # 4. 加载LoRA adapter并应用到基础模型上
    logger.info(f"加载LoRA adapter ({lora_adapter_path_to_merge}) 并应用到基础模型上...")
    final_peft_model = PeftModel.from_pretrained(
        base_model_for_final_merge,
        lora_adapter_path_to_merge
    )

    # 5. 合并权重
    logger.info("开始合并LoRA权重到模型...")
    merged_model = final_peft_model.merge_and_unload()
    logger.info("LoRA权重合并完成。")

    # 6. 保存合并后的完整模型和分词器
    if not os.path.exists(merged_model_save_path):
        os.makedirs(merged_model_save_path)
    logger.info(f"保存最终合并后的模型到: {merged_model_save_path}")
    merged_model.save_pretrained(merged_model_save_path)
    tokenizer_to_save_with_merged.save_pretrained(merged_model_save_path)
    logger.info(f"最终合并模型和分词器已成功保存。")
    return merged_model_save_path

def format_dpo_dataset_from_eval_output(json_file_path: str):
    """
    转换你的 eval_output JSON 文件或 gsm8k_train.json 文件到 DPO 需要的格式。
    这是一个示例函数，你需要根据你的 "judge" 和数据内容来决定
    如何定义 "chosen" 和 "rejected"。

    Args:
        json_file_path (str): 输入的JSON或JSONL文件路径。
    """
    prompts = []
    chosens = []
    rejecteds = []
    count_processed = 0
    count_skipped = 0

    logger.info(f"Formatting DPO dataset from: {json_file_path}")
    # 以下是基于原脚本中eval_output格式的示例逻辑，需要您适配。

    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                data = json.loads(line)
                prompt = data.get("source")[0]
                model_output_target = data.get("target")[0]
                model_output_kd = data.get("kd_data")[0]

                if not prompt or not model_output_target or not model_output_kd:
                    logger.debug(f"Line {line_idx+1}: Missing source, target, or kd_data. Skipping.")
                    count_skipped +=1
                    continue

                judge_value = data.get("judge")

                # !!! 关键逻辑: 根据 'judge' 或其他规则确定 chosen 和 rejected !!!
                # 这是一个非常假设性的例子，您必须根据您的数据含义进行修改。
                # 例如:
                # if judge_value == 'target_is_better':
                # chosen_response = model_output_target
                # rejected_response = model_output_kd
                # elif judge_value == 'kd_is_better':
                # chosen_response = model_output_kd
                # rejected_response = model_output_target
                # else: # 默认或无法判断
                # chosen_response = model_output_kd # 假设kd是较好的参考
                # rejected_response = model_output_target

                # 原脚本中的示例逻辑（需要根据您的 'judge' 字段含义调整）:
                if judge_value == 'false': # 假设 judge='false' 意味着 target 作为 chosen (逻辑可能反了，请检查)
                    chosen_response = model_output_target
                    rejected_response = model_output_kd
                elif isinstance(model_output_kd, str) and len(model_output_kd) > 10:
                    chosen_response = model_output_kd # 黄金答案作为 chosen
                    rejected_response = model_output_target # MFT模型的输出作为 rejected
                else:
                    logger.debug(f"Line {line_idx+1}: Cannot determine preference. Skipping. Data: {data}")
                    count_skipped +=1
                    continue
                
                if chosen_response == rejected_response or not chosen_response or not rejected_response:
                    logger.debug(f"Line {line_idx+1}: Chosen and rejected are same or empty. Skipping.")
                    count_skipped += 1
                    continue
                if not isinstance(prompt, str):
                    logger.warning(f"Prompt is not a string (type: {type(prompt)}), converting to string. Line: {line_idx+1}, Value: {prompt}")
                    prompt = str(prompt)
                if not isinstance(chosen_response, str):
                    logger.warning(f"Chosen response is not a string (type: {type(chosen_response)}), converting to string. Line: {line_idx+1}, Value: {chosen_response}")
                    chosen_response = str(chosen_response)
                if not isinstance(rejected_response, str):
                    logger.warning(f"Rejected response is not a string (type: {type(rejected_response)}), converting to string. Line: {line_idx+1}, Value: {rejected_response}")
                    rejected_response = str(rejected_response)

                # 确保 chosen 和 rejected 不完全相同，并且不是空的（在转换为字符串后检查）
                if chosen_response == rejected_response or not chosen_response or not rejected_response:
                    logger.debug(f"Line {line_idx+1}: Chosen and rejected are same or empty after ensuring string type. Skipping.")
                    count_skipped += 1
                    continue
                
                prompts.append(prompt)
                chosens.append(chosen_response)
                rejecteds.append(rejected_response)
                count_processed += 1

            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {line_idx+1} in {json_file_path}")
                count_skipped += 1
            except Exception as e:
                logger.warning(f"Skipping line {line_idx+1} due to error: {e} in {json_file_path}")
                count_skipped += 1
    
    logger.info(f"Processed {count_processed} entries, skipped {count_skipped} entries from {json_file_path}")
    if count_processed == 0:
        raise ValueError(f"No valid DPO preference pairs could be extracted from {json_file_path}. Please check the file format and conversion logic.")
        
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })

def main_dpo():
    # 使用我们定义的 DpoTrainArgs 和 HfArgumentParser 来解析命令行参数
    parser = HfArgumentParser(DpoTrainArgs)
    dpo_train_args = parser.parse_args_into_dataclasses()[0]

    setup_seed(dpo_train_args.seed)

    # --- 1. 加载 MFT 合并后的模型 (DPO的基础模型) ---
    mft_merged_model_path = dpo_train_args.model_name_or_path
    if not mft_merged_model_path or not os.path.isdir(mft_merged_model_path):
        raise ValueError(f"对于DPO训练，--model_name_or_path 参数必须指向MFT合并模型的有效目录。当前值: {mft_merged_model_path}")
    
    logger.info(f"为DPO加载MFT合并模型，路径: {mft_merged_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(mft_merged_model_path)

    special_tokens_to_add = {}
    if tokenizer.pad_token is None: special_tokens_to_add["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None: special_tokens_to_add["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None: special_tokens_to_add["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None: special_tokens_to_add["unk_token"] = DEFAULT_UNK_TOKEN
    
    if "<mask>" not in tokenizer.get_vocab(): # 假设MFT可能用到 "<mask>"
        current_additional = special_tokens_to_add.get("additional_special_tokens", [])
        if "<mask>" not in current_additional:
            current_additional.append("<mask>")
        special_tokens_to_add["additional_special_tokens"] = current_additional
        logger.info("为DPO tokenizer添加了 '<mask>' token (如果尚不存在)")

    if special_tokens_to_add:
        num_added = tokenizer.add_special_tokens(special_tokens_to_add)
        logger.info(f"为DPO tokenizer添加/更新了 {num_added} 个特殊tokens: {special_tokens_to_add}")

    if tokenizer.pad_token_id is None: 
        logger.warning(f"DPO Tokenizer: pad_token_id 仍然是 None。将其设置为 eos_token_id ({tokenizer.eos_token_id})。")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token # 确保 pad_token 字符串也设置了
    logger.info(f"DPO Tokenizer 使用 pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    compute_dtype = torch.bfloat16 if dpo_train_args.bf16 else torch.float32 # dpo.sh 指定 bf16=True
    
    model_load_kwargs = {
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }

    if dpo_train_args.load_in_4bit:
        logger.info("以4-bit加载MFT合并模型 (DPO的基础模型)。")
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        model_load_kwargs["quantization_config"] = q_config
    # 注意：dpo.sh 没有指定 load_in_8bit，所以这里不添加对应逻辑，除非 DpoTrainArgs 包含它并被设置

    logger.info(f"从MFT合并模型路径加载模型: {mft_merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        mft_merged_model_path,
        **model_load_kwargs
    )
    
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        logger.info(f"调整DPO基础模型词嵌入层大小，从 {model.get_input_embeddings().weight.shape[0]} 到 {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    dpo_peft_config = None
    if dpo_train_args.use_peft:
        logger.info("DPO阶段将应用新的LoRA层。")
        if dpo_train_args.load_in_4bit: # 如果是4bit加载，需要prepare
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=dpo_train_args.gradient_checkpointing)

        target_modules_dpo = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'] # Llama示例
        if not dpo_train_args.lora_no_emb: # dpo.sh 设置为 False
            target_modules_dpo.extend(['lm_head', 'embed_tokens'])
        
        modules_to_save_dpo_list = []
        if dpo_train_args.modules_to_save_flag: # dpo.sh 设置为 True
             modules_to_save_dpo_list = ['lm_head', 'embed_tokens']
        
        dpo_peft_config = LoraConfig(
            r=dpo_train_args.lora_rank, 
            lora_alpha=32, # 通常与lora_rank的两倍或根据经验设置
            lora_dropout=0.05, # 常用值
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules_dpo,
            modules_to_save=modules_to_save_dpo_list if modules_to_save_dpo_list else None
        )
    else:
        logger.info("尝试进行全参数DPO (DPO阶段不使用PEFT)。显存需求会很高。")
        if dpo_train_args.load_in_4bit:
             logger.warning("在4-bit量化模型上进行全参数DPO，将直接微调量化后的权重。")
        if dpo_train_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("为全参数DPO启用了梯度检查点。")


    # --- 2. 准备DPO数据集 ---
    if not dpo_train_args.dpo_train_dir:
        raise ValueError("请通过 --dpo_train_dir 提供DPO训练所需的偏好数据。")

    dpo_input_file_path = dpo_train_args.dpo_train_dir
    
    logger.info(f"为DPO加载训练数据，路径: {dpo_input_file_path}")
    if dpo_input_file_path.endswith(".jsonl"):
        dpo_train_dataset = load_dataset("json", data_files=dpo_input_file_path, split="train")
    elif dpo_input_file_path.endswith(".json"):
        # 调用修正后的函数签名
        dpo_train_dataset = format_dpo_dataset_from_eval_output(dpo_input_file_path)
    else:
        raise ValueError(f"--dpo_train_dir ({dpo_input_file_path}) 文件格式无法直接处理，应为 .jsonl 或特定 .json 格式。")
    
    dpo_eval_dataset = None
    if dpo_train_args.dpo_eval_dir and dpo_train_args.dpo_eval_dir.lower() != "none":
        eval_input_file_path = dpo_train_args.dpo_eval_dir
        is_direct_gsm8k_eval_file = "gsm8k_test.json" in os.path.basename(eval_input_file_path) or \
                                    "gsm8k_train.json" in os.path.basename(eval_input_file_path) # 简单判断
        logger.info(f"为DPO加载评估数据，路径: {eval_input_file_path}")
        if eval_input_file_path.endswith(".jsonl"):
            dpo_eval_dataset = load_dataset("json", data_files=eval_input_file_path, split="train")
        elif eval_input_file_path.endswith(".json"):
             dpo_eval_dataset = format_dpo_dataset_from_eval_output(eval_input_file_path)
        else:
            logger.warning(f"--dpo_eval_dir ({eval_input_file_path}) 格式无法识别，跳过评估数据集。")

    required_cols = ["prompt", "chosen", "rejected"]
    if not all(col in dpo_train_dataset.column_names for col in required_cols):
        raise ValueError(f"DPO训练数据集必须包含 'prompt', 'chosen', 'rejected' 列。实际包含: {dpo_train_dataset.column_names}")
    if dpo_eval_dataset and not all(col in dpo_eval_dataset.column_names for col in required_cols):
         raise ValueError(f"DPO评估数据集必须包含 'prompt', 'chosen', 'rejected' 列。实际包含: {dpo_eval_dataset.column_names}")

    # --- 3. 配置 DPOTrainer ---
    should_perform_eval = (dpo_train_args.evaluation_strategy.lower() != "no" and dpo_eval_dataset is not None)

    if not dpo_train_args.dpo_output_dir: # dpo_output_dir 和 output_dir 在 dpo.sh 中被设为相同的值
        raise ValueError("请通过 --dpo_output_dir (或 --output_dir) 指定DPO模型的输出目录。")
    
    effective_dpo_max_length = dpo_train_args.dpo_max_length
    effective_dpo_max_prompt_length = dpo_train_args.dpo_max_prompt_length



    # 使用 DpoTrainArgs 的字段来构建 TrainingArguments
    hf_training_args_for_dpo = DPOConfig(
        output_dir=dpo_train_args.output_dir, # 主输出目录
        learning_rate=dpo_train_args.dpo_learning_rate, # DPO特定学习率用于训练器
        num_train_epochs=dpo_train_args.dpo_num_train_epochs, # DPO特定轮数
        per_device_train_batch_size=dpo_train_args.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_train_args.gradient_accumulation_steps,
        lr_scheduler_type=dpo_train_args.lr_scheduler_type,
        warmup_ratio=dpo_train_args.warmup_ratio,
        logging_steps=dpo_train_args.logging_steps,
        save_strategy=dpo_train_args.save_strategy,
        save_total_limit=dpo_train_args.save_total_limit,
        do_eval=should_perform_eval,  # 使用计算出的标志
        bf16=dpo_train_args.bf16,
        tf32=dpo_train_args.tf32,
        gradient_checkpointing=dpo_train_args.gradient_checkpointing,
        seed=dpo_train_args.seed,
        report_to=list(dpo_train_args.report_to) if isinstance(dpo_train_args.report_to, str) else dpo_train_args.report_to,
        remove_unused_columns=dpo_train_args.remove_unused_columns, # DPO需要设为False
        logging_dir=os.path.join(dpo_train_args.output_dir, "logs"), # 日志子目录
        resume_from_checkpoint=dpo_train_args.resume_from_checkpoint,
        optim=dpo_train_args.optim,
        beta = dpo_train_args.dpo_beta,
        max_length=effective_dpo_max_length,
        max_prompt_length=effective_dpo_max_prompt_length
    )

    # DPO序列长度参数
    # dpo.sh直接指定了 dpo_max_length 和 dpo_max_prompt_length

    
    # 原脚本中有一段基于tok_max_length和tgt_max_length调整的逻辑，这里保留其思想
    # 但dpo.sh直接提供了dpo_max_length和dpo_max_prompt_length，优先使用它们。
    # 如果需要更复杂的默认逻辑，可以取消下面注释并调整
    # if not dpo_train_args.dpo_max_length: # 如果dpo_max_length未在命令行指定
    #     effective_dpo_max_length = dpo_train_args.tok_max_length
    # if not dpo_train_args.dpo_max_prompt_length: # 如果dpo_max_prompt_length未在命令行指定
    #     effective_dpo_max_prompt_length = effective_dpo_max_length // 2 # 默认一半给prompt
    #     # 确保prompt长度小于总长度，并为响应留出空间
    #     default_target_len = dpo_train_args.tgt_max_length if dpo_train_args.tgt_max_length else 256
    #     if effective_dpo_max_prompt_length >= effective_dpo_max_length:
    #         effective_dpo_max_prompt_length = effective_dpo_max_length - default_target_len
    #         if effective_dpo_max_prompt_length <= 0:
    #             effective_dpo_max_prompt_length = effective_dpo_max_length // 2


    logger.info(f"DPOTrainer 参数: max_length={effective_dpo_max_length}, max_prompt_length={effective_dpo_max_prompt_length}")

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # DPOTrainer会自动创建
        args=hf_training_args_for_dpo,
        train_dataset=dpo_train_dataset,
        eval_dataset=dpo_eval_dataset,
        processing_class=tokenizer,
        peft_config=dpo_peft_config if dpo_train_args.use_peft else None,

        # max_target_length 可以由DPOTrainer根据上面两个参数推断
    )

    # --- 4. 开始DPO训练 ---
    logger.info("开始DPO训练 (根据配置进行全参数或PEFT训练)...")
    train_result = dpo_trainer.train(resume_from_checkpoint=dpo_train_args.resume_from_checkpoint)
    # dpo_trainer.save_metrics("train", train_result.metrics) # 可选：保存训练指标

    # --- 5. 保存DPO训练结果 ---
    final_output_dir = dpo_train_args.output_dir # 使用 TrainingArguments 的 output_dir
    logger.info(f"DPO训练完成。保存模型到 {final_output_dir}")

    dpo_trainer.save_model() # 保存 LoRA adapter (如果use_peft) 或完整模型 (如果全参数)
    tokenizer.save_pretrained(final_output_dir) # 保存分词器

    if dpo_train_args.use_peft:
        logger.info(f"DPO使用了PEFT。DPO LoRA adapter已保存到 {final_output_dir}。")
        logger.info("要获得最终的完整模型，需要将此DPO LoRA adapter与作为其基础的MFT合并模型进行合并。")
        logger.info(f"用于DPO的基础MFT合并模型路径为: {mft_merged_model_path}")
        
        merged_after_dpo_path = os.path.join(final_output_dir, "final_merged_mft_plus_dpo_lora")
        logger.info(f"尝试将DPO LoRA adapter与MFT合并模型合并，保存到: {merged_after_dpo_path}")
        try:
            merge_lora_to_base_model(
                base_model_path_for_merge=mft_merged_model_path, # 这是MFT合并模型的路径
                lora_adapter_path_to_merge=final_output_dir,   # 这是DPO LoRA adapter的路径
                merged_model_save_path=merged_after_dpo_path,
                tokenizer_path_for_saving = final_output_dir,
                torch_dtype_str = "bfloat16" if dpo_train_args.bf16 else "float32"
            )
            logger.info(f"DPO LoRA adapter成功合并，最终模型保存在: {merged_after_dpo_path}")
        except Exception as e:
            logger.error(f"合并DPO LoRA adapter失败: {e}", exc_info=True)
            logger.warning("请检查合并函数和路径。DPO LoRA adapter本身已保存在输出目录中。")
    else:
        logger.info(f"全参数DPO模型已保存到 {final_output_dir}。这就是最终训练好的完整模型。")

# main.py 的 __main__ 部分 (假设 original_main 已经定义)
if __name__ == "__main__":
    main_dpo()
  