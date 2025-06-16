from datasets import load_dataset
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
from tqdm import tqdm

# 加载数据集
dataset = load_dataset("json", data_files=r"/remote-home/yinzhitao/MaskedThought/MaskedThought-main/eval_output/llama/math/mft_rank16.json", split='train')
dataset = dataset.rename_column("source", "query")

def flatten_query(example):
    example['query'] = example['query'][0] if isinstance(example['query'], list) and len(example['query']) > 0 else example['query']
    return example

dataset = dataset.map(flatten_query)
dataset = dataset.remove_columns(["kd_data", "judge", "target"])

# PPO 配置
config = PPOConfig(
    model_name=r"/root/models_res/mft_lora32_mr0.4_srcNoMask_tgtMask",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
)

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 创建带值头的模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# 为 4-bit 量化准备模型
model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model)

# 添加 LoRA 配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# 应用 LoRA 到预训练模型部分
model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# 加载奖励模型
reward_model = pipeline(
    "text-classification",
    model=r"/remote-home/yinzhitao/MaskedThought/MaskedThought-main/reward_model",
    model_kwargs={"quantization_config": bnb_config, "device_map": {"": 0}},
)

def tokenize(sample):
    if isinstance(sample["query"], list):
        sample["query"] = sample["query"][0] if sample["query"] else ""
    
    # 不使用 return_tensors="pt"，让 PPOTrainer 处理批次
    encoded = tokenizer(
        sample["query"],
        max_length=128,
        padding=False,
        truncation=True,
        return_tensors=None
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "query": sample["query"]
    }

# 应用tokenization
tokenized_dataset = dataset.map(tokenize, batched=False)

# 自定义 collate 函数来处理变长序列
def collate_fn(batch):
    # 提取所有input_ids
    input_ids = [item["input_ids"] for item in batch]
    queries = [item["query"] for item in batch]
    
    # 使用tokenizer进行padding
    padded = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )
    
    # 返回包含padded input_ids和原始queries的字典
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "query": queries
    }

# 初始化 PPOTrainer（使用自定义的collate_fn）
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

# 生成参数
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_length": 256,
}

# 训练循环
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # 从batch中提取input_ids并转换为正确格式
    query_tensors = batch["input_ids"]
    
    # 关键修复：将2D张量转换为1D张量列表
    if query_tensors.dim() == 2:
        # 将 (batch_size, seq_len) 转换为 [tensor(seq_len), tensor(seq_len), ...]
        query_tensors = [query_tensors[i] for i in range(query_tensors.size(0))]
    elif query_tensors.dim() == 1:
        # 如果已经是1D，包装成列表
        query_tensors = [query_tensors]
    
    # 调试信息
    print(f"Number of query tensors: {len(query_tensors)}")
    print(f"First query tensor shape: {query_tensors[0].shape}")
    
    # 从batch中获取原始queries
    original_queries = batch["query"]
    
    # 获取模型生成响应
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    
    # 解码响应 - 只解码新生成的部分
    responses = []
    for i, (query_tensor, response_tensor) in enumerate(zip(query_tensors, response_tensors)):
        # 移除query部分，只保留response
        query_length = len(query_tensor)
        response_only = response_tensor[query_length:]
        response_text = tokenizer.decode(response_only, skip_special_tokens=True)
        responses.append(response_text)
    
    # 计算奖励分数
    texts = [q + " " + r for q, r in zip(original_queries, responses)]
    pipe_outputs = reward_model(texts)
    
    # 转换奖励分数
    rewards = [
        torch.tensor(1.0 if output["label"] == "positive" else -1.0).to(query_tensors[0].device)
        for output in pipe_outputs
    ]
    
    # 执行 PPO 训练步骤
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    # 添加一些调试信息
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Generated {len(responses)} responses")
        print(f"Sample query: {original_queries[0][:100]}...")
        print(f"Sample response: {responses[0][:100]}...")
    
    # 避免无限循环，可以设置最大epoch数
    if epoch >= 100:  # 根据需要调整
        break

# 保存模型
model.save_pretrained("my_ppo_model_lora")
tokenizer.save_pretrained("my_ppo_model_lora")