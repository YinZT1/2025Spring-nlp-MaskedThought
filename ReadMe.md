本仓库是对2024年ACL论文Masked Thought: Simply Masking Partial Reasoning Steps Can Improve Mathematical Reasoning Learning of Language Models的复现与提高。

得到的实验结论是：MaskedThought相比于SFT有提高。但得到最优结果的条件并非掩码率为0.4，将token变为固定\<mask\>，而是replace rate=0.4，即对token进行随机替换。但鉴于实验条件不一致且个人理解可能错误，暂无法肯定这一说法。

一些复现的tips：

- 环境的配置最好按照requirements.txt
- logON.log中存储了一切重要信息！

- 当需要对模型进行mft训练时，需要将transformers版本调整为4.36.1，否则main.py与trainer文件夹下的py文件需要修改。
- 当对微调好的模型进行评估时，需要transformers版本为4.51.3（为了满足vllm的版本要求）。
- 当使用强化学习方法优化模型时，需要将datasets调整至3.6.0,以满足trl库的版本要求。
- 运行ppo时，使用的accelerate版本为1.7.0，运行dpo时，使用的accelerate版本为0.28.0。这主要是因为1.7.0提供了较为完善的reward函数接口。
- 全量参数的训练需要60-80G左右。

Config文件夹——处理并解析命令行参数

Data文件夹——存储gsm8k数据集内容，一些数据预处理文件（本次复现不需要在意）

eval_output\llama\math——模型针对gsm8k_test.json中的数学问题进行解答，生成.json文件。指定.json文件地址运行evaluation文件夹下的get_gsm8k_res.py可以得到模型解答gsm8k的acc与invaild。

evaluation文件夹——
	- run_gen_math_greedy_vllm.sh 是用来运行gen_math_greedy.py的脚本，无需在意。
	- gen_math_greedy.py 根据模型地址，结合特定的prompt以及设计好的输出格式，访问gsm8k.json下的问题，生成结果。
	- get_gsm8k_res.py 根据json中模型预测的结果与true answer比对，输出准确率accuracy。

models文件夹——
	- mask_policy_utils.py：处理token序列，该文件实现了掩码逻辑，根据命令行的要求，对token序列进行随机替换(replace) / 更换为<mask>(固定token)。
	- modelling_llama：重写了llama模型预训练的前向计算。
	- modelling_mistral.py : 针对mistral模型的，本次复现并未用到。

sh文件夹——
	我运行实验的全部脚本。

trainer文件夹——
	trainer/Trainer/trainer436.py重写了transformers中llama模型的训练逻辑，主要在于结合modelling_llama中实现的前向计算。

training_scripts文件夹——
	MaskedThought论文原仓库下的脚本，并不适配单GPU情况。因此本次复现并未用到。

logON.log —— 
	记录了实验时的数据

main.py ——
	修改了原仓库的main.py，调整为单GPU运行。

main_dpo.py——
	使用trl库对训练好的模型进行dpo优化。

main_ppo.py——
	使用trl库，结合训练出的reward_model对模型进行ppo优化，ppo可以结合dpo共同优化模型。

merge.py——
	本次实验受限于显存需求，使用lora+4bit简化了训练情况。merge.py则是将lora后的adapter内容与原llama模型进行结合。

ppo_reward.py——
	训练reward_model的核心代码。

