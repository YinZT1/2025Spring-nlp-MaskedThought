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

