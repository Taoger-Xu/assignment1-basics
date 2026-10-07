# 实验记录

训练曲线由 SwanLab 记录。本机驱动与锁定的 PyTorch CUDA 构建不兼容，实验使用本地虚拟环境中的 PyTorch 2.7.1+cu128；运行命令使用 `uv run --no-sync`。

## 7.2 TinyStories

使用 10K BPE tokenizer、context length 256、`d_model=512`、`d_ff=1344`、4 层、16 头、RoPE `theta=10000`。batch 128、训练 10,000 步，共 327.68M tokens；AdamW 使用 `β=(0.9, 0.95)`、`eps=1e-8`、weight decay 0.1，学习率预热 1,000 步至 `2e-3`，再余弦衰减至 `1e-4`。单张 RTX 4090 D 训练。

`data/TinyStoriesV2-GPT4-train.txt` 是 `.txt.1` 的截断前缀；正式训练仅使用后者。用 `uv run --no-sync python script/train/prepare_tinystories.py` 生成 `data/tokenized/tinystories_10k/` 中的 `uint16` token 数组。

| 步数 | 训练 loss | 验证 loss |
| ---: | ---: | ---: |
| 1,000 | 1.9708 | 1.9317 |
| 3,000 | 1.6049 | 1.5962 |
| 10,000 | 1.3367 | 1.3479 |

训练耗时 45.4 分钟，最终验证困惑度约 3.85。Checkpoint：`checkpoints/tinystories_10k.pt`；SwanLab：`swanlog/run-20261007_002831-0rtuabuq/`。此前误将截断前缀和完整文件同时输入训练的实验已作废。

提示词 `Once upon a time, there was a little girl named Lily.`，temperature 0.8、top-p 0.9；模型在生成 164 个 token 后输出 `<|endoftext|>`：

> Once upon a time, there was a little girl named Lily. Lily loved to play with her toys and eat yummy food. Her favorite food was spaghetti. One day, Lily's mom said, "Lily, you need to clean your room before you eat."
>
> Lily was sad, but she started to clean. As she was cleaning, she found a tiny, green box. She opened the box and saw that it was full of spaghetti! She was so happy. She took the spaghetti and started to eat.
>
> But then, something unexpected happened. The spaghetti started to talk! It said, "Please don't eat me! I am a magic spaghetti!" Lily was very surprised. She asked the spaghetti to help her clean her room. The spaghetti agreed, and they cleaned the room together. From that day on, Lily and the magic spaghetti became the best of friends.

文本通顺且有完整情节；童话语料的模式、模型规模和解码参数共同影响生成质量。

## 7.3 结构消融

各组从随机初始化训练 3,000 步，与基线使用相同数据、种子 336、batch 128 和 10,000 步学习率周期。[验证损失曲线](figures/validation_curves.svg)可直接查看，原始日志保存在本地 `ablations/tinystories/`。运行：`uv run --no-sync python -m script.ablations.run_tinystories --steps 3000`。

| 结构 | 峰值学习率 | 验证 loss：1,000 / 2,000 / 3,000 步 |
| --- | ---: | --- |
| 基线 | 2e-3 | 1.9317 / 1.6839 / 1.5962 |
| 无 RMSNorm | 2e-3 | 2.1140 / — / —；第 1,641 步发散 |
| 无 RMSNorm | 5e-4 | 2.2577 / 1.8649 / 1.7445 |
| 无 RMSNorm | 1e-4 | 2.8873 / 2.3884 / 2.1911 |
| post-norm | 2e-3 | 1.9632 / 1.7271 / 1.6339 |
| NoPE | 2e-3 | 2.0736 / 1.7754 / 1.6760 |
| SiLU FFN（`d_ff=4d_model`） | 2e-3 | 1.9995 / 1.7160 / 1.6232 |

移除 RMSNorm 后，原学习率导致发散；降低学习率可稳定训练，但收敛较慢。其余三种修改在相同 3,000 步预算下的验证 loss 均略高于基线。这些是短程结果，验证批次也有采样波动。

## 7.4 OpenWebText

用项目的 32K BPE tokenizer 将训练集编码为 2,727,120,452 tokens（4.371 bytes/token），验证集编码为 66,401,098 tokens（4.367 bytes/token）。脚本：`python -m script.train.prepare_owt`。

模型结构与 TinyStories 相同，词表改为 32K，共 45,224,448 个参数。8 张 RTX 4090 D 数据并行，每卡 batch 16、全局 batch 128，训练 10,000 步、327.68M tokens；AdamW 参数同上，学习率从 `1e-3` 余弦衰减至 `1e-4`，预热 1,000 步。

| 步数 | 训练 loss | 验证 loss |
| ---: | ---: | ---: |
| 1,000 | 5.0919 | 5.0960 |
| 10,000 | 3.9900 | 4.0549 |

训练耗时 26.1 分钟，吞吐 208,856 tokens/s。Checkpoint：`checkpoints/owt_32k.pt`；[SwanLab 云端曲线](https://swanlab.cn/@xujintao/cs336-assignment1/runs/wnq7bgzk)。两套数据使用不同 tokenizer，逐 token loss 不宜直接比较；OWT 文本也更复杂。

## Batch size 对比

在 TinyStories 上从头训练六组，固定模型、种子和约 8.39M 训练 tokens；各组从两档学习率的短程试跑中选择验证 loss 较低者。验证使用相同随机种子、每次 10 × 128 条序列。单卡 RTX 4090 D 的完整训练循环中，batch 176 稳定运行，184 及以上 OOM；因此 176 是本实现的近显存上限组。[学习曲线与吞吐图](figures/learning_curves.svg)按训练 tokens 对齐。

| batch | 选择的峰值 LR | 更新步数 | 最终验证 loss | 耗时 | tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3e-4 | 32,768 | 2.4056 | 1,129.8s | 7,425 |
| 8 | 8e-4 | 4,096 | 2.1126 | 152.8s | 54,895 |
| 32 | 1.5e-3 | 1,024 | 2.0602 | 91.1s | 92,099 |
| 64 | 2e-3 | 512 | 2.1112 | 88.4s | 94,900 |
| 128 | 2e-3 | 256 | 2.3600 | 87.7s | 95,635 |
| 176 | 2e-3 | 187 | 2.5609 | 88.0s | 95,759 |

在这个 token 预算内，batch 32 的验证 loss 最低。更大的 batch 在约 32 后吞吐增益很小，却显著减少参数更新次数；batch 1 虽然更新最多，但吞吐低且验证 loss 更高。此结论只针对这次短程训练和两档学习率试跑；吞吐包含验证与 checkpoint 开销。
