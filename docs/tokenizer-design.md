# BPE Tokenizer 设计

本文只用一个例子说明 tokenizer 各模块的职责。真实实现使用 UTF-8 字节；为了便于阅读，例子把字节写成字符。

## 1. 示例输入和目标

训练语料是：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

词频为：

```text
low: 5, lower: 2, widest: 3, newest: 6
```

目标是从初始字节词表开始，学习一组 merge 规则，使常见片段成为更大的 token。

## 2. 文件分层

```text
cs336_basics/
└── tokenizer/
    ├── __init__.py
    ├── corpus.py          # 读取语料并切分文档/预 token
    ├── specification.py   # vocabulary、merge 规则和 special token 状态
    ├── statistics.py      # 统计相邻 pair 并选择下一条 merge
    ├── train.py           # 组织 BPE 训练循环
    ├── encode.py          # 使用已训练规则把文本转成 token ID
    ├── decode.py          # 把 token ID 还原成文本
    └── serialization.py   # 保存和加载 tokenizer 规则
```

`tokenizer.py` 若是 starter code 要求的入口，应只作为兼容层，转发到上述模块。不要把 Transformer 的 `model.py`、训练循环和 optimizer 放进 tokenizer 子目录。

## 3. `corpus.py`：提供可训练的序列

对示例语料，`CorpusReader` 需要得到四个带频次的序列：

```text
(l, o, w)              : 5
(l, o, w, e, r)        : 2
(w, i, d, e, s, t)     : 3
(n, e, w, e, s, t)     : 6
```

实际实现中，每个元素是一个 UTF-8 字节。`corpus.py` 还负责：

- 按 special token 分隔文档，禁止 merge 跨文档边界；
- 按作业规定进行预 tokenization；
- 在大文件上提供分块或流式读取。

它只提供序列和频次，不决定 vocabulary ID 或 merge 顺序。

## 4. `specification.py`：保存 tokenizer 状态

该文件定义 `TokenizerSpec`、`Vocabulary` 和 merge rule 的数据结构。

训练开始时，词表包含 256 个单字节 token 和作业要求的 special token。每次 merge 都产生一个新 token。例如第一次选择 `(s, t)` 后：

```text
merge: (s, t)
new token: st
```

`specification.py` 需要保证：

- token 和整数 ID 可以双向查询；
- 新 token 的 ID 分配确定；
- merges 按创建顺序保存；
- special token 与普通字节 token 不混淆。

它只描述状态，不执行 pair 统计或文本编码。

## 5. `statistics.py`：选择下一条 merge

对初始序列统计相邻 pair，得到：

```text
(l, o): 7    (o, w): 7    (w, e): 8
(e, r): 2    (w, i): 3    (i, d): 3
(d, e): 3    (e, s): 9    (s, t): 9
(n, e): 6    (e, w): 6
```

`(e, s)` 和 `(s, t)` 同为 9，必须应用作业规定的 tie-break；示例选择 `(s, t)`。因此 `statistics.py` 的职责是：

1. 统计当前序列中相邻 pair 的总频次；
2. 在并列时使用确定性规则；
3. 返回下一条 merge；
4. 在后续优化中维护受影响的 pair 计数。

它不负责创建 token ID，也不负责保存文件。

## 6. `train.py`：执行 BPE 训练循环

`BPETrainer` 把前面的模块串起来：

1. 从 `corpus.py` 获取序列；
2. 让 `statistics.py` 选择 `(s, t)`；
3. 在所有相关序列中替换为 `st`；
4. 将 `(s, t)` 写入 `TokenizerSpec`；
5. 重复统计和合并，直到达到目标词表大小。

示例的前几轮是：

```text
(s, t) -> st
(e, st) -> est
(o, w) -> ow
(l, ow) -> low
(w, est) -> west
(n, e) -> ne
```

第二轮中，`(e, st)` 的频次为 9；之后 `newest` 的序列变为：

```text
(n, e, w, e, s, t)
-> (n, e, w, e, st)
-> (n, e, w, est)
-> (n, e, west)
-> (ne, west)
```

训练输出是 `vocab` 和按创建顺序排列的 `merges`，而不是已经编码好的某个单词。

## 7. `encode.py`：使用训练结果

编码新文本时不再重新统计频率。以 `newest` 为例：

1. 转成初始字节序列 `(n, e, w, e, s, t)`；
2. 按训练时的 merge 顺序检查可用规则；
3. 依次得到 `(ne, west)`；
4. 用 `Vocabulary` 将 `ne` 和 `west` 转成整数 ID。

编码必须在每个 pre-token 内进行，不能跨 special token 或文档边界合并。`encode.py` 只依赖 `TokenizerSpec` 和 `corpus.py` 的分段规则，不依赖训练器。

## 8. `decode.py`：恢复文本

解码时，`Vocabulary` 将 ID 查回字节：

```text
[ID(ne), ID(west)]
-> [b"ne", b"west"]
-> b"newest"
-> "newest"
```

字节应先拼接，再按 UTF-8 解码。非法字节序列按作业要求处理，通常使用 Unicode replacement character。核心检查是合法文本满足：

```text
decode(encode(text)) == text
```

## 9. `serialization.py`：保存可复现结果

训练完成后保存：

- `id -> bytes` 的 vocabulary；
- 有序 merges，例如 `(s, t), (e, st), ...`；
- special token 配置；
- 必要的版本或格式信息。

重新加载后，对 `newest` 的编码必须仍然得到相同的两个 token。序列化模块不应重新训练或修改 merge 顺序。

## 10. 最小验证顺序

围绕同一个例子依次验证：

1. `corpus.py` 是否产生四种序列及正确频次；
2. `statistics.py` 是否得到 `(s, t)`，并正确处理并列；
3. `train.py` 是否产生前六条 merge；
4. `encode.py` 是否把 `newest` 变成 `(ne, west)`；
5. `decode.py` 是否恢复 `newest`；
6. `serialization.py` 保存并加载后结果是否不变；
7. 再加入 UTF-8、special token、空文本和大语料测试。

这条链路通过后，才值得优化 pair 统计、并行预 tokenization 和大文件内存占用。

## 11. Tokenizer 压缩率实验（tokenizer_experiments a）

从 TinyStories 和 OpenWebText 的验证集各随机抽取 10 篇文档，随机种子固定为 `336`，分别使用已训练的 10K 和 32K 词表 tokenizer 编码为整数 ID。采用流式蓄水池抽样，保留文档内的原始空白，不计 `<|endoftext|>` 文档分隔符；每篇文档均检查 `decode(encode(text)) == text`。

压缩率按全部样本文档的总量计算，而不是对每篇文档的比值取平均：

```text
bytes/token = 样本文档的 UTF-8 总字节数 / 编码后的 token 总数
```

| 语料 | 词表大小 | 文档数 | UTF-8 字节数 | token 数 | bytes/token |
| --- | --- | --- | --- | --- | --- |
| TinyStories | 10,000 | 10 | 8,733 | 2,106 | 4.1467 |
| OpenWebText | 32,000 | 10 | 32,949 | 7,389 | 4.4592 |

复现命令（在项目根目录运行）：

```bash
.venv/bin/python -m script.tokenizer.evaluate_tokenizer_compression
```

作业提交用的两句话：

> 在随机抽取的 10 篇 TinyStories 文档上，词表大小为 10K 的 tokenizer 的压缩率为 4.1467 字节/token。在随机抽取的 10 篇 OpenWebText 文档上，词表大小为 32K 的 tokenizer 的压缩率为 4.4592 字节/token。

bytes/token 越大，表示每个 token 平均承载的原文字节越多。本次两个结果来自不同语料，不能仅凭这个比较断言 32K tokenizer 在相同文本上的压缩效果更好。

## 12. 编码吞吐量与 The Pile 耗时估算

使用 OpenWebText 32K tokenizer，对第 11 节同一批 10 篇 OpenWebText 文档计时。样本合计 32,949 个 UTF-8 字节，单进程调用 `encode()` 耗时 97.2505 秒，吞吐量为 `32,949 / 97.2505 ≈ 339` 字节/秒。按 The Pile 的 825 GB 为十进制 `825 × 10⁹` 字节计算，预计耗时 `825 × 10⁹ / 338.81 / 86,400 ≈ 28,183` 天，约 77 年。

复现命令：

```bash
.venv/bin/python -m script.tokenizer.benchmark_tokenizer_throughput
```

可提交的两句中文结论：

> 在 10 篇 OpenWebText 样本文档上，32K tokenizer 的单进程编码吞吐量约为 339 字节/秒。按此速度线性估算，编码 825 GB 的 The Pile 约需 28,183 天（约 77 年）；该估算只计编码时间，实际耗时会随文本内容、硬件及并行方式变化。
