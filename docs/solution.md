# 作业解答

## AdamW 训练资源核算：第 (a) 问

设 batch size 为 (B)、词表大小为 (V)、上下文长度为 (C)、Transformer block 数为 (L)、模型宽度为 (D)、注意力头数为 (H)。按题意取前馈层宽度 (F=d_{\mathrm{ff}}=\frac{8}{3}D\)。所有张量均为 float32，每个元素占 4 字节。

### 参数、参数梯度与优化器状态

输入 token embedding 和输出 LM head 各有 (VD) 个参数。每个 block 包含四个 (D\times D) 的 Q/K/V/O 投影、SwiGLU 的三个矩阵（合计 (3DF) 个参数），以及两个各有 (D) 个参数的 RMSNorm。最终 RMSNorm 另有 (D) 个参数。因此参数总数是

\[
P=2VD+L(4D^2+3DF+2D)+D
 =2VD+L(12D^2+2D)+D.
\]

参数占 (4P) 字节；对应的参数梯度占 (4P) 字节。AdamW 为每个参数保存同形状的一阶矩 (m) 和二阶矩 (v)，优化器状态占 (8P) 字节。这三项合计为 (16P) 字节。

### 激活值

按题目列出的组件计数，约定每个操作保留一份输出。单个 block、单个样本的激活元素数如下：

| 组件 | 元素数 |
| --- | ---: |
| 两个 RMSNorm | (2CD) |
| Q、K、V 投影 | (3CD) |
| (QK^\top) 分数与 softmax 权重 | (2HC^2) |
| 注意力加权求和与输出投影 | (2CD) |
| SwiGLU：(W_1\)、(W_3\)、SiLU、逐元素乘积 | (4CF) |
| SwiGLU：(W_2) 输出 | (CD) |

每个 block 合计 (8CD+4CF+2HC^2) 个元素。block 之外，再计输入 embedding、最终 RMSNorm（共 (2CD)），以及 LM head 的 logits 和交叉熵所需的一份词表大小缓冲区（共 (2CV)）。因此激活显存为

\[
M_{\mathrm{act}}
=4B\left[L(8CD+4CF+2HC^2)+2CD+2CV\right]
=4B\left[L\left(\frac{56}{3}CD+2HC^2\right)+2CD+2CV\right]
\quad\text{字节}.
\]

### 总显存

\[
\boxed{
M_{\mathrm{total}}
=16\left[2VD+L(12D^2+2D)+D\right]
+4B\left[L\left(\frac{56}{3}CD+2HC^2\right)+2CD+2CV\right]
}\quad\text{字节}.
\]

这是用于作业的简化估算。残差、RoPE、反向传播的临时张量和算子工作区等未计入；交叉熵是否另存完整的词表大小缓冲区取决于实现。后续代入具体模型时，应沿用相同的计数约定。

## AdamW 训练资源核算：第 (b) 问

作业给出的 GPT-2 XL 形状为：词表大小 $V=50{,}257$，上下文长度 $C=1{,}024$，层数 $L=48$，模型宽度 $D=1{,}600$，注意力头数 $H=25$，前馈层宽度 $F=4{,}288$。这里使用作业指定、取整到 64 的倍数后的 $F$，而不是直接使用 $8D/3$ 的非整数值。

代入第 (a) 问未替换 $F$ 的原式，参数个数为

$$
P=2VD+L(4D^2+3DF+2D)+D
 =1{,}640{,}452{,}800.
$$

参数、参数梯度和 AdamW 两份矩估计共占

$$
b=16P=26{,}247{,}244{,}800\ \text{字节}
 =26.2472448\ \text{GB}.
$$

每增加一个样本，按第 (a) 问的激活计数约定，增加

$$
\begin{aligned}
a
&=4\left[L(8CD+4CF+2HC^2)+2CD+2CV\right]\\
&=16{,}379{,}944{,}960\ \text{字节}
 =16.37994496\ \text{GB}.
\end{aligned}
$$

因此，以十进制 $1\ \text{GB}=10^9$ 字节计，总显存估算为

$$
\boxed{M(B)=16.37994496\,B+26.2472448\ \text{GB}.}
$$

$B=3$ 时约需 $75.3871$ GB；$B=4$ 时约需 $91.7670$ GB。因此，在 80 GB 限制下，按此简化估算可用的最大整数 batch size 是 **3**。若将 80 GB 理解为 $80\ \text{GiB}$，结论仍为 3。实际运行还需要为未计入的临时张量和框架开销留余量。

## AdamW 训练资源核算：第 (c) 问

设参数总数为 $P$。按每次加、减、乘、除、平方根各计一次 FLOP，且把只依赖超参数和步数的系数预先算好，每个参数的 AdamW 更新需要：

| 操作 | 每参数 FLOPs |
| --- | ---: |
| 权重衰减 $\theta\leftarrow(1-\alpha\lambda)\theta$ | 1 |
| 一阶矩 $m\leftarrow\beta_1m+(1-\beta_1)g$ | 3 |
| 二阶矩 $v\leftarrow\beta_2v+(1-\beta_2)g^2$ | 4 |
| 参数更新 $\theta\leftarrow\theta-\alpha_t m/(\sqrt v+\varepsilon)$ | 5 |
| **合计** | **13** |

因此一次优化器更新约需

$$
\boxed{13P
=13\left[2VD+L(4D^2+3DF+2D)+D\right]\ \text{FLOPs}.}
$$

取 $F=8D/3$ 后为 $13[2VD+L(12D^2+2D)+D]$ FLOPs。若把权重衰减写作 $\theta-\alpha\lambda\theta$ 并逐元素单独计入乘减操作，会得到约 $14P$；两者差别在于是否预计算公共系数。这里不计损失的前向、反向传播，也不计每步只执行一次的标量系数计算。

## AdamW 训练资源核算：第 (d) 问

以一次乘法和一次加法共 2 FLOPs 计矩阵乘法。令 $B$ 为 batch size，GPT-2 XL 形状仍取 $V=50{,}257$、$C=1{,}024$、$L=48$、$D=1{,}600$、$F=4{,}288$。一次前向传播的主要矩阵乘法包括每层 Q/K/V/O 四次投影（$8BCD^2$ FLOPs）、SwiGLU 三次投影（$6BCDF$ FLOPs）、注意力的 $QK^\top$ 和加权求和（共 $4BC^2D$ FLOPs），以及最终 LM head（$2BCDV$ FLOPs）。因此

$$
F_{\mathrm{forward}}
=B\left[L(8CD^2+6CDF+4C^2D)+2CDV\right].
$$

代入 $B=1{,}024$，得 $F_{\mathrm{forward}}=3.6011723718656\times10^{15}$ FLOPs。按题意反向传播为前向的两倍，再加上第 (c) 问的 AdamW 更新 $13P=2.13258864\times10^{10}$ FLOPs，每步合计约 $1.08035384414832\times10^{16}$ FLOPs。

H100 在 50% MFU 下的有效吞吐量为 $0.5\times495\times10^{12}=2.475\times10^{14}$ FLOPs/s。训练 $400{,}000$ 步的理论时间为

$$
\boxed{
\frac{400{,}000\,(3F_{\mathrm{forward}}+13P)}
{0.5\times495\times10^{12}\times3600}
\approx 4{,}850\ \text{小时}
\approx 202\ \text{天}.
}
$$

这是单卡在给定 MFU 下的计算时间估算。第 (b) 问显示 batch size 1,024 无法直接装进 80 GB 显存，因此实际运行还需梯度累积或其他内存管理方法；这些开销未计入估算。
