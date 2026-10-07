import math

import torch
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # 按 (d_out, d_in) 保存 W，而不是 W 的转置。
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device,dtype=dtype)
        )

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in)；输出: (..., d_out)
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # 参数是【num_embeddings,embedding_dim】的矩阵
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 长度为 d_model、初始值全为 1 的可学习参数
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        rms_inverse = torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )

        result = x_float * rms_inverse * self.weight
        return result.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_ff is None:
            # 取最接近 (8/3) * d_model 的 64 的倍数
            d_ff = max(64, round((8 * d_model / 3) / 64) * 64)

        self.w1 =Linear(d_model, d_ff, device=device,dtype=dtype)
        self.w2 =Linear(d_ff, d_model, device=device,dtype=dtype)
        self.w3 =Linear(d_model, d_ff, device=device,dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        silu_a = a * torch.sigmoid(a)
        gated = silu_a * self.w3(x)
        return self.w2(gated)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        # 第 j 对特征的旋转频率：theta^(-2j / d_k)
        pair_indice = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        frequencies = theta ** (-2 * pair_indice / d_k)

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # angles: (max_seq_len, d_k // 2)
        angles = positions[:, None] * frequencies[None, :]

        self.register_buffer("cos_table", angles.cos(), persistent=False)
        self.register_buffer("sin_table", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        # 方括号是在用它做批量索引
        # cos、sin: (..., seq_len, d_k // 2)
        cos = self.cos_table[token_positions].to(dtype=x.dtype)
        sin = self.sin_table[token_positions].to(dtype=x.dtype)

        # [a, b, c, d] --> [[a, b], [c, d]]，方便 RoPE 分别旋转这两对数
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        # first和second降维度
        first = pairs[..., 0]
        second = pairs[..., 1]

        # 乘法和减法都是逐元素运算，不会增加维度。
        # torch.stack 接收这两个形状相同的结果，并在 dim=-1 指定的位置新建一个维度
        rotated = torch.stack(
            (
                first * cos - second * sin,
                first * sin + second * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 每组减去最大值，避免 exp(很大的数) 溢出
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2,-1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        # True 表示允许关注；False 表示禁止关注。
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)
    return weights @ V

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 不传 theta 时是不带 RoPE 的版本。
        if theta is None:
            self.rope = None
        else:
            if max_seq_len is None:
                raise ValueError("Using RoPE requires max_seq_len")
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., seq_len, d_model)
        # -> (..., num_heads, seq_len, head_dim)
        seq_len = x.shape[-2]
        x = x.reshape(
            *x.shape[:-2], seq_len, self.num_heads, self.head_dim
        )

        return x.transpose(-3, -2)
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)


            # 使同一 token 的位置可广播到所有注意力头。
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            # V 不应用 RoPE。
        # mask[i, j] = True 表示位置 i 可以关注位置 j。
        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device, dtype=torch.bool
        ).tril()

        attended = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # (..., heads, seq_len, head_dim)
        # -> (..., seq_len, heads, head_dim)
        attended = attended.transpose(-3, -2)


        # 合并所有头，恢复 (..., seq_len, d_model)。
        attended = attended.reshape(*x.shape[:-2], seq_len, -1)
        return self.o_proj(attended)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            d_model, vocab_size, device=device, dtype=dtype
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("Sequence exceeds context_length")
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)