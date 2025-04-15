import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * n_heads, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False) -> torch.Tensor:
        # x: (batch_size, seq_len, d_emb)
        input_shape = x.shape
        batch_size, seq_len, d_emb = input_shape
        inter_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # define querry, key and value
        # (batch_size, seq_len, d_emb) -> (batch_size, seq_len, d_emb *3) -> slipt in 3 tensors of (batch_size, seq_len, d_emb)
        q, k, v = self.in_proj(x).chunk(3)

        # (batch_size, seq_len, d_emb) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, n_heads, seq_len, d_head)
        q = q.view(inter_shape).transpose(1, 2)
        k = k.view(inter_shape).transpose(1, 2)
        v = v.view(inter_shape).transpose(1, 2)

        # attention operation
        # (batch_size, n_heads, seq_len, d_head) @ (batch_size, n_heads, d_head, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        weigths = q @ k.transpose(-1, -2)

        if casual_mask:
            # Mask the upper triangle from weigths
            mask = torch.ones_like(weigths, dtype=torch.bool).triu(1)
            weigths.masked_fill_(mask, -torch.inf)
        weigths /= math.sqrt(self.d_head)
        weigths = F.softmax(weigths, dim=-1)

        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        output = weigths @ v

        # (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1.2)
        # (batch_size, seq_len, n_heads, d_head) -> (batch_size, seq_len, d_emb)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output
