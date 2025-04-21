from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from Machine_Learning_Python.Stable_Difusion.src import attention
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int) -> None:
        super().__init__()

        self.tokens_embed = nn.Embedding(n_vocab, n_embed)
        # learnable parameters
        self.positional_embed = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # X: (batch_size, Seq_len) -> (batch_size, Seq_len, n_embed)
        x = self.tokens_embed(tokens)
        x += self.positional_embed
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # SELF attention
        x = self.layernorm_1(x)
        x = self.attention(x)
        x += residual
        # FF LAYER
        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # WHY??
        x = x * torch.sigmoid(1.782 * x)
        x = self.linear_2(x)
        x += residual
        return x


class CLIP(nn.Module):

    def __init__(self):
        # TODO: add vales to a config file
        self.__emb_size = 49408
        self.__emb_dim = 768
        self.__max_seq_len = 77
        self.__n_heads = 12
        self.embed = CLIPEmbedding(
            self.__emb_size, self.__emb_dim, self.__max_seq_len
        )
        self.layers = nn.Module(
            CLIPLayer(self.__n_heads, self.__emb_dim)
            for _ in range(self.__n_heads)
        )
        self.layernorm = nn.LayerNorm(self.__emb_dim)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, Seq_len) -> (batch_size, Seq_len, emb_dim)
        state = self.embed(tokens)
        for layer in self.layers:
            state = layer(state)

        # (batch_size, Seq_len, emb_dim)
        output = self.layernorm(state)
        return output
