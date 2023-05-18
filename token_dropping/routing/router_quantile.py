import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import token_dropping
from token_dropping.routing.my_attention import MyMultiHeadAttention


# pylint: disable=missing-class-docstring

class RouterQuantile(torch.nn.Module):
    def __init__(self, config, num_preserved_tokens: int) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        self.token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        assert self.token_dropping_args.use_smaller_router
        self.attention = MyMultiHeadAttention(
            query_dim=config.hidden_size, key_dim=config.hidden_size,
            num_units=256, num_heads=4, output_dim=config.hidden_size,
        )


    def forward(self, hidden_states: Tensor, attention_mask: Tensor, self_attention_scores: Tensor):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
        sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D
        key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True: should be masked, [B,L]
        new_token, attention_weights = self.attention(
            sentences,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # B * 1 * D, B * 1 * L
        new_attention_mask = torch.zeros((B, 1, 1, 1), device=attention_mask.device, dtype=attention_mask.dtype)

        class_token = hidden_states[:, :1, :]
        class_attention_mask = torch.zeros((B, 1, 1, 1), device=attention_mask.device, dtype=attention_mask.dtype)

        K = self.K
        importance_scores = self_attention_scores.mean(dim=1).sum(dim=1)  # B * L
        important_indices = torch.topk(importance_scores, K, dim=-1).indices  # [B, 15]
        important_token_mask = (
            torch.zeros((B, L), device=attention_mask.device, dtype=attention_mask.dtype)
            .scatter_(dim=1, index=important_indices, value=1.0)
            .bool()
        )  # [B, L]

        preserved_tokens = hidden_states[important_token_mask].view(B, K, D)
        preserved_attention_mask = attention_mask[important_token_mask.unsqueeze(1).unsqueeze(1)].view(B, 1, 1, K)

        final_token = torch.cat(
            [
                class_token,
                preserved_tokens,
                # new_token,
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = torch.cat(
            [
                class_attention_mask,
                preserved_attention_mask,
                # new_attention_mask,
            ],
            dim=-1,
        )  # B,1,1,L+1

        return final_token, final_attention_mask
