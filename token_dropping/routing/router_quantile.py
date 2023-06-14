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
            num_units=config.hidden_size // 2, num_heads=config.num_attention_heads // 2, output_dim=config.hidden_size,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = self_attention_scores.dtype
        K = self.K

        class_token = hidden_states[:, :1, :]
        class_attention_mask = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)

        # importance_scores = self_attention_scores.mean(dim=1).sum(dim=1)  # B * L
        # importance_scores = self_attention_scores.mean(dim=1)[:,0,:] # B * L

        self_attention_scores_non_self = self_attention_scores.mean(dim=1).detach()
        for i in range(self_attention_scores_non_self.shape[-1]):
            self_attention_scores_non_self[:, i, i] = 0.
        importance_scores = self_attention_scores_non_self.sum(dim=1)  # B * L
        important_indices = torch.topk(importance_scores, K, dim=-1).indices  # [B, K]
        important_token_mask = (
            torch.zeros((B, L), device=device, dtype=dtype)
            .scatter_(dim=1, index=important_indices, value=1.0)
            .bool()
        )  # [B, L]


        # token merging
        from token_dropping.routing.tome import bipartite_soft_matching
        merge, _ = bipartite_soft_matching(hidden_states, r=L-K)
        preserved_tokens = merge(hidden_states)
        print(preserved_tokens.shape)

        # preserved_tokens = hidden_states[important_token_mask].view(B, K, D)
        if attention_mask is not None:
            preserved_attention_mask = attention_mask[important_token_mask.unsqueeze(1).unsqueeze(1)].view(B, 1, 1, K)

        # get the sentence representation
        if attention_mask is not None:
            att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
            sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D
        else:
            sentences = hidden_states[:, :1, :] # [B, 1, D]

        # new token
        if attention_mask is None:
            key_padding_mask = None
        else:
            key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True means should be masked, [B,L]
        unpreserved_tokens = hidden_states[~important_token_mask].view(B, -1, D)

        new_token, attention_weights = self.attention(
            sentences,
            unpreserved_tokens,
            unpreserved_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # B * 1 * D, B * 1 * L
        new_attention_mask = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)


        final_token = torch.cat(
            [
                class_token,
                preserved_tokens,
                new_token,
            ],
            dim=1)  # B * (L+1) * D

        final_attention_mask = None
        if attention_mask is not None:
            final_attention_mask = torch.cat(
                [
                    class_attention_mask,
                    preserved_attention_mask,
                    new_attention_mask,
                ],
                dim=-1,
            )  # B,1,1,L+1

        return final_token, final_attention_mask
