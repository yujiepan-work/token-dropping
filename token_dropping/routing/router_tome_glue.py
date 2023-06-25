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

class RouterToMeGlue(torch.nn.Module):
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
        self.force_r = None

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])
        # assert attention_mask is None or attention_mask.count_nonzero() == 0
        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = self_attention_scores.dtype
        K = self.K

        # token merging
        from token_dropping.routing.tome import bipartite_soft_matching
        r = max(0, L-K) if self.force_r is None else self.force_r
        merge, _ = bipartite_soft_matching(hidden_states, r=r, class_token=True)
        preserved_tokens = merge(hidden_states)
        
        final_attention_mask = None
        if attention_mask is not None:
            final_attention_mask = torch.zeros((B, 1, 1, preserved_tokens.shape[1]), device=device, dtype=dtype)
        print(preserved_tokens.shape)
        return preserved_tokens, final_attention_mask


class RouterToMeGlueUseKey(torch.nn.Module):
    def __init__(self, config, num_preserved_tokens: int) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        self.token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        # assert self.token_dropping_args.use_smaller_router
        # self.attention = MyMultiHeadAttention(
        #     query_dim=config.hidden_size, key_dim=config.hidden_size,
        #     num_units=config.hidden_size // 2, num_heads=config.num_attention_heads // 2, output_dim=config.hidden_size,
        # )
        self.force_r = None

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])
        # assert attention_mask is None or attention_mask.count_nonzero() == 0
        # key_layer: B,H,L,D
        # if attention_mask is not None:
        #     attention_mask_bool = (attention_mask > -10.).squeeze(1).squeeze(1)
        #     hidden_states = hidden_states[attention_mask_bool]
        #     attention_mask = None

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        K = self.K
        assert B==1

        # return hidden_states, attention_mask, torch.ones((B, L, 1), device=device, dtype=dtype)

        # token merging
        from token_dropping.routing.tome import bipartite_soft_matching, do_nothing
        r = max(0, L-K) if self.force_r is None else self.force_r
        merge, _ = bipartite_soft_matching(key_layer.mean(dim=1), r=r, class_token=True, for_onnx_export=self.token_dropping_args.export_onnx)
        preserved_tokens = merge(hidden_states)

        if tome_size is None:
            tome_size = torch.ones((B, L, 1), device=device, dtype=dtype)
        new_tome_size = merge(tome_size, mode='sum')
        
        final_attention_mask = None
        if attention_mask is not None:
            final_attention_mask = torch.zeros((B, 1, 1, preserved_tokens.shape[1]), device=device, dtype=dtype)

        return preserved_tokens, final_attention_mask, new_tome_size