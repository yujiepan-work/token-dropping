
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MyMultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]: True means should be masked
    output:
        out --- [N, T_q, output_dim]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads, output_dim, query_augment: int = 1, Wq_bias=False):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.query_augment = query_augment
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units * query_augment, bias=Wq_bias)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_output = nn.Linear(in_features=num_units, out_features=output_dim, bias=True)

    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=True,
                ):
        querys: torch.Tensor = self.W_query(query)  # [N, T_q, num_units]
        querys = querys.reshape(querys.shape[0], querys.shape[1] * self.query_augment, -1)
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(value)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        ## mask
        if key_padding_mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(key_padding_mask, -np.inf)

        scores = F.softmax(scores, dim=3)

        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        out = self.W_output(out)  # [N, T_q, output_dim]

        avg_scores = scores.mean(dim=0)  # [N, T_q, T_k]
        return out, avg_scores
