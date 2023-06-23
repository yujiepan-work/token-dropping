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
from token_dropping.routing.gumbel_sigmoid import gumbel_sigmoid

# pylint: disable=missing-class-docstring,unused-import


class RouterOursGatingNoNewToken(torch.nn.Module):
    """only prune, 3 new token.

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.num_new_token = token_dropping_args.num_new_token
        assert self.num_new_token == 1
        self.gating_last = nn.Linear(config.hidden_size, 1)
        self.gating = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            # nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = attention_mask.dtype

        # return hidden_states, attention_mask, torch.ones((B, hidden_states.shape[1], 1), device=device, dtype=dtype), torch.ones((B, L), dtype=dtype, device=device)

        K = min(self.K, L)
        K = max(K - self.num_new_token, 1)

        importance_scores = self_attention_scores.mean(dim=1).mean(dim=1)  # B * L
        importance_scores[:, 0] = 0.  # make class token importance to zero since we add it later
        importance_scores = importance_scores / importance_scores.max(dim=1, keepdim=True).values

        class_logits = torch.zeros((B, L), dtype=dtype, device=device)
        class_logits[:, 0] = 100.
        learned_scores = self.gating_last(self.gating(hidden_states)).squeeze(-1) + class_logits
        if self.training:
            learned_scores = gumbel_sigmoid(learned_scores, hard=False)
        else:
            learned_scores = torch.sigmoid(learned_scores)

        # final_scores = (importance_scores + learned_scores) / 2.
        final_scores = learned_scores
        learnable_01mask = self.re_parameterize(final_scores, threshold=0.5)

        _add_attention = (1. - learnable_01mask) * torch.finfo(dtype).min
        final_attention_mask = torch.minimum(attention_mask, _add_attention[:, None, None, :])

        final_token = hidden_states
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, attention_mask, tome_size, learnable_01mask

    def re_parameterize(self, y_soft, threshold):
        indices = (y_soft >= threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        res = y_hard - y_soft.detach() + y_soft
        return res


class RouterOursNoGumbelSigmoidGatingNoNewToken(torch.nn.Module):
    """only prune, 3 new token.

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.num_new_token = token_dropping_args.num_new_token
        assert self.num_new_token == 1
        self.gating_last = nn.Linear(config.hidden_size, 1)
        self.gating = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            # nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = attention_mask.dtype

        # return hidden_states, attention_mask, torch.ones((B, hidden_states.shape[1], 1), device=device, dtype=dtype), torch.ones((B, L), dtype=dtype, device=device)

        K = min(self.K, L)
        K = max(K - self.num_new_token, 1)

        importance_scores = self_attention_scores.mean(dim=1).mean(dim=1)  # B * L
        importance_scores[:, 0] = 0.  # make class token importance to zero since we add it later
        importance_scores = importance_scores / importance_scores.max(dim=1, keepdim=True).values

        class_logits = torch.zeros((B, L), dtype=dtype, device=device)
        class_logits[:, 0] = 100.
        learned_scores = self.gating_last(self.gating(hidden_states)).squeeze(-1) + class_logits
        learned_scores = torch.sigmoid(learned_scores)

        # final_scores = (importance_scores + learned_scores) / 2.
        final_scores = learned_scores
        learnable_01mask = self.re_parameterize(final_scores, threshold=0.5)

        _add_attention = (1. - learnable_01mask) * torch.finfo(dtype).min
        final_attention_mask = torch.minimum(attention_mask, _add_attention[:, None, None, :])

        final_token = hidden_states
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, attention_mask, tome_size, learnable_01mask

    def re_parameterize(self, y_soft, threshold):
        indices = (y_soft >= threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        res = y_hard - y_soft.detach() + y_soft
        return res


class RouterOursSoftmaxGatingNoNewToken(torch.nn.Module):
    """only prune, 3 new token.

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.token_dropping_args = token_dropping_args
        self.num_new_token = token_dropping_args.num_new_token
        assert self.num_new_token == 1
        self.gating_softmax_last = nn.Linear(config.hidden_size, 2)
        self.gating = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            # nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )

    def forward_eval(self, hidden_states, attention_mask, self_attention_scores, key_layer, tome_size):
        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        learned_scores = self.gating_softmax_last(self.gating(hidden_states))  # B, L, 2
        learnable_01mask = (learned_scores[:, :, 0] >= learned_scores[:, :, 1]).bool()
        learnable_01mask[:, 0] = True
        final_token = hidden_states[learnable_01mask].unsqueeze(0)
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, None, tome_size, None

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        if self.token_dropping_args.is_benchmark_mode:
            return self.forward_eval(hidden_states, attention_mask, self_attention_scores, key_layer, tome_size)

        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        K = min(self.K, L)
        K = max(K - self.num_new_token, 1)

        # importance_scores = self_attention_scores.mean(dim=1).mean(dim=1)  # B * L
        # importance_scores[:, 0] = 0. # make class token importance to zero since we add it later
        # importance_scores = importance_scores / importance_scores.max(dim=1, keepdim=True).values

        class_logits = torch.zeros((B, L, 2), dtype=dtype, device=device)
        class_logits[:, 0, 0] = 100.
        learned_scores = self.gating_softmax_last(self.gating(hidden_states)) + class_logits  # B,L,2
        if self.training:
            learned_scores = F.gumbel_softmax(learned_scores, tau=1.0)[:, :, 0]
        else:
            learned_scores = torch.softmax(learned_scores, dim=-1)[:, :, 0]

        # final_scores = (importance_scores + learned_scores) / 2.
        final_scores = learned_scores
        learnable_01mask = self.re_parameterize(final_scores, threshold=0.5)

        final_token = hidden_states
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, attention_mask, tome_size, learnable_01mask

    def re_parameterize(self, y_soft, threshold):
        indices = (y_soft >= threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        res = y_hard - y_soft.detach() + y_soft
        return res


class RouterOursSoftmaxAddAttentionGatingNoNewToken(torch.nn.Module):
    """only prune, 3 new token.

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.num_new_token = token_dropping_args.num_new_token
        assert self.num_new_token == 1
        self.gating_softmax_last = nn.Linear(config.hidden_size, 2)
        self.gating = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            # nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.threshold = 0.5

    def fusion_method(self, x, y):
        return (x + y) / 2.0

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = attention_mask.dtype

        # return hidden_states, attention_mask, torch.ones((B, hidden_states.shape[1], 1), device=device, dtype=dtype), torch.ones((B, L), dtype=dtype, device=device)

        K = min(self.K, L)
        K = max(K - self.num_new_token, 1)

        importance_scores = self_attention_scores.mean(dim=1).mean(dim=1)  # B * L
        importance_scores[:, 0] = 0.  # make class token importance to zero since we add it later
        importance_scores_min = importance_scores[:, 1:].min(dim=1, keepdim=True).values
        importance_scores_max = importance_scores[:, 1:].max(dim=1, keepdim=True).values
        importance_scores = (importance_scores - importance_scores_min) / importance_scores_max
        importance_scores[:, 0] = 1.

        class_logits = torch.zeros((B, L, 2), dtype=dtype, device=device)
        class_logits[:, 0, 0] = 100.
        learned_scores = self.gating_softmax_last(self.gating(hidden_states)) + class_logits  # B,L,2
        if self.training:
            learned_scores = F.gumbel_softmax(learned_scores, tau=1.0)[:, :, 0]
        else:
            learned_scores = torch.softmax(learned_scores, dim=-1)[:, :, 0]  # B * L

        final_scores = self.fusion_method(importance_scores, learned_scores)
        # final_scores = learned_scores
        learnable_01mask = self.re_parameterize(final_scores, threshold=self.threshold)

        _add_attention = (1. - learnable_01mask) * torch.finfo(dtype).min
        final_attention_mask = torch.minimum(attention_mask, _add_attention[:, None, None, :])

        final_token = hidden_states
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, attention_mask, tome_size, learnable_01mask

    def re_parameterize(self, y_soft, threshold):
        indices = (y_soft >= threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        res = y_hard - y_soft.detach() + y_soft
        return res


class RouterOursSoftmaxTimesAttentionGatingNoNewToken(RouterOursSoftmaxAddAttentionGatingNoNewToken):
    def fusion_method(self, x, y):
        return x * y


class RouterOursAttentionGatingNoNewToken(RouterOursSoftmaxAddAttentionGatingNoNewToken):
    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__(config, num_preserved_tokens)
        self.threshold = 0.0

    def fusion_method(self, pretrained_attention_scores, learned_scores):
        print(pretrained_attention_scores[0, :10])
        return pretrained_attention_scores


RouterTranskimmer = RouterOursSoftmaxGatingNoNewToken
