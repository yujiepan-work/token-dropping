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


def masked_attention_scores(self_attention_scores, attention_mask):
    B, H, L = self_attention_scores.shape[:3]
    attention_mask_float = (attention_mask > -10.).float()  # B,1,1,L
    elementwise_attention_mask = torch.bmm(attention_mask_float.view(B, L, 1), attention_mask_float.view(B, 1, L))
    scores = (self_attention_scores * elementwise_attention_mask.unsqueeze(1))
    return scores


# pylint: disable=missing-class-docstring

class RouterOursNewToken(torch.nn.Module):
    """only prune, 1 new token.

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.num_new_token = token_dropping_args.num_new_token
        self.attention = MyMultiHeadAttention(
            query_dim=config.hidden_size, key_dim=config.hidden_size,
            num_units=token_dropping_args.attention_unit,
            num_heads=token_dropping_args.attention_unit // token_dropping_args.attention_head_dim,
            output_dim=config.hidden_size,
        )
        assert self.num_new_token == 1

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = torch.float32

        K = min(self.K, L)
        K = max(K - self.num_new_token, 1)
        if attention_mask is not None:
            importance_scores = masked_attention_scores(self_attention_scores, attention_mask)
        else:
            importance_scores = self_attention_scores
        importance_scores = importance_scores.mean(dim=1).mean(dim=1)  # B * L
        importance_scores[:, 0] = math.inf  # class token
        important_indices = torch.topk(importance_scores, K, dim=-1).indices  # [B, K]
        important_token_mask = (
            torch.zeros((B, L), device=device, dtype=dtype)
            .scatter_(dim=1, index=important_indices, value=1.0)
            .bool()
        )  # [B, L]

        preserved_tokens = hidden_states[important_token_mask].view(B, K, D)
        if attention_mask is not None:
            preserved_attention_mask = attention_mask[important_token_mask.unsqueeze(1).unsqueeze(1)].view(B, 1, 1, K)

        if attention_mask is not None:
            att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
            sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D # avergae
            key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True: should be masked, [B,L]
        else:
            sentences = hidden_states[:, :1, :]
            key_padding_mask = None
        new_token, attention_weights = self.attention(
            sentences,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # B * 1 * D, B * 1 * L
        new_attention_mask = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)

        final_token = torch.cat(
            [
                preserved_tokens,
                new_token,
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = None
        if attention_mask is not None:
            final_attention_mask = torch.cat(
                [
                    preserved_attention_mask,
                    new_attention_mask,
                ],
                dim=-1,
            )  # B,1,1,L+1
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, final_attention_mask, tome_size


class RouterOursNoNew(torch.nn.Module):
    """only prune, no new token.

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.token_dropping_args = token_dropping_args
        self.attention = MyMultiHeadAttention(
            query_dim=config.hidden_size, key_dim=config.hidden_size,
            num_units=256, num_heads=4, output_dim=config.hidden_size,
        )

    def forward_eval(self, hidden_states, attention_mask, self_attention_scores, key_layer, tome_size):
        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        K = min(self.K, L)
        importance_scores = self_attention_scores.mean(dim=1).sum(dim=1)  # B * L
        importance_scores[:, 0] = math.inf  # class token
        important_indices = torch.topk(importance_scores, K, dim=-1).indices  # [B, 15]
        important_token_mask = (
            torch.zeros((B, L), device=device, dtype=torch.bool)
            .scatter_(dim=1, index=important_indices, value=True)
        )  # [B, L]
        final_token = hidden_states[important_token_mask].unsqueeze(dim=0)
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, None, tome_size

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        if self.token_dropping_args.is_benchmark_mode:
            return self.forward_eval(hidden_states, attention_mask, self_attention_scores, key_layer, tome_size)
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = torch.float32

        K = min(self.K, L)
        if attention_mask is not None:
            importance_scores = masked_attention_scores(self_attention_scores, attention_mask)
        else:
            importance_scores = self_attention_scores
        importance_scores = importance_scores.mean(dim=1).mean(dim=1)  # B * L
        importance_scores[:, 0] = math.inf  # class token
        important_indices = torch.topk(importance_scores, K, dim=-1).indices  # [B, 15]
        important_token_mask = (
            torch.zeros((B, L), device=device, dtype=dtype)
            .scatter_(dim=1, index=important_indices, value=1.0)
            .bool()
        )  # [B, L]

        preserved_tokens = hidden_states[important_token_mask].view(B, K, D)
        preserved_attention_mask = None
        if attention_mask is not None:
            preserved_attention_mask = attention_mask[important_token_mask.unsqueeze(1).unsqueeze(1)].view(B, 1, 1, K)

        final_token = torch.cat(
            [
                preserved_tokens,
            ],
            dim=1)  # B * (L+1) * D
        if attention_mask is not None:
            final_attention_mask = torch.cat(
                [
                    preserved_attention_mask,
                ],
                dim=-1,
            )  # B,1,1,L+1
        else:
            final_attention_mask = None
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, final_attention_mask, tome_size


class RouterOursWindowNoNew(torch.nn.Module):
    """only prune, no new token. windowed max pooling

    :param torch: _description_
    :type torch: _type_
    """

    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        token_dropping_args: token_dropping.config.TokenDroppingConfig = config.token_dropping
        self.attention = MyMultiHeadAttention(
            query_dim=config.hidden_size, key_dim=config.hidden_size,
            num_units=token_dropping_args.attention_unit,
            num_heads=token_dropping_args.attention_unit // token_dropping_args.attention_head_dim,
            output_dim=config.hidden_size,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = torch.float32

        K = min(self.K, L)

        importance_scores = self_attention_scores.mean(dim=1).sum(dim=1, keepdim=True)  # B * 1 * L
        importance_scores[:, :, 0] = math.inf  # class token
        adaptive_maxpool = nn.AdaptiveMaxPool1d(K, return_indices=True)
        important_indices = adaptive_maxpool(importance_scores)[1].squeeze(1)  # [B, K]
        preserved_tokens = torch.gather(
            hidden_states, dim=1,
            index=important_indices.unsqueeze(-1).repeat(1, 1, D)
        )

        preserved_attention_mask = None
        assert attention_mask is None, "attention mask has not been correctly developed"
        if attention_mask is not None:
            preserved_attention_mask = attention_mask[important_token_mask.unsqueeze(1).unsqueeze(1)].view(B, 1, 1, K)

        final_token = preserved_tokens  # B * (K) * D
        if attention_mask is not None:
            final_attention_mask = torch.cat(
                [
                    preserved_attention_mask,
                ],
                dim=-1,
            )  # B,1,1,L+1
        else:
            final_attention_mask = None
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, final_attention_mask, tome_size


class RouterOursMultipleNewToken(torch.nn.Module):
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
        self.attention = MyMultiHeadAttention(
            query_dim=config.hidden_size, key_dim=config.hidden_size,
            num_units=token_dropping_args.attention_unit,
            num_heads=token_dropping_args.attention_unit // token_dropping_args.attention_head_dim,
            output_dim=config.hidden_size,
            query_augment=self.num_new_token,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor], self_attention_scores: Tensor, key_layer, tome_size):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = torch.float32

        K = min(self.K, L)
        K = max(K - self.num_new_token, 1)
        importance_scores = self_attention_scores.mean(dim=1).sum(dim=1)  # B * L
        importance_scores[:, 0] = math.inf  # class token
        important_indices = torch.topk(importance_scores, K, dim=-1).indices  # [B, K]
        important_token_mask = (
            torch.zeros((B, L), device=device, dtype=dtype)
            .scatter_(dim=1, index=important_indices, value=1.0)
            .bool()
        )  # [B, L]

        preserved_tokens = hidden_states[important_token_mask].view(B, K, D)
        if attention_mask is not None:
            preserved_attention_mask = attention_mask[important_token_mask.unsqueeze(1).unsqueeze(1)].view(B, 1, 1, K)

        if attention_mask is not None:
            att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
            sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D # avergae
            key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True: should be masked, [B,L]
        else:
            sentences = hidden_states[:, :1, :]
            key_padding_mask = None
        new_token, attention_weights = self.attention(
            sentences,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # B * self.num_new_token * D, B * 1 * L
        new_attention_mask = torch.zeros((B, 1, 1, self.num_new_token), device=device, dtype=dtype)

        final_token = torch.cat(
            [
                preserved_tokens,
                new_token,
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = None
        if attention_mask is not None:
            final_attention_mask = torch.cat(
                [
                    preserved_attention_mask,
                    new_attention_mask,
                ],
                dim=-1,
            )  # B,1,1,L+1
        tome_size = torch.ones((B, final_token.shape[1], 1), device=device, dtype=dtype)
        return final_token, final_attention_mask, tome_size
