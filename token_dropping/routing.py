import math
import os
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# int(os.environ.get('YUJIE_PRUNE_K', '15'))

ALL_CACHE = []


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

    def __init__(self, query_dim, key_dim, num_units, num_heads, output_dim):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_output = nn.Linear(in_features=num_units, out_features=output_dim, bias=True)

    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=True,
                ):
        querys = self.W_query(query)  # [N, T_q, num_units]
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


class Router(torch.nn.Module):
    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        if config.use_smaller_router:
            self.attention = MyMultiHeadAttention(
                query_dim=config.hidden_size, key_dim=config.hidden_size,
                num_units=256, num_heads=4, output_dim=config.hidden_size,
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.hidden_size, num_heads=config.num_attention_heads, batch_first=True
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
                new_token,
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = torch.cat(
            [
                class_attention_mask,
                preserved_attention_mask,
                new_attention_mask,
            ],
            dim=-1,
        )  # B,1,1,L+1

        # ALL_CACHE.append(
        #     dict(
        #         importance_scores=importance_scores,
        #         new_token_attention_weights=attention_weights,
        #         attention_mask=attention_mask,
        #     )
        # )
        # torch.save(ALL_CACHE, Path(self.config.training_args.output_dir, 'all_routing.pth'))
        return final_token, final_attention_mask


class Routerv2(torch.nn.Module):
    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        if config.use_smaller_router:
            self.attention = MyMultiHeadAttention(
                query_dim=config.hidden_size, key_dim=config.hidden_size,
                num_units=256, num_heads=4, output_dim=config.hidden_size,
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.hidden_size, num_heads=config.num_attention_heads, batch_first=True
            )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, self_attention_scores: Tensor):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
        sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D
        
        key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True: should be masked, [B,L]
        new_token, new_attention_weights = self.attention(
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
        importance_scores = self_attention_scores.mean(dim=1).mean(dim=1)  # B * L
        importance_scores = importance_scores - new_attention_weights.mean(dim=1)
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
                new_token,
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = torch.cat(
            [
                class_attention_mask,
                preserved_attention_mask,
                new_attention_mask,
            ],
            dim=-1,
        )  # B,1,1,L+1

        # ALL_CACHE.append(
        #     dict(
        #         importance_scores=importance_scores,
        #         new_token_attention_weights=attention_weights,
        #         attention_mask=attention_mask,
        #     )
        # )
        # torch.save(ALL_CACHE, Path(self.config.training_args.output_dir, 'all_routing.pth'))
        return final_token, final_attention_mask


class Routerv3(torch.nn.Module):
    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config
        if config.use_smaller_router:
            self.attention = MyMultiHeadAttention(
                query_dim=config.hidden_size, key_dim=config.hidden_size,
                num_units=256, num_heads=4, output_dim=config.hidden_size,
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.hidden_size, num_heads=config.num_attention_heads, batch_first=True
            )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, self_attention_scores: Tensor):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        K = self.K
        att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
        sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D
        
        key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True: should be masked, [B,L]
        unimportance_scores = self_attention_scores.mean(dim=1).sum(dim=1)  # B * L
        unimportant_indices = torch.topk(unimportance_scores, L-K, dim=-1, largest=False).indices  # [B, 15]
        unimportant_token_mask = (
            torch.zeros((B, L), device=attention_mask.device, dtype=attention_mask.dtype)
            .scatter_(dim=1, index=unimportant_indices, value=1.0)
        )  # [B, L]
        unimportant_token = hidden_states * unimportant_token_mask.unsqueeze(-1) # B * L * D
        unimportant_token_L = unimportant_token_mask.sum(dim=1, keepdim=True)  # B * 1
        unimportant_token = unimportant_token.sum(dim=1) / unimportant_token_L # B * D
        new_token = unimportant_token.unsqueeze(1) # B *1 * D
        new_attention_mask = torch.zeros((B, 1, 1, 1), device=attention_mask.device, dtype=attention_mask.dtype)

        class_token = hidden_states[:, :1, :]
        class_attention_mask = torch.zeros((B, 1, 1, 1), device=attention_mask.device, dtype=attention_mask.dtype)


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
                new_token,
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = torch.cat(
            [
                class_attention_mask,
                preserved_attention_mask,
                new_attention_mask,
            ],
            dim=-1,
        )  # B,1,1,L+1

        # ALL_CACHE.append(
        #     dict(
        #         importance_scores=importance_scores,
        #         new_token_attention_weights=attention_weights,
        #         attention_mask=attention_mask,
        #     )
        # )
        # torch.save(ALL_CACHE, Path(self.config.training_args.output_dir, 'all_routing.pth'))
        return final_token, final_attention_mask


class Routerv4(torch.nn.Module):
    """only prune, no new token.

    :param torch: _description_
    :type torch: _type_
    """
    def __init__(self, config, num_preserved_tokens) -> None:
        super().__init__()
        self.K = num_preserved_tokens
        self.config = config


    def forward(self, hidden_states: Tensor, attention_mask: Tensor, self_attention_scores: Tensor):
        # return hidden_states[:, :15, :], attention_mask[:, :, :, :15]
        # print(attention_mask.shape) # torch.Size([B, 1, 1, L])
        # self-attention_scores: torch.Size([B, H, L, L])

        B, L, D = hidden_states.shape
        att = torch.softmax(attention_mask, dim=-1)  # B,1,1,L
        sentences = torch.bmm(att.squeeze(1), hidden_states)  # B * 1 * D
        
        key_padding_mask = (attention_mask.squeeze(1).squeeze(1) < -10.0).bool()  # True: should be masked, [B,L]
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
            ],
            dim=1)  # B * (L+1) * D
        final_attention_mask = torch.cat(
            [
                class_attention_mask,
                preserved_attention_mask,
            ],
            dim=-1,
        )  # B,1,1,L+1
        return final_token, final_attention_mask

