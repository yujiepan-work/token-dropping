from dataclasses import dataclass, field
from typing import Optional
import os

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
)

# pylint: disable=missing-function-docstring


@dataclass
class TokenDroppingConfig:
    """
    Config for token dropping.
    """
    export_onnx: bool = field(
        default=False,
        metadata={"help": "None"},
    )
    token_pruning_strategy: str = field(
        default="1:1",
        metadata={"help": "token_pruning_strategy"},
    )
    add_prompt_token: bool = field(
        default=False,
        metadata={"help": "add one learnable token at the beginning"},
    )
    router_version: str = field(
        default='1',
        metadata={"help": "None"},
    )
    freeze_model: bool = field(
        default=True,
        metadata={"help": "freeze pretrained model"},
    )
    use_smaller_router: bool = field(
        default=True,
    )

    def __post_init__(self):
        strategy = self.token_pruning_strategy.replace('_', ',').replace('-', ':')
        self.token_pruning_strategy = eval(f'{{{strategy}}}')  # pylint: disable=eval-used


def parse_config(json_path: str) -> TokenDroppingConfig:
    parser = HfArgumentParser((TokenDroppingConfig,))
    config = parser.parse_json_file(json_file=os.path.abspath(json_path))[0]
    return config


def patch_config(config):
    original_fn = config.__class__.to_dict

    def new_fn(self):
        output = original_fn(self)
        for k in ['token_dropping', 'token_dropping_args']:
            if k in output:
                del output[k]
        return output

    config.__class__.to_dict = new_fn
    return config