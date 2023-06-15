from dataclasses import dataclass, field, asdict
from typing import Optional, Union
import os
import json

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    logging,
)
from transformers.utils import (
    logging,
)
# pylint: disable=missing-function-docstring

logger = logging.get_logger(__name__)

@dataclass
class TokenDroppingConfig:
    """
    Config for token dropping.
    """
    export_onnx: bool = field(
        default=False,
        metadata={"help": "None"},
    )
    token_pruning_strategy: Union[str, dict] = field(
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
    tome_last_len: int = field(
        default=-1,
    )
    tome_force_r: int = field(
        default=-1,
    )
    num_new_token: int = field(
        default=1
    )
    attention_unit: int = field(
        default=256,
    )
    attention_head_dim: int = field(
        default=64,
    )
    mask_loss_alpha: float = field(
        default=0.0,
    )
    router_before_ffn: bool = field(
        default=True
    )
    reinit_router_weights: bool = field(
        default=False
    )

    def __post_init__(self):
        strategy = self.token_pruning_strategy.replace('_', ',').replace('-', ':')
        self.token_pruning_strategy = eval(f'{{{strategy}}}')  # pylint: disable=eval-used
        logger.warning('Token dropping args: \n%s', json.dumps(asdict(self), indent=2))


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