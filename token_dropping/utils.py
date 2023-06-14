from collections import defaultdict
from transformers.trainer import Trainer
import torch
from pathlib import Path

temp_storage = defaultdict(list)


def clear_temp_storage():
    temp_storage.clear()
    return temp_storage


def do_final_clean(trainer: Trainer):
    output_dir = trainer.args.output_dir
    torch.save(temp_storage, Path(output_dir, 'seq_len.pth'))