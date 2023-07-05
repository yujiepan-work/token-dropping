from collections import defaultdict
from transformers.trainer import Trainer
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

temp_storage = defaultdict(list)


def clear_temp_storage():
    temp_storage.clear()
    return temp_storage


def do_final_clean(trainer: Trainer):
    output_dir = trainer.args.output_dir
    # torch.save(temp_storage, Path(output_dir, 'seq_len.pth'))


class PreloadedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self._data = {}
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            self._data[i] = sample
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._data[index]

