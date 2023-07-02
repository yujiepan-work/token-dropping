import datasets
d = datasets.load_dataset('imdb')
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('bert-base-uncased')
i = 0
long_samples = []
from tqdm import tqdm
for s in tqdm(d['test']):
    # print(s)
    r = t(s['text'])
    # print(r)
    if (sum(r['attention_mask'])) >= 384:
        i += 1
        long_samples.append(s)

print(i)

from pathlib import Path
long_dataset = datasets.Dataset.from_list(long_samples)
long_dataset.save_to_disk(Path('~/imdb-long').expanduser())
load = datasets.load_from_disk(Path('~/imdb-long').expanduser())
