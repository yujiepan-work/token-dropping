#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint:ignore=logging-fstring-interpolation

import logging as log
import sys
import tempfile
from pathlib import Path
import time
from time import perf_counter
import statistics
from tqdm import tqdm
from local_config import LOG_PATH, NODE

import datasets
from openvino.runtime import AsyncInferQueue, Core, PartialShape, get_version
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

REPO_ROOT = Path(__file__).parent.parent
NUM_SAMPLES = 2500

from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize, ToTensor)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def get_val_transform(image_processor):
    size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    return _val_transforms


image_processor = AutoImageProcessor.from_pretrained(Path('~/vit-base-patch16-224-food101').expanduser().absolute())
_val_transforms = get_val_transform(image_processor)


def val_transforms(example_batch):
    """Apply _val_transforms across a batch."""
    example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
    d = {}
    d['pixel_values'] = example_batch['pixel_values']
    return d
    return example_batch


def get_raw_dataset():
    dataset = datasets.load_dataset((REPO_ROOT / 'examples' / 'image-classification' / 'food101.py').absolute().as_posix())
    dataset = dataset["validation"].shuffle(seed=42).select(range(NUM_SAMPLES))
    dataset.set_transform(val_transforms)
    return dataset


raw_dataset = get_raw_dataset()


def job(i):
    return raw_dataset[i]['pixel_values'].unsqueeze(0).numpy()


def get_dataset():
    import torch
    cache = Path(f'/dev/shm/food101-{NUM_SAMPLES}.pt')
    # if cache.exists():
    #     return torch.load(cache)
    import multiprocessing
    from multiprocessing.pool import ThreadPool
    # pool = ThreadPool(10)
    # results = pool.map(job, range(NUM_SAMPLES))
    results = [job(i) for i in tqdm(range(NUM_SAMPLES))]
    # torch.save(results, cache)
    return results


dataset = get_dataset()
print(dataset[0].shape)
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
log.info('OpenVINO:')
log.info(f"{'Build ':.<39} {get_version()}")

def benchmark_throughput(onnx_path):
    core = Core()
    model = core.read_model(onnx_path)

    # Optimize for throughput. Best throughput can be reached by
    # running multiple openvino.runtime.InferRequest instances asyncronously
    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}
    # Pick a device by replacing CPU, for example MULTI:CPU(4),GPU(8).
    # It is possible to set CUMULATIVE_THROUGHPUT as PERFORMANCE_HINT for AUTO device
    compiled_model = core.compile_model(model, 'CPU', tput)
    # AsyncInferQueue creates optimal number of InferRequest instances
    ireqs = AsyncInferQueue(compiled_model)

    # Warm up
    encoded_warm_up = dataset[0]
    for i in range(10):
        for _ in ireqs:
            ireqs.start_async(encoded_warm_up)
        ireqs.wait_all()

    # Benchmark
    start = perf_counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        ireqs.start_async(sample)
    ireqs.wait_all()
    end = perf_counter()
    duration = end - start
    fps = len(dataset) / duration
    # log.info(f'Duration: {duration:.2f} seconds')
    # log.info(f'Throughput: {fps:.2f} FPS')

    from datetime import datetime
    now = datetime.now() # current date and time
    return dict(onnx=onnx_path, duration=duration, throughput=fps,
                datetime=now.strftime("%m-%d-%Y-%H-%M-%S"), num_samples=len(dataset), ov_version=get_version())

def main():
    items = []
    # for folder in ['RouterOursNewTokenMeanQ,freeze-True,ratio2drop,1e-4,256-16', 'RouterTranskimmer,freeze-False,1e-4', 'RouterToMeGlueUseKey,freeze-True', 'RouterBaseline897']:
    for folder in ['RouterTranskimmer,freeze-False,1e-4']:
    # for folder in ['RouterOursNoNew,freeze-True,ratio2drop']:
        for onnx_path in tqdm(sorted(Path(LOG_PATH / 'train-food101/seed42').glob(f'{folder}/**/export_onnx/model.onnx'))):
            onnx_path = onnx_path.absolute().as_posix()
            item = benchmark_throughput(onnx_path)
            with open(Path(onnx_path).parent / f'benchmark_openvino_{NODE}.json', 'w') as f:
                import json
                json.dump(item, f, indent=2)
            print(item)
            # items.append(item)
            # from pandas import DataFrame
            # df = DataFrame(items)
            # df.to_csv('benchmark_ov_food101_v2.1.csv')

main()