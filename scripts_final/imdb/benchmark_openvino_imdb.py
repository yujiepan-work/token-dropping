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
import torch

import datasets
from openvino.runtime import AsyncInferQueue, Core, PartialShape, get_version
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

REPO_ROOT = Path(__file__).parent.parent
NUM_SAMPLES = 500


def get_dataset():
    t = AutoTokenizer.from_pretrained(Path('~/bert-base-uncased-imdb').expanduser().as_posix())
    long_imdb_dataset = datasets.load_from_disk(Path('~/imdb-long').expanduser().as_posix())
    result = []
    for i in tqdm(range(NUM_SAMPLES)):
        sample = long_imdb_dataset[i]
        ids = t(sample['text'], max_length=384)['input_ids']
        ids = torch.tensor(ids).long().reshape(1, -1).numpy()
        result.append(ids)
    return result


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
    now = datetime.now()  # current date and time
    return dict(onnx=onnx_path, duration=duration, throughput=fps,
                datetime=now.strftime("%m-%d-%Y-%H-%M-%S"), num_samples=len(dataset), ov_version=get_version())


def main():
    onnx_paths = []
    json_filename = f'benchmark_openvino_{NODE}.json'
    for folder in ['RouterTranskimmer']:
        for onnx_path in sorted(Path(LOG_PATH / 'train-imdb/seed42').glob(f'{folder}/**/export_onnx/model.onnx'))[::-1]:
            if Path(Path(onnx_path).parent / json_filename).exists():
                continue
            onnx_paths.append(onnx_path)

    for onnx_path in tqdm(onnx_paths):
        onnx_path = onnx_path.absolute().as_posix()
        item = benchmark_throughput(onnx_path)
        with open(Path(onnx_path).parent / json_filename, 'w') as f:
            import json
            json.dump(item, f, indent=2)
        print(item)


main()
