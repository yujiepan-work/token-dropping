import itertools
import re
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class BenchmarkResult:
    stdout: str = ''
    stderr: str = ''
    avg_latency: float = 0.
    throughput: float = 0.


def run_benchmark(cmd):
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        p.wait()
        stdout = p.stdout.read().decode()
        stderr = p.stderr.read().decode()
        # print(stdout)
        # print(stderr)
        stdout = stdout.strip()
        avg_line = filter(None, stdout.split('\n')[-4].split())
        throughput_line = filter(None, stdout.split('\n')[-1].split())
        avg_line = list(avg_line)
        throughput_line = list(throughput_line)
        assert 'Average:' in avg_line and 'Throughput:' in throughput_line

        return BenchmarkResult(
            stdout=stdout,
            stderr=stderr,
            avg_latency=float(list(avg_line)[-2]),
            throughput=float(list(throughput_line)[-2]),
        )


results = []
from local_config import LOG_PATH, NODE
from tqdm import tqdm

logfile_name = f'benchmark_app_stdout_try2'

FOLDERS = [
    'RouterOursNewTokenMeanQ,freeze-True,ratio2drop,1e-4,256-16',
    'RouterToMeGlueUseKey,freeze-True',
    'RouterOursNoNew,freeze-True,ratio2drop',
    'RouterBaseline897'
]

for folder in FOLDERS:
    for model in tqdm(sorted(LOG_PATH.glob(f'train-food101*/seed42/{folder}/**/export_onnx/model.onnx'))):
        model = model.absolute()
        if (model.parent / (logfile_name + '.log')).exists():
            continue
        print('***' * 3, model)
        # result = run_benchmark(cmd=f'benchmark_app -m {model} -hint latency -t 30')
        cmd = f'benchmark_app -m {model} -hint throughput -t 25'
        result = run_benchmark(cmd=cmd)
        results.append(
            dict(
                model=model,
                latency=result.avg_latency,
                throughput=result.throughput,
            )
        )
        # with open(model.parent / 'benchmark_app_stderr.log', 'w') as f:
        #     f.write(result.stderr)
        with open(model.parent / f'{logfile_name}.log', 'w') as f:
            f.write(result.stdout)
        with open(model.parent / f'{logfile_name}.json', 'w') as f:
            import json
            json.dump(dict(
                node=NODE,
                latency=result.avg_latency,
                throughput=result.throughput,
                cmd=cmd,
                model=model.absolute().as_posix(),
            ), f, indent=2)
        df = pd.DataFrame(results)
        # df.to_csv('./dgx3.csv')
