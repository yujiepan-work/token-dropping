import os
from pathlib import Path
import tempfile

import toytools
from toytools.batchrun import Launcher, Task, avail_cuda_list
from toytools.iterext import product
from toytools.misc import today_cipher, json_dump
from toytools.snapshot.log_python_env import log_python_env_status
from token_dropping.config import TokenDroppingConfig

root = Path('.').parent.absolute().parent.parent
config_path = Path('/tmp/yujiepan/token_dropping_config')
config_path.mkdir(exist_ok=True, parents=True)

from local_config import LOG_PATH, USER_NAME, IS_PRC_MACHINE, NODE

env = os.environ.copy()
env["WANDB_DISABLED"] = "true"
env["WANDB_PROJECT"] = "debug"
env["WANDB_WATCH"] = "false"
if IS_PRC_MACHINE:
    env['https_proxy'] = 'http://child-prc.intel.com:912'
    env['http_proxy'] = 'http://child-prc.intel.com:912'
env['HF_EVALUATE_OFFLINE'] = '1'


def prepare_fn(token_dropping_json_path, output_folder):
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    log_python_env_status(output_folder)
    os.system(f'cp {token_dropping_json_path} {output_folder}')


def gen_strategy(r):
    res = []
    l = 197
    for i in range(12):
        res.append(f'{i}-{l-r}')
        l -= r
    return '_'.join(res)


def gen_3drop(*p_l):
    res = []
    for l, p in zip([2, 5, 8], p_l):
        res.append(f'{l}-{p}')
    return '_'.join(res)


def gen_all(preserve=100):
    res = []
    for l in range(12):
        res.append(f'{l}-{preserve}')
    return '_'.join(res)


def gen_nlp_tome():
    res = []
    for l in range(1, 12):
        res.append(f'{l}-999')
    return '_'.join(res)


cfgs = product(
    lr=[1e-4],
    warmup_ratio=[0.0],
    strategy=[
        gen_nlp_tome(),
    ],
    version=[
        'RouterToMeGlueUseKey'
    ],
    # tome_last_len=[5, 10, 15, 20, 30, 40, 60, 80],
    tome_last_len=[100, 150, 200, 250, 300, 350],
    mask_loss_alpha=[0.],
)

tasks = []
for cfg in list(cfgs):
    folder = Path(LOG_PATH, 'train-imdb/seed42', f'{cfg.version}', f'{cfg.version},TM{cfg.tome_last_len}')
    token_dropping_json = json_dump(
        dict(
            token_pruning_strategy=cfg.strategy,
            router_version=cfg.version,
            export_onnx=False,
            freeze_model=True,
            router_before_ffn=True,
            tome_last_len=cfg.tome_last_len,
            mask_loss_alpha=cfg.mask_loss_alpha,
            reinit_router_weights=False,
            is_benchmark_mode=True,
        ),
        temp_folder=config_path,
    )
    task = Task(
        cmd=["""python run_glue.py""",
             f"--token_dropping_json_path {token_dropping_json} ",
             """--model_name_or_path ~/bert-base-uncased-imdb """,
             '--dataset_name imdb.py ',
             "--pad_to_max_length False",
             f"""--do_eval \
            --optim adamw_torch \
            --max_seq_length 384 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 1 \
            --learning_rate {cfg.lr} \
            --warmup_ratio {cfg.warmup_ratio} \
            --num_train_epochs 3 \
            --output_dir {folder} --overwrite_output_dir --save_total_limit 1 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --eval_steps 391 \
            --save_steps 391 \
            --load_best_model_at_end False \
            --metric_for_best_model accuracy \
            """],
        cwd=root / 'examples' / 'text-classification',
        io_folder=folder,
        identifier=folder.name,
        env=env,
        cuda_quantity=1,
        prepare_fn=prepare_fn,
        prepare_fn_args=(token_dropping_json, folder),
    )
    tasks.append(task)

Launcher(avail_cuda_list(20000)).run(tasks)