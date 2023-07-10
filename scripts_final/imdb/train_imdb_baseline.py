import os
from pathlib import Path
import tempfile

import toytools
from toytools.batchrun import Launcher, Task, avail_cuda_list
from toytools.iterext import product
from toytools.misc import today_cipher, json_dump
from toytools.snapshot.log_python_env import log_python_env_status
from token_dropping.config import TokenDroppingConfig

root = Path('.').absolute().parent.parent
config_path = Path('/tmp/yujiepan/token_dropping_config')
config_path.mkdir(exist_ok=True, parents=True)
from local_config import LOG_PATH, USER_NAME, NODE, IS_PRC_MACHINE, BASELINE

env = os.environ.copy()
env["WANDB_DISABLED"] = "true"
env["WANDB_PROJECT"] = "debug"
env["WANDB_WATCH"] = "false"
if IS_PRC_MACHINE:
    env['https_proxy'] = 'http://child-prc.intel.com:912'
    env['http_proxy'] = 'http://child-prc.intel.com:912'
env['HF_EVALUATE_OFFLINE'] = '1'


def prepare_fn(output_folder):
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    log_python_env_status(output_folder)


ours_cfgs = product(
    lr=[5e-5, 2e-5],
    warmup_ratio=[0.1, 0.5],
)

# all_cfgs = list(ours_cfgs) + list(transkimmer_cfgs)
all_cfgs = list(ours_cfgs)

tasks = []
for cfg in list(all_cfgs)[:]:
    folder = Path(LOG_PATH, f'train-imdb-baseline-new/', f'0703-lr{cfg.lr}_warm{cfg.warmup_ratio}_epoch6_bs8')
    task = Task(
        cmd=["""python run_glue_ori.py""",
             f"""--model_name_or_path {BASELINE} """,
             '--dataset_name imdb.py ',
             "--pad_to_max_length False",
             "--do_train",
             f"""--do_eval \
            --optim adamw_torch \
            --max_seq_length 384 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --learning_rate {cfg.lr} \
            --warmup_ratio {cfg.warmup_ratio} \
            --num_train_epochs 6 \
            --output_dir {folder} --fp16 --overwrite_output_dir --save_total_limit 1 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --eval_steps 391 \
            --save_steps 391 \
            --load_best_model_at_end True \
            --metric_for_best_model accuracy \
            """],
        cwd=root / 'examples' / 'text-classification',
        io_folder=folder,
        identifier=folder.name,
        env=env,
        cuda_quantity=1,
        prepare_fn=prepare_fn,
        prepare_fn_args=(folder,),
    )
    tasks.append(task)

Launcher(avail_cuda_list(17000)).run(tasks)
