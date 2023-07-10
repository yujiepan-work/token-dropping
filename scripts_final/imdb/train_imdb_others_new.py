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

def gen_4drop(*p_l):
    res = []
    for l, p in zip([2, 4, 7, 9], p_l):
        res.append(f'{l}-{p}')
    return '_'.join(res)

def gen_5drop(*p_l):
    res = []
    for l, p in zip([1,3,5,7,9], p_l):
        res.append(f'{l}-{p}')
    return '_'.join(res)

def gen_2drop(*p_l):
    res = []
    for l, p in zip([3, 7], p_l):
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


ours_cfgs = product(
    lr=[1e-4],
    warmup_ratio=[0.1],
    strategy=[
        # gen_3drop(115, 34, 10),
        gen_3drop(153, 61, 24),
        gen_3drop(192, 96, 48),
        gen_3drop(211, 116, 63),
        gen_3drop(230, 138, 82),
        gen_3drop(249, 162, 105),
        gen_3drop(268, 188, 131),
        gen_3drop(307, 245, 196),
        gen_3drop(326, 277, 235),
        # gen_3drop(345, 311, 279),
        gen_3drop(364, 346, 329),
    ],
    version=[
        # 'RouterOursNoNew',
        'RouterOursNewToken'
    ],
    tome_last_len=[-1],
    mask_loss_alpha=[0.],
    ours_force_reduction_ratio=[-1.],
)

ours_r_cfgs = product(
    lr=[1e-4],
    warmup_ratio=[0.0],
    strategy=[
        # gen_3drop(999, 999, 999),
        # gen_2drop(999, 999),
        gen_4drop(999,999,999,999),
        # gen_5drop(999,999,999,999,999),
    ],
    title=['RouterOursNewRatio-4drop-train-reduce-more'],
    version=[
        'RouterOursNewTokenReductionRatio'
    ],
    tome_last_len=[-1],
    mask_loss_alpha=[0.],
    attention_head_dim=[32],
    attention_unit=[256],
    ours_force_reduction_ratio=[0.97, 0.98, 0.99],
    ours_force_reduction_ratio_for_train=[0.90],
    # ours_force_reduction_ratio=[0.21],
)


transkimmer_cfgs = product(
    lr=[1e-4],
    warmup_ratio=[0.0],
    strategy=[
        gen_all(999),
    ],
    title=['RouterTranskimmer1e-4'],
    version=[
        'RouterTranskimmer'
    ],
    mask_loss_alpha=[0.005, 0.015, 0.025, 0.035],
    tome_last_len=[-1],
    attention_head_dim=[64],
    attention_unit=[256],
    ours_force_reduction_ratio=[-1.],
    ours_force_reduction_ratio_for_train=[1.],
)


# all_cfgs = list(ours_cfgs) + list(transkimmer_cfgs)
# all_cfgs = list(transkimmer_cfgs)
all_cfgs = list(transkimmer_cfgs)
# all_cfgs = list(ours_r_cfgs)

freeze_model_dict = {
    'RouterOursNoNew': True,
    'RouterOursNewToken': True,
    'RouterOursNewTokenReductionRatio': True,
    'RouterTranskimmer': False,
}

router_before_ffn_dict = {
    'RouterOursNoNew': True,
    'RouterOursNewToken': True,
    'RouterOursNewTokenReductionRatio': True,
    'RouterTranskimmer': False,
}
do_train_dict = {
    'RouterOursNoNew': False,
    'RouterOursNewToken': True,
    'RouterOursNewTokenReductionRatio': True,
    'RouterTranskimmer': True,
}

tasks = []
for cfg in list(all_cfgs)[:]:
    folder = Path(LOG_PATH, f'train-imdb/seed42new/{cfg.title}/', f'{cfg.version},{cfg.mask_loss_alpha},{cfg.ours_force_reduction_ratio},{cfg.strategy},lr{cfg.lr}_warm{cfg.warmup_ratio}_epoch3_head{cfg.attention_unit}+{cfg.attention_head_dim}')
    token_dropping_json = json_dump(
        dict(
            token_pruning_strategy=cfg.strategy,
            router_version=cfg.version,
            export_onnx=False,
            freeze_model=freeze_model_dict[cfg.version],
            router_before_ffn=router_before_ffn_dict[cfg.version],
            tome_last_len=cfg.tome_last_len,
            mask_loss_alpha=cfg.mask_loss_alpha,
            reinit_router_weights=True,
            attention_head_dim=cfg.attention_head_dim,
            attention_unit=cfg.attention_unit,
            ours_force_reduction_ratio=cfg.ours_force_reduction_ratio,
            ours_force_reduction_ratio_training=cfg.ours_force_reduction_ratio * cfg.ours_force_reduction_ratio_for_train,
        ),
        temp_folder=config_path,
    )
    load_best = True if 'oursnew' in cfg.version.lower() else False
    task = Task(
        cmd=["""python run_glue_new.py""",
             f"--token_dropping_json_path {token_dropping_json} ",
             f"""--model_name_or_path {BASELINE} """,
             '--dataset_name imdb.py ',
             "--pad_to_max_length False",
             "--do_train" if do_train_dict[cfg.version] else '',
             f"""--do_eval \
            --optim adamw_torch \
            --max_seq_length 384 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate {cfg.lr} \
            --warmup_ratio {cfg.warmup_ratio} \
            --num_train_epochs 3 \
            --output_dir {folder} --fp16 --overwrite_output_dir --save_total_limit 1 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --eval_steps 391 \
            --save_steps 391 \
            --load_best_model_at_end {load_best} \
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

Launcher(avail_cuda_list(17000)).run(tasks)
