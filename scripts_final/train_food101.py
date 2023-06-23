import os
from pathlib import Path

from toytools.batchrun import Launcher, Task, avail_cuda_list
from toytools.iterext import product
from toytools.misc import json_dump, today_cipher
from toytools.snapshot.log_python_env import log_python_env_status
from typing import List

from token_dropping.config import TokenDroppingConfig

root = Path('.').absolute().parent
config_path = Path(f'/tmp/{os.getlogin()}/token_dropping_config')
config_path.mkdir(exist_ok=True, parents=True)

env = os.environ.copy()
env["WANDB_DISABLED"] = "true"
env["WANDB_PROJECT"] = "debug"
env["WANDB_WATCH"] = "false"
env['HF_EVALUATE_OFFLINE'] = '1'


def prepare_fn(token_dropping_json_path, output_folder):
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    log_python_env_status(output_folder)
    os.system(f'cp {token_dropping_json_path} {output_folder}')


def gen_gradual_strategy(r):
    res = []
    l = 197
    for i in range(12):
        res.append(f'{i}-{l-r}')
        l -= r
    return '_'.join(res)


def gen_all(preserve=100):
    res = []
    for l in range(12):
        res.append(f'{l}-{preserve}')
    return '_'.join(res)


def gen_3drop(r):
    res = []
    res.append(f'2-{197-r}')
    res.append(f'5-{197-r * 2}')
    res.append(f'8-{197-r * 3}')
    return '_'.join(res)


def gen_3drop_by_ratio(r):
    res = []
    res.append(f'2-{int(197*r)}')
    res.append(f'5-{int(197 * r * r)}')
    res.append(f'5-{int(197 * r * r * r)}')
    return '_'.join(res)


def new_cfg_tasks(
    strategy,
    version,
    mask_loss_alpha=(0.0,),
    freeze_bert=(False,),
):
    cfgs = product(
        lr=[1e-4],
        warmup_ratio=[0.05],
        strategy=strategy,
        version=version,
        attention_unit=[256],
        attention_head_dim=[32],
        mask_loss_alpha=mask_loss_alpha,
        freeze_bert=freeze_bert,
        seed=[42, 1],
    )
    return get_cfg_tasks(cfgs)


def get_cfg_tasks(cfgs) -> List[Task]:
    tasks = []
    for cfg in list(cfgs):
        folder = Path(
            env.get('TOKEN_DROPPING_LOGDIR', root), 'logs-final/', 'train-food101', f'seed{cfg.seed}',
            f'{cfg.version},freeze-{cfg.freeze_bert}',
            f'{cfg.version},{cfg.freeze_bert},{cfg.mask_loss_alpha},{cfg.strategy},{cfg.lr}'
        )
        token_dropping_json = json_dump(
            dict(
                token_pruning_strategy=cfg.strategy,
                router_version=cfg.version,
                export_onnx=False,
                freeze_model=cfg.freeze_bert,
                attention_unit=cfg.attention_unit,
                attention_head_dim=cfg.attention_head_dim,
                num_new_token=1,
                mask_loss_alpha=cfg.mask_loss_alpha,
                router_before_ffn=False if cfg.version in ['RouterOursSoftmaxGatingNoNewToken', 'RouterTranskimmer'] else True,
                reinit_router_weights=True,
            ),
            temp_folder=config_path,
        )
        eval_bs = 128 if cfg.version not in ['RouterToMeGlueUseKey'] else 1

        if not cfg.freeze_bert:
            do_train = True
        else:
            # freeze bert
            if cfg.version in ['RouterOursNoNew', 'RouterToMeGlueUseKey']:
                do_train = False
            else:
                do_train = True

        task = Task(
            cmd=["""python run_image_classification.py""",
                 f"--token_dropping_json_path {token_dropping_json} ",
                 """ --model_name_or_path""",
                 "eslamxm/vit-base-food101",
                 ("--do_train" if do_train else ''),
                 f""" \
                --dataset_name food101.py \
                --remove_unused_columns False \
                --do_eval \
                --optim adamw_torch \
                --warmup_ratio {cfg.warmup_ratio} \
                --learning_rate {cfg.lr} \
                --num_train_epochs 3 \
                --dataloader_num_workers 10 \
                --per_device_train_batch_size 64 \
                --gradient_accumulation_steps 2 \
                --per_device_eval_batch_size {eval_bs} \
                --output_dir {folder} --fp16 --overwrite_output_dir --save_total_limit 1 \
                --evaluation_strategy steps \
                --eval_steps 296 \
                --save_steps 296 \
                --load_best_model_at_end False \
                --metric_for_best_model accuracy \
                --seed {cfg.seed} --data_seed {cfg.seed} \
                """],
            cwd=root / 'examples' / 'image-classification',
            io_folder=folder,
            identifier=folder.name,
            env=env,
            cuda_quantity=1,
            prepare_fn=prepare_fn,
            prepare_fn_args=(token_dropping_json, folder),
        )
        tasks.append(task)

    return tasks


# Transkimmer
transkimmer_cfgs = new_cfg_tasks(
    strategy=[gen_all(999)],
    version=['RouterTranskimmer'],
    mask_loss_alpha=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0],
    freeze_bert=[False],
)

# pretrained_attention_by_constant_r
ours_nonewtoken_by_constant_r = new_cfg_tasks(
    strategy=[gen_3drop(x) for x in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62]],
    version=['RouterOursNoNew'],
    mask_loss_alpha=[0.],
    freeze_bert=[True, False],
)

# pretrained_attention_by_ratio
ours_nonewtoken_by_ratio = new_cfg_tasks(
    strategy=[gen_3drop_by_ratio(r) for r in [1.0, 0.8, 0.85, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3]],
    version=['RouterOursNoNew'],
    mask_loss_alpha=[0.],
    freeze_bert=[True, False],
)

# tome
tome_cfgs = new_cfg_tasks(
    strategy=[gen_gradual_strategy(r) for r in [3, 6, 9, 12, 13, 15, 16]],
    version=['RouterToMeGlueUseKey'],
    mask_loss_alpha=[0.],
    freeze_bert=[True],
)

# pretrained_attention_by_constant_r
ours_1newtoken_by_constant_r = new_cfg_tasks(
    strategy=[gen_3drop(x) for x in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62]],
    version=['RouterOursNewToken'],
    mask_loss_alpha=[0.],
    freeze_bert=[True, False]
)

# pretrained_attention_+new_token_by_ratio
ours_1newtoken_by_ratio = new_cfg_tasks(
    strategy=[gen_3drop_by_ratio(r) for r in [1.0, 0.8, 0.85, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3]],
    version=['RouterOursNewToken'],
    mask_loss_alpha=[0.],
    freeze_bert=[True, False]
)

all_tasks = transkimmer_cfgs + ours_nonewtoken_by_constant_r + ours_nonewtoken_by_ratio + tome_cfgs + ours_1newtoken_by_ratio + ours_1newtoken_by_ratio


Launcher(avail_cuda_list(16000)).run(all_tasks, add_timestamp_to_log=False)
