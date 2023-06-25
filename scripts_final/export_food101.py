import os
from pathlib import Path
import tempfile

import toytools
from toytools.batchrun import Launcher, Task, avail_cuda_list
from toytools.iterext import product
from toytools.misc import today_cipher, json_dump
from toytools.snapshot.log_python_env import log_python_env_status
from token_dropping.config import TokenDroppingConfig
from local_config import LOG_PATH, IS_PRC_MACHINE

root = Path('.').absolute().parent
config_path = Path('/tmp/yujiepan/token_dropping_config')
config_path.mkdir(exist_ok=True, parents=True)

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
    # log_python_env_status(output_folder)
    os.system(f'cp {token_dropping_json_path} {output_folder}')


def get_json(folder) -> dict:
    for js in Path(folder).glob('*.json'):
        if len(js.name) == 16 + 5 and '_' not in js.name:
            with open(js, 'r') as f:
                import json
                return json.load(f)


tasks = []
for model in list(Path(LOG_PATH, 'train-food101/seed42/').glob('R*/R*')):
    folder = model / 'export_onnx'
    token_dropping_content: dict = get_json(model)
    token_dropping_content['reinit_router_weights'] = False
    token_dropping_content['is_benchmark_mode'] = True
    token_dropping_content['export_onnx'] = True
    token_dropping_json = json_dump(token_dropping_content, temp_folder=config_path)
    model_path = str(model.absolute().as_posix()) if Path(model, 'pytorch_model.bin').exists() else '~/vit-base-patch16-224-food101'
    task = Task(
        cmd=[
            """python run_image_classification.py""",
            f"--token_dropping_json_path {token_dropping_json} ",
            """ --model_name_or_path""",
            model_path,
            '--max_eval_samples 10 ',
            f"""--dataset_name food101.py \
            --remove_unused_columns False \
            --do_eval \
            --dataloader_num_workers 4 \
            --per_device_eval_batch_size 1 \
            --output_dir {folder} --overwrite_output_dir \
            --seed 42 --data_seed 42
            """
        ],
        cwd=root / 'examples' / 'image-classification',
        io_folder=folder,
        identifier=model.name,
        env=env,
        cuda_quantity=1,
        prepare_fn=prepare_fn,
        prepare_fn_args=(token_dropping_json, folder),
    )
    tasks.append(task)

# Launcher([0,1,2]).run(tasks)
Launcher(avail_cuda_list(10000)).run(tasks)
