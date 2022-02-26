'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-10 03:38:08
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-26 02:32:16
'''

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

dataset = "mnist"
model = "cnn"
root = f"log/{dataset}/{model}/super_train"
script = "train.py"
config_file = f"configs/{dataset}/{model}/super_train_8.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    min_block, max_block, lower_bound, upper_bound, id = args
    with open(
        os.path.join(root, f"adept_blkmin-{min_block}_blkmax-{max_block}_bound-{upper_bound}_run-{id}.log"),
        "w",
    ) as wfid:
        exp = [
            f"--super_layer.arch.n_blocks={max_block}",
            f"--super_layer.arch.n_front_share_blocks={min_block}",
            f"--super_layer.arch.device_cost.area_upper_bound={upper_bound}",
            f"--super_layer.arch.device_cost.area_lower_bound={lower_bound}",
            f"--run.random_state={41+id}",
            f"--run.force_perm_legal_epoch=40",
            # f"--super_layer.name=ps_dc_cr",
            f"--super_layer.name=ps_dc_cr_adept",
            f"--run.train_arch_epoch=10",
            f"--checkpoint.model_comment=adept_blkmin-{min_block}_blkmax-{max_block}_bound-{upper_bound}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    r = 0.8

    tasks = [
        (2, 16, 300*r, 300, 1),
        (2, 16, 420*r, 420, 1),
        (2, 16, 540*r, 540, 1),
        (2, 16, 660*r, 660, 1),
        (2, 16, 780*r, 780, 1),
    ]

    with Pool(5) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
