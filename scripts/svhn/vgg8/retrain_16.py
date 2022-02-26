'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-16 13:20:38
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-26 02:24:42
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

dataset = "svhn"
model = "vgg8"
root = f"log/{dataset}/{model}/retrain_16_svhn"
script = "retrain.py"
config_file = f"configs/{dataset}/{model}/retrain_16.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    min_block, max_block, lower_bound, upper_bound, ckpt, id = args
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
            # f"--super_layer.name=ps_dc_cr",
            f"--super_layer.name=ps_dc_cr_adept",
            f"--run.train_arch_epoch=10",
            f"--checkpoint.model_comment=adept_blkmin-{min_block}_blkmax-{max_block}_bound-{upper_bound}",
            f"--checkpoint.supermesh_checkpoint={ckpt}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks = []
    r = 0.8
    tasks = [
        (
            2,
            16,
            840 * r,
            840,
            "./checkpoint/mnist/cnn/super_train_16/SuperOCNN_adept_blkmin-2_blkmax-16_bound-840_acc-84.83_epoch-87.pt",
            2,
        ),
        (
            2,
            16,
            1320 * r,
            1320,
            "./checkpoint/mnist/cnn/super_train_16/SuperOCNN_adept_blkmin-2_blkmax-16_bound-1320_acc-85.97_epoch-84.pt",
            2,
        ),
    ]  # 1e-7, 1.3 amf pdk

    with Pool(2) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
