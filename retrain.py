"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-13 03:03:14
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-26 02:21:49
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Iterable

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder
from core.models.layers.super_utils import ArchSampler, get_named_sample_arch
from core.models.layers.utils import clip_grad_value_


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
    teacher: nn.Module = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    phase_noise_std = float(getattr(configs.noise, "phase_noise_std", 0))
    dc_noise_std = float(getattr(configs.noise, "dc_noise_std", 0))
    model.set_phase_noise(phase_noise_std)
    model.set_dc_noise(dc_noise_std)

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        def _get_loss(output, target):
            if teacher:
                with torch.no_grad():
                    teacher_score = teacher(data).detach()
                loss = criterion(output, teacher_score)
            else:
                loss = criterion(output, target)
            return loss

        # random subnet
        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        loss = _get_loss(output, target)
        class_loss = loss

        loss.backward()
        optimizer.step()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info(
                "Train Epoch: {} (phase noise std={}) [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}".format(
                    epoch,
                    phase_noise_std,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    class_loss.data.item(),
                )
            )
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.item(), "lr": get_learning_rate(optimizer)}, step=epoch)


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )
    mlflow.log_metrics({"val_acc": accuracy.data.item(), "val_loss": val_loss}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    mlflow.log_metrics({"test_acc": accuracy.data.item(), "test_loss": val_loss}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    model = builder.make_model(
        device, int(configs.run.random_state) if int(configs.run.deterministic) else None
    )

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    if validation_loader is None:
        validation_loader = test_loader
    optimizer = builder.make_weight_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = (
        f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    )

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "inbit": configs.quantize.input_bit,
            "wbit": configs.quantize.weight_bit,
            "init_lr": configs.weight_optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        solution, score = None, None
        if int(configs.checkpoint.resume):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

        if configs.teacher.name and configs.teacher.checkpoint:
            lg.info(f"Build teacher model {configs.teacher.name}")
            teacher = builder.make_model(
                device,
                int(configs.run.random_state) if int(configs.run.deterministic) else None,
                model_name=configs.teacher.name,
            )
            load_model(teacher, path=configs.teacher.checkpoint)
            teacher_criterion = builder.make_criterion(name="ce").to(device)
            teacher.eval()
            lg.info(f"Validate teacher model {configs.teacher.name}")
            validate(teacher, validation_loader, -1, teacher_criterion, [], [], False, device)
        else:
            teacher = None

        arch_sampler = ArchSampler(
            model=model,
            strategy=configs.super_layer.sampler.strategy.dict(),
            n_layers_per_block=configs.super_layer.arch.n_layers_per_block,
        )
        arch_sampler.set_total_steps(configs.run.n_epochs * len(train_loader))
        sample_arch = get_named_sample_arch(model.arch_space, name="largest")
        model.set_sample_arch(sample_arch)

        if (
            configs.checkpoint.supermesh_checkpoint is not None
            and len(configs.checkpoint.supermesh_checkpoint) > 0
        ):
            model.load_arch_solution(configs.checkpoint.supermesh_checkpoint)

        ps_weight = float(getattr(configs.super_layer.arch.device_cost, "ps_weight", 1))
        dc_weight = float(getattr(configs.super_layer.arch.device_cost, "dc_weight", 1))
        cr_weight = float(getattr(configs.super_layer.arch.device_cost, "cr_weight", 1))
        area_upper_bound = float(getattr(configs.super_layer.arch.device_cost, "area_upper_bound", 100))
        area_lower_bound = float(getattr(configs.super_layer.arch.device_cost, "area_lower_bound", 80))
        model.sample_submesh(
            n_samples=int(getattr(configs.run, "n_samples", 100)),
            ps_weight=ps_weight,
            dc_weight=dc_weight,
            cr_weight=cr_weight,
            upper_bound=area_upper_bound,
            lower_bound=area_lower_bound,
        )

        if bool(getattr(configs.run, "test_only", False)):
            lossv_cur, accv_cur = [], []
            for _ in range(20):
                test(
                    model,
                    test_loader,
                    epoch,
                    teacher_criterion if teacher else criterion,
                    lossv_cur,
                    accv_cur,
                    device=device,
                )
            avg_acc, std_acc = np.mean(accv_cur), np.std(accv_cur)
            accv.append(avg_acc)
            lg.info(f"Test: average acc: {avg_acc:.4f}, std acc: {std_acc:.6f}")
            exit(0)

        for epoch in range(1, int(configs.run.n_epochs) + 1):

            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                device,
                teacher=teacher,
            )

            lg.info(f"Validating...")
            lossv_cur, accv_cur = [], []
            for _ in range(5):
                validate(
                    model,
                    validation_loader,
                    epoch,
                    teacher_criterion if teacher else criterion,
                    lossv_cur,
                    accv_cur,
                    device=device,
                )
            avg_acc, std_acc = np.mean(accv_cur), np.std(accv_cur)
            accv.append(avg_acc)
            lg.info(f"Validation: average acc: {avg_acc:.4f}, std acc: {std_acc:.6f}")
            lg.info(f"Test...")
            test(
                model,
                test_loader,
                epoch,
                teacher_criterion if teacher else criterion,
                [],
                [],
                device=device,
            )
            saver.save_model(model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True)
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
