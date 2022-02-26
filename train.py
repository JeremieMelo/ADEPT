"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-26 15:49:07
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-26 16:06:20
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Iterable

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


def legalize_perm(model, area_loss_func: Callable):
    """Stochastic permutation legalization (SPL)

    Args:
        model (_type_): _description_
        area_loss_func (Callable): _description_
    """
    from core.models.layers import SuperCRLayer

    optimizer = torch.optim.Adam(
        [m.weight for m in model.super_layer.super_layers_all if isinstance(m, SuperCRLayer)], lr=1e-3
    )  # max_lambda = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=2e-4)
    lg.info(f"Force to legalize permutation")
    for step in range(3000):
        model.build_arch_mask(mode="gumbel_soft")
        optimizer.zero_grad()
        alm_perm_loss = model.get_alm_perm_loss(rho=1e-7)
        area_loss = area_loss_func()
        cross_density_loss = model.get_crossing_density_loss(margin=0.8)
        loss = alm_perm_loss + area_loss + 1 * cross_density_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_alm_multiplier(rho=1e-7)
        with torch.no_grad():
            if step % 200 == 0:
                legal = model.check_perm()
                perm_loss = model.get_perm_loss()
                num_cr = model.get_num_crossings()
                lg.info(
                    f"Step: {step}, Perm Loss: {perm_loss}, Perm legality: {legal}, Area Loss: {area_loss.data.item()}, Area: {model.area.item()}, CR Density Loss: {cross_density_loss.data.item()}, #CR: {num_cr}"
                )
    legal = model.check_perm()
    lg.info(f"Legalize permutation...")
    model.sinkhorn_perm(n_step=200, t_min=0.005, noise_std=0.01, svd=True, legal_mask=legal)
    legal = model.check_perm()
    lg.info(f"Final perm legality: {legal}...")
    if all(legal):
        lg.info("All permutations are legal")
    else:
        lg.info("Not all permutations are legal!")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    weight_optimizer: Optimizer,
    arch_optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
    teacher: nn.Module = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0
    init_T = float(getattr(configs.super_layer, "init_gumbel_temperature", 5))
    gamma_T = float(getattr(configs.super_layer, "gumbel_decay_rate", 0.956))
    ps_weight = float(getattr(configs.super_layer.arch.device_cost, "ps_weight", 1))
    dc_weight = float(getattr(configs.super_layer.arch.device_cost, "dc_weight", 1))
    cr_weight = float(getattr(configs.super_layer.arch.device_cost, "cr_weight", 1))
    area_upper_bound = float(getattr(configs.super_layer.arch.device_cost, "area_upper_bound", 100))
    area_lower_bound = float(getattr(configs.super_layer.arch.device_cost, "area_lower_bound", 80))
    first_active_block = bool(getattr(configs.super_layer.arch.device_cost, "first_active_block", 1))
    area_loss_rho = float(getattr(configs.criterion, "area_loss_rho", 0))
    cross_density_loss_rho = float(getattr(configs.criterion, "cross_density_loss_rho", 0))
    perm_loss_rho = float(getattr(configs.criterion, "perm_loss_rho", 0))
    perm_loss_rho_gamma = float(getattr(configs.criterion, "perm_loss_rho_gamma", 1))
    max_lambda = float(getattr(configs.criterion, "max_lambda", 1))
    force_perm_legal_epoch = int(getattr(configs.run, "force_perm_legal_epoch", 60))
    train_arch_epoch = int(getattr(configs.run, "train_arch_epoch", 10))
    train_arch_interval = int(getattr(configs.run, "train_arch_interval", 3))
    phase_noise_std = float(getattr(configs.noise, "phase_noise_std", 0))
    dc_noise_std = float(getattr(configs.noise, "dc_noise_std", 0))
    model.set_phase_noise(phase_noise_std)
    model.set_dc_noise(dc_noise_std)

    if epoch >= train_arch_epoch:
        perm_loss_rho = perm_loss_rho * perm_loss_rho_gamma ** (epoch - train_arch_epoch)
        lg.info(f"Permutation ALM Rho: {perm_loss_rho}")
    # set gumbel softmax temperature
    T = init_T * gamma_T ** (epoch - 1)
    model.set_gumbel_temperature(T)
    lg.info(f"Gumbel temperature: {T:.4f}/{init_T}")

    arch_mask_mode = getattr(configs.super_layer, "arch_mask_mode", "gumbel_soft")

    train_arch_flag = False

    model.enable_weight_params()

    if epoch == force_perm_legal_epoch:
        legalize_perm(
            model,
            lambda x=None: area_loss_rho
            * model.get_area_bound_loss(
                ps_weight,
                dc_weight,
                cr_weight,
                area_upper_bound,
                area_lower_bound,
                first_active_block=first_active_block,
            ),
        )

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if epoch >= train_arch_epoch and batch_idx % train_arch_interval == (train_arch_interval - 1):
            model.enable_arch_params()
            model.freeze_weight_params()
            arch_optimizer.zero_grad()
            train_arch_flag = True
        else:
            model.enable_weight_params()
            model.freeze_arch_params()
            weight_optimizer.zero_grad()
            train_arch_flag = False

        def _get_loss(output, target):
            if teacher:
                with torch.no_grad():
                    teacher_score = teacher(data).detach()
                loss = criterion(output, teacher_score)
            else:
                loss = criterion(output, target)
            return loss

        # sample random subnet
        model.build_arch_mask(mode=arch_mask_mode)
        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        loss = _get_loss(output, target)
        class_loss = loss

        if epoch >= train_arch_epoch and area_loss_rho > 0:  # no area penalty in warmup
            area_loss = model.get_area_bound_loss(
                ps_weight,
                dc_weight,
                cr_weight,
                area_upper_bound,
                area_lower_bound,
                first_active_block=first_active_block,
            )
            loss = loss + area_loss_rho * area_loss
        else:
            area_loss = torch.zeros(1)

        # if train_arch_flag and perm_loss_rho > 0: # only train permutation in arch opt phase
        if (
            epoch >= train_arch_epoch and not train_arch_flag and perm_loss_rho > 0
        ):  # only train permutation in weight opt phase; no constraints in warmup
            alm_perm_loss = model.get_alm_perm_loss(rho=perm_loss_rho)
            loss = loss + alm_perm_loss

        with torch.no_grad():
            perm_loss = model.get_perm_loss()

        if cross_density_loss_rho > 0 and not train_arch_flag:
            cross_density_loss = model.get_crossing_density_loss(margin=0.95)
            loss = loss + cross_density_loss_rho * cross_density_loss
        else:
            cross_density_loss = torch.zeros(1)

        loss.backward()
        if train_arch_flag:
            arch_optimizer.step()
        else:
            weight_optimizer.step()

        # update permutation ALM multiplier
        if epoch >= train_arch_epoch and not train_arch_flag and perm_loss_rho > 0:
            model.update_alm_multiplier(perm_loss_rho, max_lambda=max_lambda)

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info(
                "Train Epoch: {} ({}) [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f} Perm Loss: {} Perm ALM: {} Area Loss: {:.4f} Area ALM: {:.4f} Area Aux: {:.4f} Area: {:.2f} CR_D_Loss: {:.4f} N_CR: {} N_DC: {} Theta\n{}".format(
                    epoch,
                    "train_arch" if train_arch_flag else "train_weight",
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    class_loss.data.item(),
                    perm_loss,
                    model.get_alm_multiplier(),
                    area_loss.item(),
                    model.get_area_multiplier().item(),
                    model.area_aux_variable.data.item(),
                    model.area.data,
                    cross_density_loss.item(),
                    model.get_num_crossings(),
                    model.get_num_dc(),
                    model.super_layer.sampling_coeff.data,
                )
            )
            lg.info(f"arch_mask:\n{model.super_layer.arch_mask.data}")
            lg.info(f"Check permutation legality: {model.check_perm()}")
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.item(), "lr": get_learning_rate(weight_optimizer)}, step=epoch)


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
    model.partition_parameters(arch_param_list=["theta"])

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    weight_optimizer = builder.make_weight_optimizer(model)
    arch_optimizer = builder.make_arch_optimizer(model)
    scheduler = builder.make_scheduler(weight_optimizer)
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
            "init_weight_lr": configs.weight_optimizer.lr,
            "init_arch_lr": configs.arch_optimizer.lr,
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

            lg.info("Validate resumed model...")
            validate(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device=device,
            )
            state_dict = torch.load(configs.checkpoint.restore_checkpoint)
            state_dict = state_dict.get("state_dict", state_dict)
            if "solution" in state_dict.keys():
                solution = state_dict["solution"]
                lg.info(f"Loading the solution {solution}")
                lg.info(f"Original score: {state_dict['score']}")
            model.set_sample_arch(solution["arch"])
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

        for epoch in range(1, int(configs.run.n_epochs) + 1):

            train(
                model,
                train_loader,
                weight_optimizer,
                arch_optimizer,
                scheduler,
                epoch,
                criterion,
                device,
                teacher=teacher,
            )

            if epoch > int(configs.run.n_epochs) - 10:  # validate and store in the last 10 epochs
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
                lg.info(f"Validation: average acc: {avg_acc}, std acc: {std_acc}")
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
                saver.save_model(
                    model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True
                )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
