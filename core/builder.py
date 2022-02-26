"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-31 17:48:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:51:50
"""

from typing import Tuple

import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.types import Device

from core.datasets import CIFAR10Dataset, FashionMNISTDataset, MNISTDataset, SVHNDataset
from core.models import *

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(name: str = None) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "mnist":
        train_dataset, validation_dataset, test_dataset = (
            MNISTDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                binarize_threshold=0.273,
                digits_of_interest=list(range(10)),
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "fashionmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )

    return train_loader, validation_loader, test_loader


def make_model(device: Device, random_state: int = None) -> nn.Module:
    if "mlp" in configs.model.name.lower():
        model = eval(configs.model.name)(
            n_feat=configs.dataset.img_height * configs.dataset.img_width,
            n_class=configs.dataset.n_class,
            hidden_list=configs.model.hidden_list,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "cnn" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            pool_out_size=configs.model.pool_out_size,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=configs.model.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=configs.model.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
        # model.super_layer.set_sample_arch(configs.super_layer.sample_arch)
    elif "vgg" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            pool_out_size=configs.model.pool_out_size,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=configs.model.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=configs.model.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
    elif "resnet" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channel=configs.dataset.in_channel,
            n_class=configs.dataset.n_class,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")

    return model


def make_weight_optimizer(model: nn.Module, name: str = None) -> Optimizer:
    name = (name or configs.weight_optimizer.name).lower()

    weight_decay = float(getattr(configs.weight_optimizer, "weight_decay", 0))
    bn_weight_decay = float(getattr(configs.weight_optimizer, "bn_weight_decay", 0))
    bias_decay = float(getattr(configs.weight_optimizer, "bias_decay", 0))
    perm_decay = float(getattr(configs.weight_optimizer, "perm_decay", 0))
    dc_decay = float(getattr(configs.weight_optimizer, "dc_decay", 0))
    groups = {
        str(d): []
        for d in set(
            [
                weight_decay,
                bn_weight_decay,
                bias_decay,
                perm_decay,
                dc_decay,
            ]
        )
    }

    conv_linear = tuple([nn.Linear, _ConvNd] + list(getattr(model, "_conv_linear", [])))
    for m in model.modules():
        if isinstance(m, conv_linear):
            groups[str(weight_decay)].append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                groups[str(bias_decay)].append(m.bias)
        elif isinstance(m, _BatchNorm) and not bn_weight_decay:
            if m.weight is not None and m.weight.requires_grad:
                groups[str(bn_weight_decay)].append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                groups[str(bn_weight_decay)].append(m.bias)
        elif isinstance(m, SuperCRLayer):
            if hasattr(m, "weight") and m.weight.requires_grad:
                groups[str(perm_decay)].append(m.weight)
        elif isinstance(m, SuperDCFrontShareLayer):
            if hasattr(m, "weight") and m.weight.requires_grad:
                groups[str(dc_decay)].append(m.weight)

    selected_params = []
    for v in groups.values():
        selected_params += v

    params_grad = model.weight_params
    other_params = list(set(params_grad) - set(selected_params))
    groups[
        str(weight_decay)
    ] += other_params  # unassigned parameters automatically assigned to weight decay group

    assert len(params_grad) == sum(len(p) for p in groups.values())
    params = [dict(params=params, weight_decay=float(decay_rate)) for decay_rate, params in groups.items()]
    return make_optimizer(params, name, configs.weight_optimizer)


def make_arch_optimizer(model: nn.Module, name: str = None) -> Optimizer:
    name = (name or configs.arch_optimizer.name).lower()

    theta_decay = float(getattr(configs.arch_optimizer, "weight_decay", 5e-4))
    theta = [model.super_layer.sampling_coeff]
    params = [
        dict(params=theta, weight_decay=theta_decay),
    ]
    return make_optimizer(params, name, configs.arch_optimizer)


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(configs.run.n_epochs), eta_min=float(configs.scheduler.lr_min)
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "mixed_kl":
        criterion = KLLossMixed(
            T=getattr(configs.criterion, "T", 3),
            alpha=getattr(configs.criterion, "alpha", 0.9),
        )
    else:
        raise NotImplementedError(name)
    return criterion
