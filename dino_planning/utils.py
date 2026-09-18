# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""Utility functions for dino_planning module."""

import random

import numpy as np
import torch
from omegaconf import OmegaConf


def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(str(x) for x in cfg_dict[key])
    return cfg_dict


def seed(s):
    random.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct


def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}
