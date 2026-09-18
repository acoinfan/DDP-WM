# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Aggregate a list of per-environment dicts into a dict of stacked arrays.

Used by the vectorised environment (`dino_planning/vector_env.py`) and by the PushT gym wrapper
(`env/pusht/pusht_wrapper.py`), which is why it lives here instead of in either of them.
"""

import numpy as np
import torch


def aggregate_dct(dcts: list[dict]) -> dict:
    """``[{k: value}, ...]`` -> ``{k: stacked values}`` (torch tensors stay tensors)."""
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            full_dct.setdefault(key, []).append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct
