# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Loading checkpoints with an explicit `weights_only` policy.

PyTorch changed the default of `torch.load(..., weights_only=...)` in 2.6: older versions default to
`False` (unpickle anything), 2.6+ default to `True` (tensors and basic python types only). Relying on
the default therefore means the *same code* behaves differently on two machines -- an old checkpoint
that pickles classes loads on torch 2.3 and raises on torch 2.7. Every load in this repository goes
through `load_checkpoint()` (or passes `weights_only` explicitly) so that the behaviour is fixed here
instead of by the installed torch version.

What is safe with which policy:

* checkpoints written by this repository -> `weights_only=True`. They contain tensors plus ints /
  floats / bools / strings / lists / dicts (including the optimizer state), nothing else.
* the reference DINO-WM checkpoints -> `weights_only=False`, because they pickle `nn.Module`
  objects. Only `dino_planning.plan.load_dinowm_model` loads those, and it asks for it explicitly.
"""

import torch


def load_checkpoint(path, map_location="cpu", weights_only: bool = True):
    """``torch.load`` with an explicit, version-independent ``weights_only`` policy."""
    return torch.load(path, map_location=map_location, weights_only=weights_only)
