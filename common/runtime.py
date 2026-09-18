# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Runtime helpers shared by the training and planning entry points."""

import torch


def log_device(log, device, where: str) -> None:
    """Announce the compute device, and explain what to fix when CUDA is not available.

    A missing GPU is easy to miss: the entry points fall back to the CPU, which only shows up as a
    run that is an order of magnitude slower. The most common reason is a torch build whose bundled
    CUDA version is newer than the installed driver (a fresh `pip install torch` picks the newest
    wheel, which may be built for a driver the machine does not have yet).
    """
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        log.info("[DEVICE] %s uses %s (%s)", where, device, torch.cuda.get_device_name(device))
        return
    log.warning(
        "[DEVICE] %s uses %s: CUDA is not available to this interpreter, so it runs on the CPU "
        "(roughly an order of magnitude slower). Installed torch is %s, built for CUDA %s; "
        "`nvidia-smi` prints the newest CUDA version the driver supports. Install a torch wheel "
        "that matches the driver, e.g. `pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121`, and re-run.",
        where,
        device,
        torch.__version__,
        torch.version.cuda,
    )
