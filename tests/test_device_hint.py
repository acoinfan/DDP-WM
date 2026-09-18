# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""The device hint must fire whenever a run falls back to the CPU.

A missing GPU is otherwise silent: the entry points fall back to the CPU and the run just gets
~10x slower, which is exactly the failure mode a fresh `pip install torch` produces on a machine
whose driver is older than the newest wheel.

    python tests/run_all.py tests/test_device_hint.py
"""

import logging
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.runtime import log_device  # noqa: E402


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_cpu_fallback_warns_with_a_fix():
    log = logging.getLogger("tests.device_hint")
    handler = _Capture()
    log.addHandler(handler)
    try:
        log_device(log, "cpu", "planning")
    finally:
        log.removeHandler(handler)

    warnings = [m for m in handler.messages if "CUDA is not available" in m]
    assert warnings, f"expected a CUDA warning, got {handler.messages}"
    assert "pip install torch --index-url" in warnings[0], "the hint must say how to fix it"
    assert "planning" in warnings[0], "the hint must name the caller"


def test_cuda_device_is_reported_as_info_when_available():
    if not torch.cuda.is_available():
        return  # nothing to check on a CPU-only machine
    log = logging.getLogger("tests.device_hint.cuda")
    log.setLevel(logging.INFO)  # the announcement is an info record, WARNING is the default level
    handler = _Capture()
    log.addHandler(handler)
    try:
        log_device(log, "cuda:0", "training (classifier)")
    finally:
        log.removeHandler(handler)
    assert any("[DEVICE]" in m and "training (classifier)" in m for m in handler.messages)
