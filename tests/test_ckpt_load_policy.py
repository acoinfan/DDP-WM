# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""`torch.load` must never depend on the installed torch version's default.

PyTorch 2.6 changed the default of `weights_only` from `False` to `True`. A `torch.load(...)` without
that keyword therefore loads an old checkpoint on one machine and raises on another -- exactly the
kind of "different environment, different outcome" this repository must not have. These tests pin the
policy down: every call in the published tree passes `weights_only` explicitly, `load_checkpoint()`
fills it in, and only the DINO-WM reference loader is allowed to ask for `False`.

    python tests/run_all.py tests/test_ckpt_load_policy.py      # just this file
    pytest tests/test_ckpt_load_policy.py
"""

import ast
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.ckpt_io import load_checkpoint  # noqa: E402

# The only place allowed to unpickle arbitrary objects: the reference DINO-WM checkpoints store
# nn.Module instances, so that loader passes weights_only=False on purpose.
ALLOWED_FALSE = {"dino_planning/plan.py"}

# This module *is* the policy: it forwards a `weights_only` parameter, so its value is not a literal.
POLICY_MODULE = "models/ckpt_io.py"


# Directories that are not part of the published source tree (weights, outputs, internal tooling).
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "_archive",
    "_archive_md",
    "_audit",
    "artifacts",
    "build",
    "CKPT_backup",
    "dist",
    "docs",
    "evals",
    "evals_test",
    "garbage",
    "history",
    "logging",
    "plan_outputs",
    "pretrained",
    "runs",
    "wandb",
}


def _source_files():
    """Every python file of the tree (relative paths), skipping vendored and generated code."""
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
                if not rel.startswith("env/pusht/"):  # vendored: not ours to change
                    found.append(rel)
    return sorted(found)


def _torch_load_calls(path):
    """Every `torch.load(...)` call in one file, as (lineno, keyword dict),"""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_torch_load = (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        )
        if is_torch_load:
            yield node.lineno, {k.arg: k.value for k in node.keywords}


def _literal_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _literal_false(node):
    return isinstance(node, ast.Constant) and node.value is False


def test_every_torch_load_passes_weights_only():
    offenders = []
    for rel in _source_files():
        for lineno, kw in _torch_load_calls(os.path.join(ROOT, rel)):
            if "weights_only" not in kw:
                offenders.append(f"{rel}:{lineno} (no weights_only)")
            elif (
                rel not in ALLOWED_FALSE
                and rel != POLICY_MODULE
                and not _literal_true(kw["weights_only"])
            ):
                offenders.append(f"{rel}:{lineno} (weights_only is not the literal True)")
    assert not offenders, f"torch.load without the explicit policy: {offenders}"


def test_only_the_reference_loader_uses_weights_only_false():
    users = []
    for rel in _source_files():
        for _lineno, kw in _torch_load_calls(os.path.join(ROOT, rel)):
            if "weights_only" in kw and _literal_false(kw["weights_only"]):
                users.append(rel)
    assert set(users) == ALLOWED_FALSE, users


def test_load_checkpoint_forwards_weights_only():
    recorded = {}
    original = torch.load

    def fake(path, **kwargs):
        recorded.update(kwargs)
        return {"path": str(path)}

    torch.load = fake
    try:
        assert load_checkpoint("some.pth") == {"path": "some.pth"}
    finally:
        torch.load = original
    assert recorded.get("weights_only") is True
    assert recorded.get("map_location") == "cpu"


def test_our_checkpoints_load_with_weights_only_true():
    """The contract the policy rests on: our layout is tensors plus basic python types."""
    import tempfile

    ckpt = {
        "epoch": 2,
        "predictor": {"history_fusion.w": torch.zeros(3)},
        "action_encoder": {},
        "history_fusion_disabled": True,
        "optimizer": {"state": {}, "param_groups": [{"lr": 7e-4, "betas": (0.9, 0.999)}]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model_latest.pth")
        torch.save(ckpt, path)
        loaded = load_checkpoint(path)
    assert loaded["epoch"] == 2 and loaded["history_fusion_disabled"] is True
    assert torch.equal(loaded["predictor"]["history_fusion.w"], torch.zeros(3))
