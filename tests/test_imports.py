# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Every module the pipeline imports must import cleanly (fast, CPU, no data needed).

This is the cheapest guard against a refactor that breaks an import somewhere in the code paths
that only run on a GPU machine: `train_ddpwm.py` and `python -m dino_planning.plan` pull in
everything listed below.

    python tests/run_all.py tests/test_imports.py      # just this file
    pytest tests/test_imports.py
"""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = [
    "models.ddp_world_model",
    "models.ddp_predictor",
    "models.dino",
    "models.proprio",
    "models.vit",
    "dsets.pusht_dset",
    "dsets.point_maze_dset",
    "dsets.wall_dset",
    "dsets.deformable_env_dset",
    "dsets.traj_dset",
    "dsets.img_transforms",
    "dino_planning.plan",
    "dino_planning.mpc",
    "dino_planning.cem",
    "dino_planning.evaluator",
    "dino_planning.objectives",
    "dino_planning.vector_env",
    "env.pusht.pusht_env",
    "env.pusht.pusht_wrapper",
    "dino_planning.preprocessor",
    "train_ddpwm",
]


def test_modules_import():
    failures = {}
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report whatever happened
            failures[name] = f"{type(exc).__name__}: {exc}"
    assert not failures, "modules failed to import:\n" + "\n".join(
        f"  {k}: {v}" for k, v in failures.items()
    )


def test_env_registers_pusht():
    import gym

    import env  # noqa: F401 - registers the gym ids

    assert gym.spec("pusht") is not None
