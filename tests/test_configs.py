# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Configuration contract: the YAML files the pipeline ships must satisfy what the code expects.

Checks the three stage sections, the per-environment dataset wiring, and the two ways a config can
be wrong (missing stage section / stale section name) which must fail loudly instead of silently
falling back to the built-in defaults.

    python tests/run_all.py tests/test_configs.py      # just this file
    pytest tests/test_configs.py
"""

import argparse
import inspect
import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_ddpwm  # noqa: E402

STAGES = ["classifier", "predictor", "lrm"]


def _args(**overrides):
    base = dict(
        config=os.path.join(ROOT, "conf", "ddpwm.yaml"),
        stage="predictor",
        env=None,
        data_dir=None,
        batch_size=None,
        lr=None,
        epochs=None,
        num_workers=None,
        output_dir=None,
        classifier_ckpt=None,
        predictor_ckpt=None,
        log_every_n_steps=None,
        save_every_n_epochs=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_training_config_has_every_stage():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "conf", "ddpwm.yaml")))
    for stage in STAGES:
        assert stage in cfg, f"conf/ddpwm.yaml has no '{stage}' section"
        assert "num_hist" in cfg[stage]
    assert cfg["classifier"]["num_hist"] == 3
    for stage in ("predictor", "lrm"):
        assert cfg[stage]["num_hist"] == 5


def test_env_configs_are_complete():
    env_dir = os.path.join(ROOT, "conf", "env")
    names = [f for f in os.listdir(env_dir) if f.endswith(".yaml")]
    assert names, "no conf/env/*.yaml found"
    for f in names:
        cfg = yaml.safe_load(open(os.path.join(env_dir, f)))
        for key in ("name", "data_root", "data_subdir", "dataset"):
            assert key in cfg, f"{f} misses '{key}'"
        assert cfg["name"] == os.path.splitext(f)[0]
        assert "_target_" in cfg["dataset"], f"{f}: dataset needs a _target_ loader"
        # `gym:` was removed on purpose: only PushT ships a gym environment and plan.py does not
        # read a per-environment gym block.
        assert "gym" not in cfg, f"{f} still carries the unused 'gym:' block"


def test_env_dataset_arguments_are_read_by_the_loader():
    """Every <env>.yaml `dataset:` key other than _target_ must be a parameter of its loader.

    build_dataset forwards the block as **kwargs, so a key the loader does not accept (or accepts
    but ignores) is silently dead: `split_ratio` sat in conf/env/pusht.yaml for a while even though
    the PushT split comes from the train/val directories.
    """
    env_dir = os.path.join(ROOT, "conf", "env")
    for f in sorted(os.listdir(env_dir)):
        if not f.endswith(".yaml"):
            continue
        cfg = yaml.safe_load(open(os.path.join(env_dir, f)))
        loader = train_ddpwm._import_loader(cfg["dataset"]["_target_"])
        params = set(inspect.signature(loader).parameters)
        provided = set(cfg["dataset"]) - {"_target_"}
        assert (
            provided <= params
        ), f"{f}: loader {loader.__name__} has no parameter(s) {sorted(provided - params)}"


def test_resolve_config_reads_the_stage_section():
    cfg = train_ddpwm.resolve_config(_args(stage="classifier"))
    assert cfg.num_hist == 3
    for stage in ("predictor", "lrm"):
        cfg = train_ddpwm.resolve_config(_args(stage=stage))
        assert cfg.num_hist == 5, f"{stage}: num_hist must come from the YAML section"
        assert cfg.epochs == 2 and cfg.lr == 7e-4


def test_resolve_config_rejects_a_config_without_the_stage():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({"common": {"env": "pusht"}, "classifier": {"num_hist": 3}}, fh)
        path = fh.name
    try:
        try:
            train_ddpwm.resolve_config(_args(config=path, stage="predictor"))
        except SystemExit as exc:
            assert "predictor" in str(exc)
        else:
            raise AssertionError("a config without the requested stage must be rejected")
    finally:
        os.unlink(path)


def test_resolve_config_rejects_the_old_section_name():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({"common": {"env": "pusht"}, "primary_predictor": {"num_hist": 5}}, fh)
        path = fh.name
    try:
        try:
            train_ddpwm.resolve_config(_args(config=path, stage="predictor"))
        except SystemExit as exc:
            assert "primary_predictor" in str(exc)
        else:
            raise AssertionError("the old 'primary_predictor' section name must be rejected")
    finally:
        os.unlink(path)


def test_zero_valued_cli_options_are_kept():
    """`--num-workers 0`, `--epochs 0` and `--lr 0` are values, not "use the default".

    "`args.x or default`" silently turned them into the YAML defaults (0 workers became 6, which also
    means the trainer spawns DataLoader processes when it was asked not to).
    """
    cfg = train_ddpwm.resolve_config(_args(stage="predictor", num_workers=0, epochs=0, lr=0.0))
    assert cfg.num_workers == 0, cfg.num_workers
    assert cfg.epochs == 0, cfg.epochs
    assert cfg.lr == 0.0, cfg.lr
    # and the YAML/CLI precedence still works for non-zero values
    cfg = train_ddpwm.resolve_config(_args(stage="predictor", num_workers=2, epochs=3, lr=1e-3))
    assert (cfg.num_workers, cfg.epochs, cfg.lr) == (2, 3, 1e-3)
