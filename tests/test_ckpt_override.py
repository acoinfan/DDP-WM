# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Evaluation override order: classifier > predictor > LRM.

Each training stage is built on the previous one, so a checkpoint carries its own modules plus the
inherited ones. `dino_planning.plan.merge_checkpoint_layers` resolves the layering; this test pins
the priority order down with fake tensors (no real checkpoint needed).

    python tests/run_all.py tests/test_ckpt_override.py      # just this file
    pytest tests/test_ckpt_override.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dino_planning.plan import _encoder_state_by_priority  # noqa: E402
from models.ckpt_layout import (  # noqa: E402
    GROUPS_CLASSIFIER,
    GROUPS_LRM,
    GROUPS_PREDICTOR,
    merge_checkpoint_layers,
)

# Same keys in every layer, different values, so the winner is obvious.
LRM = {
    "history_fusion.a": "lrm",
    "localizer.b": "lrm",
    "primary_predictor.c": "lrm",
    "lrm.d": "lrm",
}
PRED = {"history_fusion.a": "pred", "localizer.b": "pred", "primary_predictor.c": "pred"}
CLS = {"history_fusion.a": "cls", "localizer.b": "cls"}


def test_priority_order():
    merged = merge_checkpoint_layers(
        [
            ("lrm.pth", LRM, GROUPS_LRM),
            ("predictor.pth", PRED, GROUPS_PREDICTOR),
            ("classifier.pth", CLS, GROUPS_CLASSIFIER),
        ]
    )
    assert merged == {
        "history_fusion.a": "cls",  # classifier wins over predictor and LRM
        "localizer.b": "cls",
        "primary_predictor.c": "pred",  # predictor wins over LRM
        "lrm.d": "lrm",
    }, merged  # LRM is the only source


def test_single_layer_is_the_identity():
    assert merge_checkpoint_layers([("lrm.pth", LRM, GROUPS_LRM)]) == LRM


def test_predictor_only_keeps_its_own_groups():
    """A predictor checkpoint must not bring an `lrm.*` group by accident."""
    merged = merge_checkpoint_layers([("predictor.pth", PRED, GROUPS_PREDICTOR)])
    assert set(merged) == set(PRED)


def test_encoders_follow_the_same_priority_as_the_modules():
    """The shared encoders come from the classifier layer, not from the predictor layer."""
    layers = [
        ("lrm.pth", {}, GROUPS_LRM, {"action_encoder": "lrm", "proprio_encoder": "lrm"}),
        ("predictor.pth", {}, GROUPS_PREDICTOR, {"action_encoder": "pred"}),
        ("classifier.pth", {}, GROUPS_CLASSIFIER, {"action_encoder": "cls"}),
    ]
    assert _encoder_state_by_priority(layers) == {"action_encoder": "cls", "proprio_encoder": "lrm"}
