# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Checkpoint contract for the released weights in pretrained/ (skipped if they are absent).

The repository supports exactly one key layout: `history_fusion.*` / `localizer.*` /
`primary_predictor.*` / `lrm.*`. Legacy flat-key checkpoints raise a clear error at load time and
have to be converted offline, so a test that walks the shipped files is cheap insurance.

    python tests/run_all.py tests/test_ckpt_layout.py      # just this file
    pytest tests/test_ckpt_layout.py
"""

import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.ddp_world_model import DDPWorldModel  # noqa: E402
from models.proprio import ProprioceptiveEmbedding  # noqa: E402
from tests.test_model_forward import FakeDino  # noqa: E402

GROUPS = ("history_fusion.", "localizer.", "primary_predictor.", "lrm.")
LEGACY = (
    "dino_wm.",
    "dino_wm_ti.",
    "glow_decoder.",
    "history_encoder.",
    "cls_head.",
    "dim_reduce_layer.",
    "hist_",
)


def _ckpts():
    return sorted(glob.glob(os.path.join(ROOT, "pretrained", "*.pth")))


def test_pretrained_files_use_the_current_layout():
    files = _ckpts()
    if not files:
        print("  (no pretrained/*.pth on this machine -- skipped)")
        return
    for path in files:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("predictor", ckpt)
        legacy = [k for k in state if k.startswith(LEGACY)]
        assert not legacy, f"{path}: legacy keys {legacy[:3]}"
        unknown = [k for k in state if not k.startswith(GROUPS)]
        assert not unknown, f"{path}: unexpected keys {unknown[:3]}"
        assert "history_fusion_disabled" in ckpt, f"{path}: missing the recipe flag"
        assert torch.is_tensor(next(iter(state.values())))


def test_pretrained_weights_load_into_the_model():
    files = _ckpts()
    if not files:
        print("  (no pretrained/*.pth on this machine -- skipped)")
        return
    model = DDPWorldModel(
        image_size=224,
        num_hist=5,
        training_stage="inference",
        encoder=FakeDino(),
        action_encoder=ProprioceptiveEmbedding(in_chans=10, emb_dim=10, tubelet_size=1),
        proprio_encoder=ProprioceptiveEmbedding(in_chans=4, emb_dim=10, tubelet_size=1),
        proprio_dim=10,
        action_dim=10,
    )
    for path in files:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = dict(ckpt.get("predictor", ckpt))
        missing, unexpected = model.predictor.load_state_dict(state, strict=False)
        assert not unexpected, f"{path}: unexpected keys {unexpected[:3]}"
        # A classifier checkpoint legitimately misses the predictor / LRM groups; what must hold is
        # that it filled the groups it does carry.
        assert len(missing) < len(model.predictor.state_dict()), f"{path}: nothing was loaded"
