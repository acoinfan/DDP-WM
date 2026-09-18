# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Forward + backward smoke test for the three training stages, without DINOv2 or a dataset.

The visual encoder is replaced by a stub that returns exactly what the real DINOv2 ViT-S/14
produces (196 patch tokens x 384 dims), so this runs on CPU in a few seconds and still catches the
things that break most often: token shapes, mask bookkeeping and the per-stage loss dispatch.

    python tests/run_all.py tests/test_model_forward.py      # just this file
    pytest tests/test_model_forward.py
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.ddp_world_model import DDPWorldModel  # noqa: E402
from models.proprio import ProprioceptiveEmbedding  # noqa: E402

D_VIS, N_PATCHES = 384, 196
B, T = 2, 5


class FakeDino(nn.Module):
    """Stands in for models.dino.DinoV2Encoder (same output shape, nothing to download)."""

    name = "dinov2_vits14"
    patch_size = 14

    def forward(self, x):  # (B*T, 3, H, W)
        return torch.randn(x.shape[0], N_PATCHES, D_VIS)


def build(stage, num_hist=5):
    return DDPWorldModel(
        image_size=224,
        num_hist=num_hist,
        training_stage=stage,
        encoder=FakeDino(),
        action_encoder=ProprioceptiveEmbedding(in_chans=10, emb_dim=10, tubelet_size=1),
        proprio_encoder=ProprioceptiveEmbedding(in_chans=4, emb_dim=10, tubelet_size=1),
        proprio_dim=10,
        action_dim=10,
    )


def make_batch(t=T):
    obs = {"visual": torch.randn(B, t, 3, 224, 224), "proprio": torch.randn(B, t, 4)}
    act = torch.randn(B, t, 10)
    return obs, act


def _forward_backward(model):
    """One training step; asserts the loss is finite and that something got a gradient."""
    obs, act = make_batch()
    model.train()
    _, _, _, loss, components = model(obs, act)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()
    graded = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    assert graded, "backward() reached no parameter"
    return float(loss), components


def test_classifier_stage():
    loss, _ = _forward_backward(build("classifier", num_hist=3))
    assert loss >= 0


def test_predictor_stage_fused_history():
    model = build("predictor")
    model.history_fusion_disabled = False
    loss, _ = _forward_backward(model)
    assert loss >= 0


def test_predictor_stage_single_frame():
    model = build("predictor")
    model.history_fusion_disabled = True
    loss, _ = _forward_backward(model)
    assert loss >= 0


def test_lrm_stage():
    _, components = _forward_backward(build("lrm"))
    assert "loss_lrm_mse" in components


def test_rollout_sees_the_history_the_recipe_asks_for():
    """1 frame when history_fusion_disabled, 3 frames otherwise."""
    model = build("inference", num_hist=3)

    class SpyPredictor(nn.Module):
        seen = None

        def forward(self, z_hist):
            SpyPredictor.seen = z_hist.shape[1]
            return {"prediction": torch.zeros(B, 1, N_PATCHES, 404)}

    model.predictor = SpyPredictor()
    model.history_fusion_disabled = True
    model.rollout(*make_batch(t=3))
    assert SpyPredictor.seen == 1

    model.history_fusion_disabled = False
    model.rollout(*make_batch(t=3))
    assert SpyPredictor.seen == 3
