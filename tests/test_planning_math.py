# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Planning-side maths on CPU: the sparse cost mask, the CEM initialisation and the MPC mask.

These are the pieces that decide the success rate, and until now a change to them was only visible in
a full 50-scene evaluation (tens of minutes on a GPU). Everything here is pure tensor maths, so it
runs in milliseconds without a dataset, a checkpoint or a GPU.

    python tests/run_all.py tests/test_planning_math.py      # just this file
    pytest tests/test_planning_math.py
"""

import ast
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.aggregate import aggregate_dct  # noqa: E402
from dino_planning.cem import CEMPlanner  # noqa: E402
from dino_planning.mpc import MPCPlanner  # noqa: E402
from dino_planning.objectives import (  # noqa: E402
    compute_patch_mask,
    dilate_mask,
    masked_visual_loss,
)
from dino_planning.preprocessor import Preprocessor  # noqa: E402
from dino_planning.utils import slice_trajdict_with_t  # noqa: E402

N_PATCHES = 196  # 14x14 DINOv2 patch grid


def _mask_with(active_indices):
    mask = torch.zeros(1, N_PATCHES)
    mask[0, active_indices] = 1.0
    return mask


def test_dilate_mask_uses_the_requested_connectivity():
    # patch 98 = (row 7, col 0) touches the left border: the cross kernel adds 3 patches, the square
    # kernel 5. Patch 105 = (row 7, col 7) is interior: 5 respectively 9.
    assert dilate_mask(_mask_with([98]), connectivity=0).sum().item() == 1
    assert dilate_mask(_mask_with([98]), connectivity=4).sum().item() == 4
    assert dilate_mask(_mask_with([98]), connectivity=8).sum().item() == 6
    assert dilate_mask(_mask_with([105]), connectivity=4).sum().item() == 5
    assert dilate_mask(_mask_with([105]), connectivity=8).sum().item() == 9
    # the value stays binary (the convolution produces counts, not booleans)
    assert set(dilate_mask(_mask_with([105]), connectivity=8).unique().tolist()) == {0.0, 1.0}


def test_compute_patch_mask_locates_the_changed_region():
    obs_0 = torch.zeros(1, 3, 224, 224)
    obs_g = obs_0.clone()
    obs_g[0, :, 0:16, 0:16] = 1.0  # exactly patch (0, 0): 224 / 14 = 16 pixels per patch

    mask = compute_patch_mask(obs_0, obs_g, threshold=0.1, dilation=0)
    assert mask.shape == (1, N_PATCHES)
    assert mask[0, 0].item() == 1.0 and mask.sum().item() == 1.0

    # dilation=4 adds the in-grid neighbours of the corner patch -> 3 active patches
    assert compute_patch_mask(obs_0, obs_g, threshold=0.1, dilation=4).sum().item() == 3

    # a difference below the threshold is not a task-relevant patch
    faint = obs_0 + 0.05
    assert compute_patch_mask(obs_0, faint, threshold=0.1, dilation=0).sum().item() == 0

    # the same computation accepts the (B, 1, C, H, W) layout the planner passes
    mask_5d = compute_patch_mask(obs_0[:, None], obs_g[:, None], threshold=0.1, dilation=0)
    assert torch.equal(mask_5d, mask)


def test_masked_visual_loss_averages_over_the_active_patches():
    """The masked cost is the mean squared error of the last frame *on the active patches only*."""
    d_feat, d_proprio = 4, 2
    pred = {
        "visual": torch.zeros(1, 1, N_PATCHES, d_feat),
        "proprio": torch.zeros(1, 1, d_proprio),
    }
    tgt = {
        "visual": torch.zeros(1, 1, N_PATCHES, d_feat),
        "proprio": torch.zeros(1, 1, d_proprio),
    }
    pred["visual"][0, 0, 0] = 1.0  # squared error 1 on patch 0
    pred["visual"][0, 0, 5] = 2.0  # squared error 4 on patch 5

    only_0 = torch.zeros(1, N_PATCHES)
    only_0[0, 0] = 1.0
    only_5 = torch.zeros(1, N_PATCHES)
    only_5[0, 5] = 1.0
    both = (only_0 + only_5).clamp(max=1.0)

    assert torch.isclose(masked_visual_loss(pred, tgt, only_0), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(masked_visual_loss(pred, tgt, only_5), torch.tensor(4.0), atol=1e-6)
    assert torch.isclose(masked_visual_loss(pred, tgt, both), torch.tensor(2.5), atol=1e-6)
    # an empty mask must not divide by zero
    assert torch.isfinite(masked_visual_loss(pred, tgt, torch.zeros(1, N_PATCHES)))


def test_patch_mask_is_computed_from_the_current_start_frame():
    """The mask follows the start frame: once the object has moved onto the goal patch, it is empty.

    This is what the reference CEM does on every MPC round (`start = the round's current frame`), as
    opposed to computing one mask from the very first frame and freezing it.
    """
    goal = torch.zeros(1, 3, 224, 224)
    goal[0, :, 0:16, 0:16] = 1.0  # the object sits on patch (0, 0)
    start_far = torch.zeros(1, 3, 224, 224)
    start_near = goal.clone()

    assert compute_patch_mask(start_far, goal, threshold=0.1, dilation=0)[0, 0] == 1.0
    assert compute_patch_mask(start_near, goal, threshold=0.1, dilation=0).sum().item() == 0.0


def test_cem_init_mu_sigma_pads_to_the_horizon():
    planner = CEMPlanner.__new__(CEMPlanner)  # this method only needs three attributes
    planner.horizon, planner.action_dim, planner.var_scale = 5, 2, 1.0
    obs_0 = {"visual": torch.zeros(3, 1, 3, 224, 224)}

    mu, sigma = planner.init_mu_sigma(obs_0)
    assert mu.shape == (3, 5, 2) and sigma.shape == (3, 5, 2)
    assert torch.all(mu == 0) and torch.all(sigma == 1.0)

    given = torch.ones(3, 2, 2)
    mu, sigma = planner.init_mu_sigma(obs_0, given)
    assert mu.shape == (3, 5, 2)
    assert torch.all(mu[:, :2] == 1.0) and torch.all(mu[:, 2:] == 0.0)


def test_mpc_success_mask_zeroes_finished_trajectories():
    planner = MPCPlanner.__new__(MPCPlanner)
    planner.is_success = [False, True]
    planner.evaluator = SimpleNamespace(frameskip=5)
    planner.preprocessor = Preprocessor(
        action_mean=torch.zeros(2),
        action_std=torch.ones(2),
        state_mean=torch.zeros(2),
        state_std=torch.ones(2),
        proprio_mean=torch.zeros(2),
        proprio_std=torch.ones(2),
        transform=None,
    )
    # the planner keeps each trajectory's taken actions as (n_taken, frameskip * action_dim), so a
    # finished trajectory can be reset to "no action" before it is sent back to the environment
    actions = torch.ones(2, 5, 10)  # (n_evals, n_taken_actions, frameskip * action_dim)
    masked = planner._apply_success_mask(actions)
    assert torch.all(masked[0] == 1.0)  # unfinished trajectory keeps its plan
    assert torch.all(masked[1] == 0.0)  # finished trajectory is held at the (normalised) origin


def test_aggregate_dct_and_slice_trajdict():
    per_env = [{"x": torch.zeros(2, 3), "y": [1, 2]}, {"x": torch.ones(2, 3), "y": [3, 4]}]
    aggregated = aggregate_dct(per_env)
    assert aggregated["x"].shape == (2, 2, 3)
    assert isinstance(aggregated["y"], np.ndarray) and aggregated["y"].shape == (2, 2)

    data = {"visual": torch.arange(2 * 4).view(2, 4, 1)}
    assert slice_trajdict_with_t(data, start_idx=-1)["visual"].shape == (2, 1, 1)
    assert slice_trajdict_with_t(data, start_idx=0, end_idx=2)["visual"].shape == (2, 2, 1)


def test_env_layer_does_not_import_the_planner():
    """`env/` must not depend on `dino_planning/`; the shared helper lives in `common/`."""
    wrapper = os.path.join(ROOT, "env", "pusht", "pusht_wrapper.py")
    tree = ast.parse(open(wrapper).read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [m for m in imported if m.startswith("dino_planning")], imported
