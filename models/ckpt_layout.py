# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Checkpoint key layout of this reproduction, shared by training, evaluation and the planner.

Every stage is trained on top of the previous one, so a checkpoint contains the modules it owns
*plus* the ones it inherited:

    classifier : history fusion + localizer (+ the action/proprio encoders)
    predictor  : the above + the sparse predictor (ViT / heads)
    LRM        : the above + the LRM layer

At evaluation time the checkpoint being evaluated is the base layer and the optional classifier /
predictor checkpoints override its modules, highest priority last:
``classifier > predictor > LRM``.

This module is deliberately dependency-free (plain dicts, no torch) so that the trainer, the
planner and the model code all share one definition of the layout and of the override order.
"""

# Keys of the pipeline this repository replaces. A checkpoint that still uses them has to be
# remapped offline; runtime key remapping has been removed on purpose.
LEGACY_KEY_PREFIXES = (
    "dino_wm.",
    "dino_wm_ti.",
    "glow_decoder.",
    "history_encoder.",
    "cls_head.",
    "dim_reduce_layer.",
    "hist_",
)

# Module groups, from the stage that trains them first to the last one.
GROUPS_CLASSIFIER = ("history_fusion.", "localizer.")
GROUPS_PREDICTOR = GROUPS_CLASSIFIER + ("primary_predictor.",)
GROUPS_LRM = GROUPS_PREDICTOR + ("lrm.",)

# Human-readable labels, used in the "this checkpoint has no weights for X" errors.
_GROUP_LABELS = {
    "history_fusion.": "history fusion",
    "localizer.": "localizer",
    "primary_predictor.": "predictor (ViT/heads)",
    "lrm.": "LRM",
}

# What a forward pass of each training stage needs to have come from a checkpoint.
_STAGE_GROUPS = {
    "classifier": GROUPS_CLASSIFIER,
    "predictor": GROUPS_PREDICTOR,
    "lrm": GROUPS_LRM,
    "inference": GROUPS_LRM,
}


def groups_for_stage(stage: str) -> tuple:
    """Module groups a checkpoint must provide for ``stage`` (classifier < predictor < LRM)."""
    try:
        return _STAGE_GROUPS[stage]
    except KeyError:
        raise ValueError(
            f"unknown stage '{stage}'; expected one of {sorted(_STAGE_GROUPS)}"
        ) from None


def assert_current_layout(state: dict, path: str, source: str = "checkpoint") -> None:
    """Raise when ``state`` still uses the legacy key layout (see the conversion notes)."""
    legacy = [k for k in state if k.startswith(LEGACY_KEY_PREFIXES)]
    if legacy:
        raise RuntimeError(
            f"[{source}] {path} uses the legacy checkpoint layout "
            f"({len(legacy)} legacy keys, e.g. {legacy[:3]}).\n"
            "  Runtime key remapping has been removed: convert the checkpoint to the current "
            "layout\n  (history_fusion.* / localizer.* / primary_predictor.* / lrm.*) offline first."
        )


def merge_checkpoint_layers(layers: list) -> dict:
    """Merge ``(path, state_dict, groups)`` layers, lowest priority first.

    For every key the highest-priority layer that carries it wins, which is exactly the override
    rule: ``ddpwm_cls_ckpt`` beats ``ddpwm_predictor_ckpt`` beats the evaluated checkpoint.
    """
    merged = {}
    for _path, state, groups in layers:
        for key, value in state.items():
            if key.startswith(groups):
                merged[key] = value
    return merged


def missing_groups(state: dict, groups: tuple) -> list[str]:
    """Labels of the groups in ``groups`` that no weight in ``state`` belongs to."""
    return [
        _GROUP_LABELS[prefix]
        for prefix in groups
        if not any(key.startswith(prefix) for key in state)
    ]
