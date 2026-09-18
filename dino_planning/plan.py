# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""
DINO-WM / DDP-WM Planning Entry Script.

Supports:
  - DINO-WM (VWorldModel) with old-format checkpoints
  - DDP-WM (DDPWorldModel) with key-remapped old-format checkpoints
  - Dense MSE cost (standard DINO-WM objective)
  - Sparse masked MSE cost (DDP-WM's Sparse MPC Cost Mask)

Usage:
    cd <repository root>

    # DDP-WM with dense cost:
    CUDA_VISIBLE_DEVICES=0 python -m dino_planning.plan use_sparse_cost=false

    # DDP-WM with sparse cost:
    CUDA_VISIBLE_DEVICES=0 python -m dino_planning.plan use_sparse_cost=true
"""

import contextlib
import json
import logging
import os
import pickle
import random
import sys
import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from dino_planning.evaluator import PlanEvaluator
from dino_planning.utils import cfg_to_dict, seed
from common.runtime import log_device
from models.ckpt_io import load_checkpoint
from models.ckpt_layout import (
    GROUPS_CLASSIFIER,
    GROUPS_LRM,
    GROUPS_PREDICTOR,
    assert_current_layout,
    merge_checkpoint_layers,
    missing_groups,
)

from .preprocessor import Preprocessor

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Register Hydra resolver for config paths
OmegaConf.register_new_resolver("replace_slash", lambda x: str(x).replace("/", "_"), replace=True)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# =====================================================================
# Checkpoint layout
# =====================================================================
# The key layout, the module groups and the override order live in models/ckpt_layout.py so that the
# trainer, the planner and the model code cannot drift apart. `_GROUPS_*` are re-exported here
# because the evaluation code (and its tests) read them from this module.
# Model shape used at evaluation time. The PushT training stages build the world model with exactly
# these dimensions (see train_ddpwm.build_model) and the checkpoints do not store them, so the
# evaluation has to mirror the recipe: 3 history frames, a 2-D action and a 4-D raw proprio input.
_PUSHT_NUM_HIST = 3
_PUSHT_ACTION_DIM = 10
_PUSHT_PROPRIO_RAW_DIM = 4


@contextlib.contextmanager
def _temporary_sys_path(*paths):
    """Put the existing directories of `paths` on sys.path for the duration of the block.

    Used before unpickling the reference DINO-WM checkpoints, whose classes have to be importable.
    """
    added = []
    for path in paths:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    try:
        yield
    finally:
        for path in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def _assert_current_layout(state: dict, path: str) -> None:
    assert_current_layout(state, path, source="DDPWorldModel")


# =====================================================================
# Model Loaders
# =====================================================================


def _resolve_checkpoint_path(path_value: str) -> str:
    """Absolute path of a checkpoint given on the command line / in the config."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(_PROJECT_ROOT, path_value)


def _optional_checkpoint_layer(path_value, groups, what, base_ckpt_path):
    """Resolve an optional override checkpoint into ``(path, state, groups, raw)``, or None."""
    if not path_value or str(path_value).lower() in ("null", "none", ""):
        return None
    path = _resolve_checkpoint_path(path_value)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what}={path} does not exist. It is an optional override for the modules that "
            f"checkpoint owns; drop the argument (or set it to null) to use the ones carried by "
            f"{base_ckpt_path} instead."
        )
    data = load_checkpoint(path)
    state = data.get("predictor", data)
    _assert_current_layout(state, path)
    return path, state, groups, data


def _checkpoint_layers(cfg_dict):
    """Resolve the evaluated checkpoint plus its optional overrides, lowest priority first.

    Returns ``(base_ckpt_path, layers, encoder_state)``; a layer is
    ``(path, state_dict, group_prefixes, raw_checkpoint)``.
    """
    base_ckpt_path = _resolve_checkpoint_path(cfg_dict["ddpwm_ckpt"])
    log.info(f"[DDPWorldModel] Loading: {base_ckpt_path}")
    base = load_checkpoint(base_ckpt_path)
    base_state = base.get("predictor", base)
    _assert_current_layout(base_state, base_ckpt_path)
    layers = [(base_ckpt_path, base_state, GROUPS_LRM, base)]

    # Mid layer: a predictor checkpoint overrides the predictor (and the classifier modules it
    # carries, because the classifier modules of the base layer have lower priority).
    pred_layer = _optional_checkpoint_layer(
        cfg_dict.get("ddpwm_predictor_ckpt"),
        GROUPS_PREDICTOR,
        "ddpwm_predictor_ckpt",
        base_ckpt_path,
    )
    if pred_layer:
        layers.append(pred_layer)

    # Highest layer: an explicit classifier checkpoint overrides history fusion + localizer.
    cls_layer = _optional_checkpoint_layer(
        cfg_dict.get("ddpwm_cls_ckpt"), GROUPS_CLASSIFIER, "ddpwm_cls_ckpt", base_ckpt_path
    )
    if cls_layer:
        layers.append(cls_layer)

    return base_ckpt_path, layers, _encoder_state_by_priority(layers)


def _encoder_state_by_priority(layers) -> dict:
    """Action/proprio encoders of the layers, highest-priority layer first.

    The encoders are trained in the classifier stage and inherited by the later ones, so they follow
    the same ``classifier > predictor > LRM`` order as the modules above; the caller takes the first
    entry of each name (``layers`` is ordered lowest priority first).
    """
    encoder_state = {}
    for _path, _state, _groups, data in reversed(layers):
        for name in ("action_encoder", "proprio_encoder"):
            if name in data:
                encoder_state.setdefault(name, data[name])
    return encoder_state


def _check_evaluation_layers_are_complete(pred_state, ckpt_path) -> None:
    """Raise when some part of the inference path is missing from every checkpoint layer."""
    missing = missing_groups(pred_state, GROUPS_LRM)
    if missing:
        raise RuntimeError(
            f"the evaluated checkpoint {ckpt_path} (plus the optional overrides) has no weights "
            f"for: {', '.join(missing)}. Pass the checkpoint of the stage that trains them "
            f"(ddpwm_predictor_ckpt=<predictor.pth> / ddpwm_cls_ckpt=<classifier.pth>), or evaluate "
            f"an LRM checkpoint: checkpoints trained with this repository already contain every "
            f"earlier stage."
        )


def _check_no_random_mask_path(result, ckpt_path) -> None:
    """Raise when the mask/prediction path would stay at its random initialisation.

    The localizer produces the sparse mask and the history fusion produces the ViT's input, so both
    must come from a checkpoint. The Hungarian-matching heads (coord_head / cls_head_pred) are
    excluded on purpose: they belong to a loss the reproduction does not use, so released
    checkpoints lack them without harm.
    """
    unused_at_inference = ("primary_predictor.coord_head.", "primary_predictor.cls_head_pred.")
    uninitialised = [
        k
        for k in result.missing_keys
        if k.startswith(("localizer.", "history_fusion.", "primary_predictor."))
        and not k.startswith(unused_at_inference)
    ]
    if uninitialised:
        raise RuntimeError(
            f"{ckpt_path} does not contain {len(uninitialised)} tensors of the mask/prediction "
            f"path (e.g. {uninitialised[:3]}), so they would stay randomly initialized and the "
            f"evaluation would be meaningless. Pass the classifier checkpoint they were trained "
            f"with (`ddpwm_cls_ckpt=<classifier.pth>`); checkpoints trained with this repository "
            f"bundle them and need no extra argument."
        )


def _build_inference_model():
    """A DDPWorldModel wired for evaluation: frozen DINOv2 encoder + encoder stubs + predictor."""
    from models.ddp_world_model import DDPWorldModel
    from models.dino import DinoV2Encoder
    from models.proprio import ProprioceptiveEmbedding

    log.info("[DDPWorldModel] Building model...")
    enc = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
    for p in enc.parameters():
        p.requires_grad = False
    ae = ProprioceptiveEmbedding(in_chans=_PUSHT_ACTION_DIM, emb_dim=10, tubelet_size=1)
    pe = ProprioceptiveEmbedding(in_chans=_PUSHT_PROPRIO_RAW_DIM, emb_dim=10, tubelet_size=1)
    return DDPWorldModel(
        training_stage="inference",
        num_hist=_PUSHT_NUM_HIST,
        encoder=enc,
        action_encoder=ae,
        proprio_encoder=pe,
        action_dim=10,  # embedding dim of ProprioceptiveEmbedding(emb_dim=10)
        proprio_dim=10,
        num_action_repeat=1,
        num_proprio_repeat=1,
    )


def load_ddpwm_model(cfg_dict: dict, device):
    """Build a DDPWorldModel and load the checkpoint (and override layers) named by ``cfg_dict``."""
    model = _build_inference_model()
    ckpt_path, layers, encoder_state = _checkpoint_layers(cfg_dict)

    pred_state = merge_checkpoint_layers(
        [(path_, state_, groups_) for path_, state_, groups_, _ in layers]
    )
    log.info(
        "[DDPWorldModel] checkpoint layers (low -> high priority): "
        + " < ".join(os.path.basename(layer[0]) for layer in layers)
    )
    _check_evaluation_layers_are_complete(pred_state, ckpt_path)

    result = model.predictor.load_state_dict(pred_state, strict=False)
    loaded = len(pred_state) - len(result.unexpected_keys)
    log.info(
        f"[DDPWorldModel] {loaded}/{len(pred_state)} tensors loaded "
        f"({len(result.missing_keys)} model tensors left at init)"
    )
    if result.missing_keys and len(result.missing_keys) <= 10:
        log.info(f"  Missing: {result.missing_keys}")

    _check_no_random_mask_path(result, ckpt_path)

    for name, target in (
        ("action_encoder", model.action_encoder),
        ("proprio_encoder", model.proprio_encoder),
    ):
        if name in encoder_state:
            target.load_state_dict(encoder_state[name])
            log.info(f"[DDPWorldModel] + {name}")

    model.to(device).eval()

    # A checkpoint trained without the history fusion sees a single frame at inference time (the
    # predictor stage records this flag when it saves). The evaluated checkpoint decides; an
    # override layer that carries the flag wins, because it describes the predictor being used.
    _single_frame = next(
        (
            bool(data["history_fusion_disabled"])
            for _p, _s, _g, data in reversed(layers)
            if "history_fusion_disabled" in data
        ),
        False,
    )
    if _single_frame:
        model.history_fusion_disabled = True
        log.info("[DDPWorldModel] AUTO: history_fusion_disabled=True (from checkpoint)")

    return model


def load_dinowm_model(model_ckpt: str, train_cfg, device):
    """Load DINO-WM VWorldModel from old-format checkpoint.

    Uses dino_planning.visual_world_model.VWorldModel (local copy)
    to avoid depending on ddpwm's models/ directory.
    """
    # The old-format checkpoint pickles nn.Module objects, so torch.load needs their original class
    # definitions importable: the reference checkout (if present next to this repository) and the
    # torch.hub cache that holds the DINOv2 class.
    import torch.hub as _thub

    search_paths = [
        os.path.join(os.path.dirname(_PROJECT_ROOT), "dino_wm"),
        os.path.join(_thub.get_dir(), "facebookresearch_dinov2_main"),
    ]
    with _temporary_sys_path(*search_paths):
        log.info(f"[DINO-WM] Loading: {model_ckpt}")
        with model_ckpt.open("rb") as f:
            # weights_only=False: the payload holds pickled modules, not plain state dicts (see
            # above). PyTorch >= 2.6 defaults to weights_only=True, which rejects them.
            payload = torch.load(f, map_location=device, weights_only=False)

    ALL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]
    result = {}
    for k, v in payload.items():
        if k in ALL_KEYS:
            result[k] = v.to(device) if isinstance(v, torch.nn.Module) else v

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in checkpoint")
    # Always skip decoder for baseline evaluation (no pixel visualization needed)
    result["decoder"] = None

    # Use local VWorldModel copy to avoid modifying ddpwm's models/
    from dino_planning.visual_world_model import VWorldModel

    model = VWorldModel(
        image_size=train_cfg.get("img_size", 224),
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=train_cfg.get("num_action_repeat", 1),
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device).eval()

    log.info(f"[DINO-WM] Model loaded, epoch {payload.get('epoch', '?')}")
    return model


# =====================================================================
# PlanWorkspace
# =====================================================================


class PlanWorkspace:
    def __init__(self, cfg_dict, wm, dset, env, frameskip):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.frameskip = frameskip
        self.device = next(wm.parameters()).device

        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        self.n_evals = cfg_dict["n_evals"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict.get("debug_dset_init", False)

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        self.prepare_targets()

        # Build the objective function. `use_sparse_cost` does not change it: as in the reference
        # implementation, the CEM planner always receives the dense objective and, when the sparse
        # (masked) cost is enabled, scores the visual term on the active patches itself and uses the
        # objective only for the proprioceptive term.
        objective_fn = hydra.utils.call(cfg_dict["objective"])
        use_sparse = bool(cfg_dict.get("use_sparse_cost", False))
        log.info(
            "[SPARSE] masked cost on the patches that differ between start and goal "
            f"(threshold={cfg_dict.get('sparse_threshold', 0.1)}, "
            f"dilation={cfg_dict.get('sparse_dilation', 4)})"
            if use_sparse
            else "[DENSE] Using standard MSE cost on all patches"
        )

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
        )

        # All evaluation artifacts go to saved_folder (the per-evaluation directory); fall back to cwd
        self.saved_folder = cfg_dict.get("saved_folder") or os.getcwd()
        self.log_filename = os.path.join(self.saved_folder, "logs.json")
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            log_filename=self.log_filename,
        )

        from dino_planning.mpc import MPCPlanner

        if isinstance(self.planner, MPCPlanner):
            # The masked-cost switch belongs to the CEM sub-planner (it owns the patch mask), so it
            # is set here rather than in the YAML planner block: one flag (`use_sparse_cost`) drives
            # both the log line above and the sub-planner.
            self.planner.sub_planner.use_masked_heatmap = use_sparse
            self.planner.sub_planner.pixel_diff_thresh = cfg_dict.get("sparse_threshold", 0.1)
            self.planner.sub_planner.mask_dilation = cfg_dict.get("sparse_dilation", 4)
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        # Debug/eval aid: plan and report only the listed episode indices of the sampled batch.
        # The CEM still draws its candidate noise for every episode, so the selected ones reproduce
        # a full run exactly while the rest is skipped (see CEMPlanner.eval_episodes).
        eval_episodes = cfg_dict.get("eval_episodes")
        if isinstance(eval_episodes, str):
            # `cfg_to_dict` joins list values into "a,b" strings (legacy behaviour of the original
            # config plumbing), so accept both "[27,30]" and "27,30".
            eval_episodes = [
                int(x)
                for x in eval_episodes.strip("[] ").replace(" ", "").split(",")
                if x not in ("", "None", "null")
            ]
        if eval_episodes:
            selected = [int(i) for i in eval_episodes]
            if isinstance(self.planner, MPCPlanner):
                self.planner.sub_planner.eval_episodes = selected
            self.evaluator.report_indices = selected
            log.info(
                "[SUBSET] planning only episodes %s of %d; per-episode metrics below are only "
                "meaningful for those indices",
                selected,
                self.n_evals,
            )

        # Optional independent knob for the planning noise: the scenes above were sampled with
        # `seed`, and everything the CEM draws afterwards comes from torch's global RNG. Re-seeding
        # here keeps the scenes identical while making the planning draw selectable/reproducible.
        plan_rng_seed = cfg_dict.get("plan_rng_seed")
        if plan_rng_seed not in (None, "", "None", "null"):
            plan_rng_seed = int(plan_rng_seed)
            torch.manual_seed(plan_rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(plan_rng_seed)
            log.info(
                "[PLAN-RNG] planning noise re-seeded with %d (scenes still from seed=%s)",
                plan_rng_seed,
                cfg_dict.get("seed"),
            )

        self.dump_targets()

    def prepare_targets(self):
        # Goals come from the validation split (the reference's `goal_source` switch is not ported).
        observations, states, actions, env_info = self.sample_traj_segment_from_dset(
            traj_len=self.frameskip * self.goal_H + 1
        )
        self.env.update_env(env_info)
        init_state = np.array([x[0] for x in states])
        actions = torch.stack(actions)
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
        exec_actions = self.data_preprocessor.denormalize_actions(actions)
        rollout_obses, rollout_states = self.env.rollout(
            self.eval_seed, init_state, exec_actions.numpy()
        )
        self.obs_0 = {key: np.expand_dims(arr[:, 0], axis=1) for key, arr in rollout_obses.items()}
        self.obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
        self.state_0 = init_state
        self.state_g = rollout_states[:, -1]
        self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states, actions, observations, env_info = [], [], [], []
        for _ in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:
                traj_id = random.randint(0, len(self.dset) - 1)
                # Handle both 3-element (ddpwm) and 4-element (dino_wm) returns
                result = self.dset[traj_id]
                if len(result) == 4:
                    obs, act, state, e_info = result
                else:
                    obs, act, state = result[:3]
                    e_info = {"shape": "T"}
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy() if hasattr(state, "numpy") else state
            offset = random.randint(0, max_offset)
            obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def dump_targets(self):
        # Sampled scenes are written into this evaluation's own directory, so concurrent
        # evaluations cannot overwrite each other (they used to be written to the repository root).
        with open(os.path.join(self.saved_folder, "targets.pkl"), "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )

    def perform_planning(self):
        actions_init = self.gt_actions if self.debug_dset_init else None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(actions.detach(), action_len)
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        logs_entry = {
            key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value)
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


# =====================================================================
# Main
# =====================================================================


def resolve_data_path(cfg_dict: dict) -> str:
    """Locate the PushT dataset directory used for planning/evaluation.

    Precedence: `data_path=<dir>` on the command line / in the config, then `$DDPWM_PUSHT_DATA`
    (the PushT directory itself), then `$DDPWM_DATA_DIR/pusht_noise` -- the same root variable the
    trainer takes via `--data-dir`, so a normal setup needs only one environment variable.
    """
    data_path = os.environ.get("DDPWM_PUSHT_DATA") or cfg_dict.get("data_path")
    if not data_path and os.environ.get("DDPWM_DATA_DIR"):
        candidate = os.path.join(os.environ["DDPWM_DATA_DIR"], "pusht_noise")
        if os.path.isdir(candidate):
            data_path = candidate
    if not data_path:
        raise ValueError(
            "data_path is not set: pass data_path=<dir> on the command line, or export "
            "DDPWM_PUSHT_DATA=<.../pusht_noise>, or export DDPWM_DATA_DIR=<dataset root> "
            "containing pusht_noise/"
        )
    return str(data_path)


def build_model_and_dataset(cfg_dict: dict, device) -> tuple:
    """Build the world model and the validation trajectory dataset for the requested model type.

    Returns (model, dset, frameskip).
    """
    model_type = cfg_dict.get("model_type", "ddpwm")
    from dsets.img_transforms import default_transform
    from dsets.pusht_dset import load_pusht_slice_train_val

    transform = default_transform(img_size=224)
    data_path = resolve_data_path(cfg_dict)

    if model_type == "ddpwm":
        model = load_ddpwm_model(cfg_dict, device)
        _, traj_dset = load_pusht_slice_train_val(
            data_path=data_path,
            with_velocity=True,
            n_rollout=None,
            normalize_action=True,
            transform=transform,
            num_hist=cfg_dict["goal_H"],
            num_pred=1,
            frameskip=5,
        )
        return model, traj_dset["valid"], 5

    if model_type == "dinowm":
        # Load the DINO-WM baseline from an original checkpoint
        model_dir = Path(f"{cfg_dict['ckpt_base_path']}/outputs/{cfg_dict['model_name']}/")
        model_cfg = OmegaConf.load(model_dir / "hydra.yaml")
        # The configuration stored next to such a checkpoint resolves its dataset through
        # `${oc.env:DATASET_DIR}/<env>` (see dino_wm/conf/env/*.yaml), and the reference code itself
        # reads the environment variable (it has no parameter for the path). Both are handled
        # explicitly here: the variable is exported only when the caller has not set it, and the
        # loaded config is pinned to the same path our own loader below uses.
        os.environ.setdefault("DATASET_DIR", os.path.dirname(str(data_path)))
        if model_cfg.get("env", {}).get("dataset", {}).get("data_path") is not None:
            model_cfg["env"]["dataset"]["data_path"] = str(data_path)
        log.info(
            f"[DINO-WM] dataset pinned to {data_path} "
            f"(DATASET_DIR={os.environ['DATASET_DIR']}, read by the reference code)"
        )
        model_ckpt = model_dir / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
        model = load_dinowm_model(model_ckpt, model_cfg, device)
        # The original model attends over 588 tokens (3 frames x 196 patches) with batch=300,
        # which needs ~6.6 GB for the attention scores alone (>the 11 GB of a 2080 Ti).
        # Chunking the CEM rollouts keeps memory in bounds; candidates are independent, so the
        # result is unchanged.
        model._rollout_chunk_size = 50
        _, traj_dset = load_pusht_slice_train_val(
            data_path=data_path,
            with_velocity=True,
            n_rollout=None,
            normalize_action=True,
            transform=transform,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        return model, traj_dset["valid"], int(model_cfg.frameskip)

    raise ValueError(f"Unknown model_type: {model_type}")


def build_env(cfg_dict: dict):
    """Create one Pusht environment per evaluation scene."""
    # __import__('env') runs env/__init__.py, which registers "pusht" in gym
    __import__("env")
    from dino_planning.vector_env import SerialVectorEnv

    return SerialVectorEnv(
        [
            gym.make("pusht", with_velocity=True, with_target=True)
            for _ in range(cfg_dict["n_evals"])
        ]
    )


def planning_main(cfg_dict: dict) -> None:
    import time as _time

    _t_start = _time.perf_counter()
    output_dir = cfg_dict.get("saved_folder", os.getcwd())
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"[EVAL-DIR] {output_dir}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_device(log, device, "planning")

    if cfg_dict.get("deterministic"):
        # Same knobs as `train_ddpwm.py --deterministic`, so an evaluation can be compared
        # bit-for-bit across code revisions (see tests/README.md).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        log.info("[DETERMINISTIC] cudnn.deterministic=True, benchmark=False, warn_only=True")

    seed(cfg_dict["seed"])

    # 1. Load model and dataset
    model, dset, frameskip = build_model_and_dataset(cfg_dict, device)

    # 2. Create environment
    env = build_env(cfg_dict)

    # 3. Plan
    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        frameskip=frameskip,
    )
    logs = plan_workspace.perform_planning()
    # Persist a structured result (equivalent to the RESULT line in eval.log, easier to parse)
    try:
        result = {
            "eval_dir": output_dir,
            "seed": cfg_dict.get("seed"),
            "n_evals": cfg_dict.get("n_evals"),
            "goal_H": cfg_dict.get("goal_H"),
            "use_sparse_cost": cfg_dict.get("use_sparse_cost"),
            "ddpwm_ckpt": cfg_dict.get("ddpwm_ckpt"),
            "ddpwm_cls_ckpt": cfg_dict.get("ddpwm_cls_ckpt"),
            "model_type": cfg_dict.get("model_type"),
            "logs": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in logs.items()
            },
            "wall_time_sec": round(_time.perf_counter() - _t_start, 1),
        }
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:  # a failed write must not break the run
        log.warning(f"[result.json] write failed: {e}")
    log.info(
        f"[TIMING] total wall time {_time.perf_counter() - _t_start:.1f}s "
        f"({(_time.perf_counter() - _t_start)/60:.1f} min) for {cfg_dict['n_evals']} evals, "
        f"seed={cfg_dict['seed']}, sparse={cfg_dict.get('use_sparse_cost')}, "
        f"ckpt={cfg_dict.get('ddpwm_ckpt')}"
    )
    log.info(f"\n{'='*60}")
    log.info(f"  RESULT: {logs}")
    log.info(f"{'='*60}")
    return logs


@hydra.main(config_path="../conf", config_name="plan_pusht", version_base=None)
def main(cfg: OmegaConf) -> None:
    with open_dict(cfg):
        # Paths are resolved *before* chdir: every relative path in the config is relative to the
        # directory the command was launched from, and an evaluation must not change what its own
        # inputs mean just because it moved into its output directory.
        for key in ("data_path", "ckpt_base_path"):
            if cfg.get(key):
                cfg[key] = os.path.abspath(cfg[key])
        eval_dir = cfg.get("eval_dir")
        if eval_dir:
            # every relative-path artifact (targets.pkl / logs.json / result.json) lands here
            os.makedirs(eval_dir, exist_ok=True)
            eval_dir = os.path.abspath(eval_dir)
            os.chdir(eval_dir)
            cfg["saved_folder"] = eval_dir
        else:
            cfg["saved_folder"] = os.getcwd()
    cfg_dict = cfg_to_dict(cfg)
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
