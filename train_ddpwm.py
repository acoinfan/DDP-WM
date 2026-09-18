# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""
DDP-WM Training Launcher with Stage Selection.

Self-contained training loop that does NOT depend on orgtrain.py's Trainer class.
Directly uses Accelerate for distributed training.

Usage:
    # Stage 1: Train classifier (Dynamic Localization Network)
    python train_ddpwm.py --stage classifier --epochs 2 --env pusht

    # Stage 2: Train predictor (requires classifier checkpoint)
    python train_ddpwm.py --stage predictor \
        --classifier-ckpt checkpoints/classifier/model_latest.pth --epochs 2

    # Stage 3: Train LRM (requires predictor checkpoint)
    python train_ddpwm.py --stage lrm \
        --predictor-ckpt checkpoints/predictor/model_latest.pth --epochs 2
"""

import os
import signal
import sys

# Ensure project root is in Python path regardless of launch method
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

# NCCL safety net for the older GPUs this reproduction was verified on: on 2080 Ti / Titan Xp with
# PCIe-only P2P (no NVLink) NCCL can SIGSEGV while building the rings. Disabling P2P and IB costs
# some bandwidth on machines that do have NVLink/IB, so both are overridable from the environment:
#     NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 python -m torch.distributed.run ... train_ddpwm.py ...
# They must be set before torch initialises NCCL, hence the module-level placement.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

from accelerate import Accelerator

from common.runtime import log_device
from models.ckpt_io import load_checkpoint
from models.ckpt_layout import (
    GROUPS_CLASSIFIER,
    assert_current_layout,
    groups_for_stage,
    merge_checkpoint_layers,
    missing_groups,
)

log = logging.getLogger(__name__)

# Log line format of this entry point. Deliberately the same as the one hydra installs for
# `python -m dino_planning.plan`, so the training and evaluation logs of a reproduction can be read
# (and parsed) the same way.
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the process-wide logging of a training run.

    Called from `main()` rather than at import time: importing this module (the tests do) must not
    change the logging configuration of the importing process. `force=True` makes repeated calls
    (and a surrounding framework) deterministic instead of stacking handlers.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)


# Graceful shutdown: SIGTERM/SIGINT set this flag; the loop checks it after each batch.
_shutdown_requested = False


def _handle_shutdown_signal(signum: int, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    log.info(f"\n[SIGNAL] Received signal {signum}. Will save and exit after current batch.")


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config file. Returns dict or empty dict if not found."""
    if config_path and os.path.exists(config_path):
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP-WM Staged Training")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["classifier", "predictor", "lrm"],
        help='Training stage. "predictor" trains the world model (see '
        '--history-fusion-disabled); "classifier" and "lrm" have a single '
        "recipe each.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="YAML config file path (default: conf/ddpwm.yaml)"
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default=None,
        help="Classifier checkpoint path (for the predictor/lrm stages)",
    )
    parser.add_argument(
        "--predictor-ckpt", type=str, default=None, help="Predictor checkpoint path (for lrm)"
    )
    parser.add_argument(
        "--lrm-ckpt",
        type=str,
        default=None,
        help="LRM checkpoint path. Loads only lrm.* keys for continuing LRM training.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (default 0). Controls the data-slice permutation (numpy) and the "
        "model initialisation (torch).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=f"Environment / dataset name, one of {available_envs()} "
        f"(each maps to conf/env/<env>.yaml). Defaults to env.name in the "
        f"run config, then common.env in conf/ddpwm.yaml.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="deformable_env only: rope / granular (default: rope)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for checkpoints"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=None,
        help="Write one training-CSV row every N steps (default logging.log_every_n_steps = 10)",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=None,
        help="Save model_<N>.pth / model_latest.pth at the end of every N epochs "
        "(default logging.save_every_n_epochs = 1; 0 = never at epoch end). "
        "Mid-epoch resumable saves use --save-every-n-steps.",
    )
    parser.add_argument(
        "--eval-ckpt",
        type=str,
        default=None,
        help="Path to checkpoint for eval-only mode. Skips training, just computes val_loss.",
    )
    # The predictor is the only stage with more than one recipe. The two recipes differ solely in how
    # much history is fed to the ViT, so they are a boolean instead of two stage names (the second
    # recipe is a warm-start trick of the reproduction and is not part of the paper).
    parser.add_argument(
        "--history-fusion-disabled",
        action="store_true",
        default=False,
        help="Train the predictor WITHOUT the frozen 3-frame history fusion inherited "
        "from --classifier-ckpt: the ViT sees a single frame (this is the recipe that "
        "bootstraps the ViT from a random initialisation). Without this flag the "
        "frozen history fusion is applied first, i.e. the predictor consumes 3 frames. "
        "Both recipes share the same loss, rollout and per-history-length gradients.",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=0,
        help="Save resumable checkpoint every N steps. 0 = disabled.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from mid-epoch checkpoint. --lr overrides saved lr.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Make the run as bit-reproducible as torch allows (cuDNN deterministic kernels, "
        "torch.use_deterministic_algorithms in warn_only mode, fixed cuBLAS workspace). Needed to "
        "compare two code revisions on the same machine; slightly slower. A few LRM ops have no "
        "deterministic kernel, so that stage keeps a ~1e-5 spread.",
    )

    return parser.parse_args()


@dataclass
class TrainConfig:
    """Everything a training run needs: CLI flags plus the values resolved from the YAML config.

    Built once by :func:`resolve_config`; the rest of the file only reads this object (there is no
    hidden global or mutated ``argparse`` namespace).
    """

    # --- resolved from the YAML config (common < stage section < CLI) ---
    stage: str
    env: str
    config_path: str
    env_override: dict[str, Any] | None
    data_dir: str | None
    batch_size: int
    lr: float
    epochs: int
    num_hist: int
    frameskip: int
    num_workers: int
    output_dir: str
    classifier_ckpt: str | None
    predictor_ckpt: str | None
    log_every: int
    save_every_n_epochs: int
    # --- CLI-only flags ---
    seed: int = 0
    object_name: str | None = None
    lrm_ckpt: str | None = None
    eval_ckpt: str | None = None
    resume: str | None = None
    save_every_n_steps: int = 0
    history_fusion_disabled: bool = False
    deterministic: bool = False


def resolve_config(args) -> TrainConfig:
    """Merge YAML defaults < stage-specific section < CLI overrides into a TrainConfig."""
    # 1. Load YAML config
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, "conf", "ddpwm.yaml")
    yaml_cfg = load_yaml_config(config_path)

    # 2. Extract common and stage-specific sections
    common = yaml_cfg.get("common", {})
    stage_cfg = yaml_cfg.get(args.stage, {})
    logging_cfg = yaml_cfg.get("logging", {})
    # Every run config must carry the section of the stage it starts: falling back to the built-in
    # defaults would silently train with the wrong window (e.g. num_hist=3 instead of 5 for the
    # predictor) or the wrong lr. This also rejects configs written before the CLI cleanup, which
    # name the predictor stage 'primary_predictor', and the legacy hydra template.
    if not stage_cfg:
        sections = [k for k in yaml_cfg if isinstance(yaml_cfg[k], dict)]
        hint = (
            "  (it uses the old section name 'primary_predictor'; rename it to " f"'{args.stage}')"
            if "primary_predictor" in yaml_cfg
            else ""
        )
        sys.exit(
            f"ERROR: {config_path} has no '{args.stage}' section; found {sections}.\n"
            f"       Every stage needs its own section (see conf/ddpwm.yaml).{hint}"
        )

    # 3. Build resolved config (common < stage < CLI) with type enforcement
    # Optional top-level `env:` block of a run config: overrides keys of conf/env/<env>.yaml
    env_block = yaml_cfg.get("env") or {}
    env_override = env_block if isinstance(env_block, dict) else {}
    cfg = TrainConfig(
        stage=args.stage,
        env=args.env
        or env_override.get("name")
        or stage_cfg.get("env")
        or common.get("env", "pusht"),
        env_override=env_override,
        # Dataset root: --data-dir > $DDPWM_DATA_DIR > common.data_dir (conf/ddpwm.yaml) >
        # env.data_root (conf/env/<env>.yaml); build_dataset applies that order.
        # If none is set, build_dataset falls back to the per-environment default in
        # conf/env/<env>.yaml (data_root), so this may legitimately stay None.
        data_dir=(args.data_dir or os.environ.get("DDPWM_DATA_DIR") or common.get("data_dir")),
        # Numeric CLI options are compared with `None`, not with `or`: 0 is a legitimate value
        # (`--num-workers 0` = no DataLoader worker processes, `--epochs 0` = train nothing,
        # `--lr 0` = freeze) and `or` would silently fall back to the YAML default.
        batch_size=int(
            args.batch_size
            if args.batch_size is not None
            else (stage_cfg.get("batch_size") or common.get("batch_size", 64))
        ),
        lr=float(args.lr if args.lr is not None else stage_cfg.get("lr", 7e-4)),
        epochs=int(args.epochs if args.epochs is not None else stage_cfg.get("epochs", 2)),
        num_hist=int(stage_cfg.get("num_hist", 3)),
        frameskip=int(common.get("frameskip", 5)),
        num_workers=int(
            args.num_workers if args.num_workers is not None else common.get("num_workers", 6)
        ),
        output_dir=args.output_dir or os.path.join("checkpoints", args.stage),
        classifier_ckpt=args.classifier_ckpt or stage_cfg.get("classifier_ckpt"),
        predictor_ckpt=args.predictor_ckpt or stage_cfg.get("predictor_ckpt"),
        log_every=int(
            args.log_every_n_steps
            if args.log_every_n_steps is not None
            else logging_cfg.get("log_every_n_steps", 10)
        ),
        save_every_n_epochs=int(
            args.save_every_n_epochs
            if args.save_every_n_epochs is not None
            else logging_cfg.get("save_every_n_epochs", 1)
        ),
        config_path=config_path,
        seed=int(getattr(args, "seed", None) or 0),  # `--seed 0` and the default are the same here
        object_name=getattr(args, "object_name", None),
        lrm_ckpt=getattr(args, "lrm_ckpt", None),
        eval_ckpt=getattr(args, "eval_ckpt", None),
        resume=getattr(args, "resume", None),
        save_every_n_steps=int(getattr(args, "save_every_n_steps", 0) or 0),
        history_fusion_disabled=bool(getattr(args, "history_fusion_disabled", False)),
        deterministic=bool(getattr(args, "deterministic", False)),
    )
    return cfg


# Stage configs are now read from YAML via resolve_config()


# ============================================================
# Dataset wiring: conf/env/<env>.yaml is the single source of truth (one file per environment).
#   * a run config can override individual keys through its top-level `env:` block;
#   * --data-dir / --object-name override the YAML afterwards.
# Adding an environment = adding conf/env/<env>.yaml; nothing here has to change.
# ============================================================
ENV_CONF_DIR = os.path.join(_PROJECT_ROOT, "conf", "env")


def available_envs() -> list[str]:
    """Environment names found in conf/env/*.yaml (= the environments currently supported)."""
    if not os.path.isdir(ENV_CONF_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(ENV_CONF_DIR) if f.endswith((".yaml", ".yml"))
    )


def load_env_config(env_name: str, override: dict | None = None):
    """Read conf/env/<env>.yaml; the run config's `env:` block acts as an override layer."""
    from omegaconf import OmegaConf

    path = os.path.join(ENV_CONF_DIR, f"{env_name}.yaml")
    if not os.path.exists(path):
        raise ValueError(
            f"Unknown env '{env_name}': {path} does not exist. " f"Available: {available_envs()}"
        )
    cfg = OmegaConf.load(path)
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override))
    return cfg


def check_ckpt_layout(state: dict, path: str) -> None:
    """Raise a clear error when a checkpoint still uses a legacy key layout."""
    assert_current_layout(state, path, source="train_ddpwm")


def _import_loader(dotted: str) -> Callable:
    """'dsets.pusht_dset:load_pusht_slice_train_val' -> the loader function object."""
    import importlib

    if ":" in dotted:
        mod_name, fn_name = dotted.split(":")
    else:  # also accept the 'pkg.mod.func' spelling
        mod_name, fn_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ImportError(
            f"dataset loader '{dotted}' cannot be imported ({exc}). The environment config points "
            f"at a module that is not part of this repository -- this is typical for a config "
            f"copied from an older pipeline. Use conf/env/<env>.yaml as the template."
        ) from exc
    return getattr(module, fn_name)


def build_dataset(cfg: TrainConfig, num_hist: int, frameskip: int) -> dict:
    """Build the datasets described by conf/env/<env>.yaml; stage-specific values are injected here."""
    from omegaconf import OmegaConf

    from dsets.img_transforms import default_transform

    env_cfg = load_env_config(cfg.env, cfg.env_override)

    candidates = []
    subdir = env_cfg.get("data_subdir")
    if cfg.data_dir and subdir:
        candidates.append(os.path.join(cfg.data_dir, subdir))
    if env_cfg.get("data_path"):  # the YAML already gives a full path
        candidates.append(env_cfg.data_path)
    elif subdir and env_cfg.get("data_root"):  # data_root/data_subdir
        candidates.append(os.path.join(env_cfg.data_root, subdir))

    data_path = next((p for p in candidates if os.path.isdir(p)), None)
    if data_path is None:
        tried = candidates if candidates else "(nothing: no data root is configured)"
        raise FileNotFoundError(
            f"[{cfg.env}] dataset directory not found; tried: {tried}.\n"
            f"  Pass --data-dir <root> (or export $DDPWM_DATA_DIR), or set data_root in "
            f"conf/env/{cfg.env}.yaml; the datasets are not shipped with this repository."
        )
    if len(candidates) > 1 and data_path != candidates[0]:
        log.info(f"[{cfg.env}] using {data_path} (nothing found under --data-dir)")

    ds_kwargs = OmegaConf.to_container(env_cfg.get("dataset", {}), resolve=True) or {}
    target = ds_kwargs.pop("_target_", None)
    if not target:
        raise ValueError(f"[{cfg.env}] conf/env/{cfg.env}.yaml has no dataset._target_")

    if cfg.object_name:
        if "object_name" in ds_kwargs:  # only loaders such as deformable_env accept this
            ds_kwargs["object_name"] = cfg.object_name
        else:
            log.warning(f"[{cfg.env}] --object-name ignored: this loader does not accept it")
    ds_kwargs.update(
        data_path=data_path,
        transform=default_transform(img_size=int(env_cfg.get("img_size", 224))),
        num_hist=num_hist,
        num_pred=1,
        frameskip=frameskip,
    )

    loader = _import_loader(target)
    log.info(
        f"[{cfg.env}] env_config=conf/env/{cfg.env}.yaml loader={target} data_path={data_path}"
    )
    log.info(
        f"[{cfg.env}] dataset kwargs: "
        f"{ {k: v for k, v in ds_kwargs.items() if k != 'transform'} }"
    )
    datasets, traj_dsets = loader(**ds_kwargs)
    return datasets


def build_model(cfg: TrainConfig, datasets: dict):
    """Build the DDPWorldModel with all required components."""
    from models.ddp_world_model import DDPWorldModel
    from models.dino import DinoV2Encoder
    from models.proprio import ProprioceptiveEmbedding

    # Read actual dimensions from dataset (PushT with frameskip=5: action=10, proprio=20)
    action_dim = datasets["train"].action_dim
    proprio_dim = datasets["train"].proprio_dim
    log.info(f"Dataset dims: action_dim={action_dim}, proprio_dim={proprio_dim}")

    # Encoder (frozen DINOv2)
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
    for p in encoder.parameters():
        p.requires_grad = False

    # Action/Proprio encoders - in_chans must match dataset dimensions
    action_encoder = ProprioceptiveEmbedding(in_chans=action_dim, emb_dim=10, tubelet_size=1)
    proprio_encoder = ProprioceptiveEmbedding(in_chans=proprio_dim, emb_dim=10, tubelet_size=1)

    # World model
    model = DDPWorldModel(
        image_size=224,
        num_hist=cfg.num_hist,
        training_stage=cfg.stage,
        encoder=encoder,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        classifier_ckpt=cfg.classifier_ckpt,
        predictor_ckpt=cfg.predictor_ckpt,
        cls_pos_weight=10.0,
        num_action_repeat=1,
        num_proprio_repeat=1,
        proprio_dim=10,  # embedding dim after ProprioceptiveEmbedding
        action_dim=10,  # embedding dim after ProprioceptiveEmbedding
    )
    return model


def get_trainable_params(model) -> list[torch.nn.Parameter]:
    """Get only the parameters that should be trained for the current stage."""
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    return params


def save_checkpoint(
    model,
    epoch: int,
    output_dir: str,
    step: int | None = None,
    epoch_step: int | None = None,
    optimizer=None,
) -> None:
    """Save model checkpoint. Compatible with next-stage loading and mid-epoch resume.

    Args:
        model: Unwrapped DDPWorldModel instance.
        epoch: Current epoch number (1-based).
        output_dir: Directory to save checkpoints.
        step: Global training step (mid-epoch save only). None for epoch-boundary.
        epoch_step: Batch index within current epoch (mid-epoch save only).
        optimizer: Optimizer to save state dict (mid-epoch save only).
    """
    os.makedirs(output_dir, exist_ok=True)

    predictor_state = model.predictor.state_dict()

    ckpt = {
        "epoch": epoch,
        "predictor": predictor_state,
        "action_encoder": model.action_encoder.state_dict(),
        "proprio_encoder": model.proprio_encoder.state_dict(),
        # Records whether this predictor was trained without history fusion (single-frame input) so
        # that a later evaluation can reproduce the matching forward path.
        "history_fusion_disabled": getattr(model, "history_fusion_disabled", False),
    }

    # Mid-epoch saves carry the state needed to resume.
    if step is not None:
        ckpt["step"] = step
    if epoch_step is not None:
        ckpt["epoch_step"] = epoch_step
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    def _safe_save(data, target):
        """Atomic write: write to .tmp then rename; an interrupted save leaves the target intact."""
        tmp = target + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, target)

    path_latest = os.path.join(output_dir, "model_latest.pth")
    _safe_save(ckpt, path_latest)

    if step is not None:
        path_step = os.path.join(output_dir, f"model_epoch{epoch}_step{step}.pth")
        _safe_save(ckpt, path_step)
        log.info(f"Checkpoint: epoch={epoch} step={step} -> {path_latest}")
    else:
        path_epoch = os.path.join(output_dir, f"model_{epoch}.pth")
        _safe_save(ckpt, path_epoch)
        log.info(f"Checkpoint: epoch={epoch} -> {path_latest}")


def log_stage_recipe(cfg: TrainConfig, accelerator) -> None:
    """Log which recipe the requested stage runs. Nothing is configured here.

    Every stage has exactly one recipe, and the single CLI switch of the whole pipeline
    (--history-fusion-disabled, predictor stage) is applied where the model is built.

    classifier : joint history fusion + localizer + encoders, BCE(pos_weight=10) on 2x2 sub-patches,
                 gradients accumulated per history length (h=1..3)
    predictor  : 32-token ViT applied to the K=32 classifier-masked foreground tokens, MSE over all
                 196 positions against the classifier-masked next-frame features, GT-filled rollout
                 (teacher forcing) and per-history-length gradients (h=1..5). Two variants:
                   * default                     -> the frozen 3-frame history fusion inherited from
                                                    the classifier checkpoint is applied first
                   * --history-fusion-disabled   -> a single frame is fed to the ViT
    lrm        : frozen backbone + one cross-attention layer, self-predicted rollout (a single real
                 frame per step, classifier mask binarised before dilate(4)), per-h gradients
    """
    if cfg.stage == "classifier":
        if accelerator.is_main_process:
            log.info(
                "Mode: classifier - joint history_fusion+localizer+encoders, "
                "BCE(pos_weight=10) on 2x2 sub-patches, per-h gradients h=1..3"
            )

    elif cfg.stage == "predictor":
        if accelerator.is_main_process:
            variant = (
                "single frame, history fusion disabled"
                if cfg.history_fusion_disabled
                else "frozen 3-frame history fusion"
            )
            log.info(
                f"Mode: predictor ({variant}) - 32-token ViT + classifier-masked MSE + "
                f"GT-filled rollout (teacher forcing) + per-h gradients"
            )

    elif cfg.stage == "lrm":
        if accelerator.is_main_process:
            log.info(
                "Mode: LRM - frozen backbone + single cross-attention layer, self-predicted "
                "rollout (single real frame per step, binarise-then-dilate mask), per-h gradients"
            )
    else:
        raise ValueError(f"Unknown stage: {cfg.stage}")


def run_eval_only(cfg: TrainConfig, model, val_loader, accelerator) -> None:
    """--eval-ckpt mode: load a checkpoint and report the validation loss (per history length for
    the predictor stage, single forward for the others). Returns after printing the result."""
    log.info(f"=== EVAL-ONLY MODE: loading {cfg.eval_ckpt} ===")
    ckpt = load_checkpoint(cfg.eval_ckpt)
    unwrapped = accelerator.unwrap_model(model)
    # The stage recipe (apply_recipe_flags) is applied before this function is called, so the
    # forward path used here matches the one the stage was trained with.
    log.info(f"[EVAL-ONLY] recipe flags applied for --stage {cfg.stage}")
    # The checkpoint records whether its predictor was trained without the history fusion; that
    # decision belongs to the checkpoint, so it overrides the command line here.
    if "history_fusion_disabled" in ckpt:
        _ckpt_single_frame = bool(ckpt["history_fusion_disabled"])
        if _ckpt_single_frame != bool(getattr(unwrapped, "history_fusion_disabled", False)):
            log.info(
                f"[EVAL-ONLY] history_fusion_disabled={_ckpt_single_frame} (from checkpoint): "
                f"the predictor now sees "
                f"{'a single frame' if _ckpt_single_frame else '3 frames'}"
            )
        unwrapped.history_fusion_disabled = _ckpt_single_frame

    # Checkpoint layers, lowest priority first: the evaluated checkpoint, then --classifier-ckpt as
    # an override. This is the same rule the planner uses (`models/ckpt_layout.py`), so `--eval-ckpt`
    # and `python -m dino_planning.plan` cannot disagree about which weights win.
    base_state = dict(ckpt.get("predictor", ckpt))
    check_ckpt_layout(base_state, cfg.eval_ckpt)
    layers = [(cfg.eval_ckpt, base_state, groups_for_stage(cfg.stage), ckpt)]
    if cfg.classifier_ckpt:
        if not os.path.exists(cfg.classifier_ckpt):
            raise FileNotFoundError(f"--classifier-ckpt={cfg.classifier_ckpt} does not exist")
        cls_ckpt_data = load_checkpoint(cfg.classifier_ckpt)
        cls_pred = cls_ckpt_data.get("predictor", cls_ckpt_data)
        check_ckpt_layout(cls_pred, cfg.classifier_ckpt)
        layers.append((cfg.classifier_ckpt, cls_pred, GROUPS_CLASSIFIER, cls_ckpt_data))
        log.info(f"  + classifier layer {cfg.classifier_ckpt} overrides history fusion/localizer")

    predictor_state = merge_checkpoint_layers(
        [(path, state, groups) for path, state, groups, _ in layers]
    )
    absent = missing_groups(predictor_state, groups_for_stage(cfg.stage))
    if absent:
        raise RuntimeError(
            f"{cfg.eval_ckpt} (plus --classifier-ckpt / --predictor-ckpt) has no weights for: "
            f"{', '.join(absent)}. Pass the checkpoint of the stage that trains them."
        )
    missing, unexpected = unwrapped.predictor.load_state_dict(predictor_state, strict=False)
    log.info(
        f"Predictor: {len(predictor_state) - len(unexpected)}/{len(predictor_state)} tensors loaded "
        f"({len(missing)} model tensors left at init)"
    )

    # Encoders follow the same priority as the modules (classifier > evaluated checkpoint).
    encoder_state = {}
    for _path, _state, _groups, data in reversed(layers):
        for name in ("action_encoder", "proprio_encoder"):
            if name in data:
                encoder_state.setdefault(name, data[name])
    for name, target in (
        ("action_encoder", unwrapped.action_encoder),
        ("proprio_encoder", unwrapped.proprio_encoder),
    ):
        if name in encoder_state:
            target.load_state_dict(encoder_state[name])
            log.info(f"  + {name}")

    # === Stage-aware validation ===
    model.eval()
    val_losses = []
    h_losses_sum = {f"h{h}": 0.0 for h in range(1, 6)}
    h_counts = {f"h{h}": 0 for h in range(1, 6)}

    if cfg.stage == "predictor":
        # Per-h evaluation (matching predictor training loop)
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc="Eval [predictor per-h]", disable=not accelerator.is_main_process
            ):
                obs, act, state = batch
                T_full = obs["visual"].shape[1]
                batch_loss = 0.0
                n_pos = 0
                for hist_len in range(1, T_full):
                    T_slice = hist_len + 1
                    obs_slice = {k: v[:, :T_slice] for k, v in obs.items()}
                    act_slice = act[:, :T_slice]
                    _, _, _, loss_h, _ = model(obs_slice, act_slice)
                    val = accelerator.gather(loss_h).mean().item()
                    batch_loss += val
                    n_pos += 1
                    h_losses_sum[f"h{hist_len}"] += val
                    h_counts[f"h{hist_len}"] += 1
                val_losses.append(batch_loss / max(n_pos, 1))
    else:
        # LRM/other: single forward on full window (matching LRM training)
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="Eval [LRM single-forward]",
                disable=not accelerator.is_main_process,
            ):
                obs, act, state = batch
                _, _, _, loss, _ = model(obs, act)
                val_losses.append(accelerator.gather(loss).mean().item())

    avg_val = sum(val_losses) / len(val_losses) if val_losses else 0
    if accelerator.is_main_process:
        log.info(f"\n{'='*60}")
        log.info(f"  EVAL RESULT: val_loss = {avg_val:.6f}")
        if cfg.stage == "predictor":
            for h in range(1, 6):
                key = f"h{h}"
                if h_counts[key] > 0:
                    log.info(f"  {key}_loss = {h_losses_sum[key] / h_counts[key]:.6f}")
        log.info(f"{'='*60}\n")


def validate_epoch(model, val_loader, accelerator, epoch: int, epochs: int) -> float:
    """Run the validation loop for one epoch and return the mean loss."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{epochs} [val]",
            disable=not accelerator.is_main_process,
        ):
            obs, act, state = batch
            _, _, _, loss, _ = model(obs, act)
            val_losses.append(accelerator.gather(loss).mean().item())
    return sum(val_losses) / len(val_losses) if val_losses else 0.0


def load_warm_start_checkpoints(cfg: TrainConfig, model, accelerator, device) -> None:
    """Load the optional --predictor-ckpt (warm start + encoders) and --lrm-ckpt (lrm.* only)."""
    if not cfg.resume and cfg.predictor_ckpt and cfg.stage in ["predictor", "lrm"]:
        ckpt = load_checkpoint(cfg.predictor_ckpt)
        raw = ckpt.get("predictor", ckpt)
        check_ckpt_layout(raw, cfg.predictor_ckpt)
        unwrapped = accelerator.unwrap_model(model)
        missing, unexpected = unwrapped.predictor.load_state_dict(raw, strict=False)
        if accelerator.is_main_process:
            # "loaded" = tensors we supplied that the model accepted. `missing` are model tensors the
            # checkpoint does not contain (they keep their initialization), which is expected when
            # e.g. warm-starting a predictor from a ViT-only checkpoint.
            log.info(
                f"Predictor pretrained: {len(raw)-len(unexpected)}/{len(raw)} tensors loaded "
                f"({len(missing)} model tensors left at init)"
            )
        if "action_encoder" in ckpt:
            unwrapped.action_encoder.load_state_dict(ckpt["action_encoder"])
            if accelerator.is_main_process:
                log.info("  + action_encoder")
        if "proprio_encoder" in ckpt:
            unwrapped.proprio_encoder.load_state_dict(ckpt["proprio_encoder"])
            if accelerator.is_main_process:
                log.info("  + proprio_encoder")

    # LRM checkpoint: restore only the lrm.* weights (to continue LRM training).
    if cfg.lrm_ckpt and not cfg.resume:
        lrm_ckpt_data = load_checkpoint(cfg.lrm_ckpt, map_location=device)
        lrm_pred_state = lrm_ckpt_data.get("predictor", lrm_ckpt_data)
        unwrapped = accelerator.unwrap_model(model)
        own_state = unwrapped.predictor.state_dict()
        loaded_lrm = 0
        for k, v in lrm_pred_state.items():
            if k.startswith("lrm.") and k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded_lrm += 1
        if accelerator.is_main_process:
            if loaded_lrm > 0:
                log.info(f"LRM checkpoint: loaded {loaded_lrm} lrm.* keys from {cfg.lrm_ckpt}")
            else:
                log.warning(
                    f"WARNING: No lrm.* keys found in {cfg.lrm_ckpt}! LRM weights are random."
                )


def train_one_epoch(
    cfg: TrainConfig,
    model,
    optimizer,
    train_loader,
    accelerator,
    device,
    output_dir: str,
    csv_writer,
    csv_file,
    epoch: int,
    start_epoch: int,
    start_epoch_step: int,
    global_step: int,
    log_every: int,
    save_every_n_steps: int,
) -> tuple[float, int, bool]:
    """One epoch: per-history-length backward, CSV logging, periodic and graceful saves.

    Returns (mean training loss of the epoch, updated global step, aborted by signal).
    """
    model.train()
    epoch_losses = []
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{cfg.epochs} [train]",
        disable=not accelerator.is_main_process,
    )

    for batch_idx, batch in enumerate(pbar):
        # skip already-processed batches when resuming
        if epoch == start_epoch and batch_idx < start_epoch_step:
            continue

        obs, act, state = batch
        optimizer.zero_grad()

        # Every stage accumulates gradients over the history lengths h=1..T-1 of the window
        # (one forward/backward per h), matching the reference trainer.
        T_full = obs["visual"].shape[1]
        total_loss = torch.tensor(0.0, device=device)
        all_components = {}
        n_positions = 0
        for hist_len in range(1, T_full):
            T_slice = hist_len + 1
            obs_slice = {k: v[:, :T_slice] for k, v in obs.items()}
            act_slice = act[:, :T_slice]
            _, _, _, loss_h, components_h = model(obs_slice, act_slice)
            accelerator.backward(loss_h)
            total_loss = total_loss + loss_h.detach()
            n_positions += 1
            for k, v in components_h.items():
                all_components[f"h{hist_len}_{k}"] = v.detach() if torch.is_tensor(v) else v
        loss = total_loss / max(n_positions, 1)
        loss_components = all_components
        loss_components["loss"] = loss
        optimizer.step()

        loss_val = accelerator.gather(loss).mean().item()
        epoch_losses.append(loss_val)
        global_step += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)

        # Graceful shutdown: save a checkpoint and exit on SIGTERM/SIGINT.
        if _shutdown_requested:
            if accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    epoch,
                    output_dir,
                    step=global_step,
                    epoch_step=batch_idx + 1,
                    optimizer=optimizer,
                )
                log.info(f"Graceful shutdown: saved at epoch={epoch} step={global_step}. Exiting.")
            train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            return train_loss, global_step, True

        # Periodic mid-epoch save: bounds the loss from a SIGKILL to `save_every_n_steps` steps.
        if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
            if accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    epoch,
                    output_dir,
                    step=global_step,
                    epoch_step=batch_idx + 1,
                    optimizer=optimizer,
                )
        # Log every N steps: extract h1-h5 loss values
        if global_step % log_every == 0 and accelerator.is_main_process and csv_writer:
            h_losses = []
            for h in range(1, 6):
                key = f"h{h}_loss"
                if key in loss_components:
                    val = loss_components[key]
                    h_losses.append(f"{val.item() if torch.is_tensor(val) else val:.6f}")
                else:
                    h_losses.append("")
            row = [epoch, global_step, f"{loss_val:.6f}", ""] + h_losses
            csv_writer.writerow(row)
            csv_file.flush()

    return (
        sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
        global_step,
        _shutdown_requested,
    )


def inherit_encoders_from_classifier(cfg: TrainConfig, model) -> None:
    """Copy the action/proprio encoders from the classifier checkpoint into ``model``.

    The encoders are trained in the classifier stage and frozen from then on, so every later stage
    must start from the classifier's weights instead of its own random initialisation.
    """
    if not (cfg.classifier_ckpt and cfg.stage in ("predictor", "lrm")):
        return
    cls_ckpt = load_checkpoint(cfg.classifier_ckpt)
    inherited = []
    if "action_encoder" in cls_ckpt:
        model.action_encoder.load_state_dict(cls_ckpt["action_encoder"])
        inherited.append("action_encoder")
    if "proprio_encoder" in cls_ckpt:
        model.proprio_encoder.load_state_dict(cls_ckpt["proprio_encoder"])
        inherited.append("proprio_encoder")
    log.info(
        f"[ENCODER] Inherited {', '.join(inherited) if inherited else 'nothing'}"
        f" from classifier ckpt: {cfg.classifier_ckpt}"
    )


def freeze_inherited_encoders(cfg: TrainConfig, model) -> None:
    """Freeze action/proprio encoder for the stages that only train the predictor/LRM."""
    if cfg.stage not in ("predictor", "lrm"):
        return
    for module in (model.action_encoder, model.proprio_encoder):
        for param in module.parameters():
            param.requires_grad_(False)
    log.info("[PREDICTOR/LRM] Frozen: action_encoder, proprio_encoder (matching old code)")


def resume_from_checkpoint(
    cfg: TrainConfig, model, optimizer, accelerator, device
) -> tuple[int, int, int]:
    """Restore training state from ``--resume``.

    Returns ``(start_epoch, batches_to_skip_in_start_epoch, global_step)``; with no ``--resume``
    the run simply starts at epoch 1.
    """
    if not cfg.resume:
        return 1, 0, 0

    resume_ckpt = load_checkpoint(cfg.resume, map_location=device)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.predictor.load_state_dict(resume_ckpt["predictor"])
    for key, module in (
        ("action_encoder", unwrapped.action_encoder),
        ("proprio_encoder", unwrapped.proprio_encoder),
    ):
        if key in resume_ckpt:
            module.load_state_dict(resume_ckpt[key])
    if "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
    # --lr overrides the learning rate stored in the checkpoint (useful when resuming)
    for param_group in optimizer.param_groups:
        param_group["lr"] = cfg.lr

    ckpt_epoch = resume_ckpt.get("epoch", 1)
    ckpt_epoch_step = resume_ckpt.get("epoch_step", None)
    if ckpt_epoch_step is not None and ckpt_epoch_step > 0:
        start_epoch, start_epoch_step = ckpt_epoch, ckpt_epoch_step
    else:
        start_epoch, start_epoch_step = ckpt_epoch + 1, 0
    if accelerator.is_main_process:
        log.info(
            f"Resumed from {cfg.resume}: epoch={start_epoch}, skip={start_epoch_step}, lr={cfg.lr}"
        )
        if "optimizer" in resume_ckpt:
            log.info("+ optimizer state restored (Adam moments preserved)")
    return start_epoch, start_epoch_step, resume_ckpt.get("step", start_epoch_step)


def validate_args(cfg: TrainConfig) -> None:
    """Reject stage/flag combinations that would train something other than the requested stage."""
    # --eval-ckpt loads everything from the checkpoint itself (an eval-only run trains nothing and
    # does not need the classifier).
    if cfg.stage == "predictor" and not cfg.classifier_ckpt and not cfg.eval_ckpt:
        sys.exit("ERROR: --classifier-ckpt required for the predictor stage")
    if cfg.stage == "lrm" and not cfg.predictor_ckpt and not cfg.eval_ckpt:
        sys.exit("ERROR: --predictor-ckpt required for lrm stage")
    if cfg.history_fusion_disabled and cfg.stage != "predictor":
        sys.exit("ERROR: --history-fusion-disabled only applies to --stage predictor")


def log_run_header(cfg: TrainConfig) -> None:
    """Log the resolved configuration of the run."""
    log.info(f"\n{'='*60}")
    log.info(f"  DDP-WM Training - Stage: {cfg.stage}")
    log.info(f"  Config: {cfg.config_path}")
    log.info(f"  Epochs: {cfg.epochs}, LR: {cfg.lr}, Batch: {cfg.batch_size}")
    log.info(
        f"  log_every_n_steps: {cfg.log_every}, save_every_n_epochs: {cfg.save_every_n_epochs}"
    )
    log.info(f"  num_hist: {cfg.num_hist}, env: {cfg.env}")
    log.info(f"  Output: {cfg.output_dir}")
    if cfg.classifier_ckpt:
        log.info(f"  Classifier: {cfg.classifier_ckpt}")
    if cfg.predictor_ckpt:
        log.info(f"  Predictor: {cfg.predictor_ckpt}")
    log.info(f"{'='*60}\n")


def seed_everything(seed_value: int) -> None:
    """Seed python, numpy and torch so that a run is reproducible.

    The numpy seed matters because TrajSlicerDataset permutes its slices with
    np.random.permutation, so a different numpy seed = a different data order; the torch seed
    determines the model's random initialisation (e.g. the LRM).
    """
    # Match the old trainer's `seed(cfg.training.seed)` (which is seed(0)): it seeds python's
    # `random`, numpy AND torch.
    import random as _random

    import numpy as _np

    _random.seed(seed_value)
    _np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def enable_determinism() -> None:
    """Ask torch for deterministic kernels (--deterministic).

    cuDNN picks its algorithms per run, which makes the classifier and LRM losses drift by ~1e-2 /
    ~1e-4 between otherwise identical runs (cuBLAS workspace choice matters for the LRM). Fixing the
    algorithms is what makes a before/after-refactor comparison meaningful; `warn_only=True` keeps
    ops without a deterministic kernel (a few in the LRM) working instead of raising.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    log.info("Deterministic mode: cudnn.deterministic=True, benchmark=False, warn_only=True")


def build_accelerator() -> Accelerator:
    """Accelerator with the DDP kwargs the frozen modules of the later stages need."""
    # Initialize accelerator with DDP kwargs for frozen parameters
    # predictor/lrm stages freeze some modules, causing "unused parameters" in DDP
    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(kwargs_handlers=[ddp_kwargs])


def build_dataloaders(cfg: TrainConfig, datasets: dict, accelerator) -> tuple:
    """One DataLoader per split, with the per-process batch size."""
    gpu_batch_size = cfg.batch_size // accelerator.num_processes
    train_loader = torch.utils.data.DataLoader(
        datasets["train"], batch_size=gpu_batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        datasets["valid"], batch_size=gpu_batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader


def open_csv_log(cfg: TrainConfig, output_dir: str, accelerator) -> tuple:
    """Open the per-stage training CSV (main process only) and return (writer, file)."""
    import csv

    # Derive run_dir from output_dir (output_dir = run_XXXX/checkpoints/stage)
    run_dir = (
        os.path.dirname(os.path.dirname(output_dir)) if "checkpoints" in output_dir else output_dir
    )
    csv_log_dir = os.path.join(run_dir, "csv_log")
    os.makedirs(csv_log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if not accelerator.is_main_process:
        return None, None
    csv_file = open(os.path.join(csv_log_dir, f"train_log_{cfg.stage}.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "epoch",
            "step",
            "train_loss",
            "val_loss",
            "h1_loss",
            "h2_loss",
            "h3_loss",
            "h4_loss",
            "h5_loss",
        ]
    )
    # Actual data rows are written after the first logged batch (see the LOG_EVERY check).
    csv_file.flush()
    return csv_writer, csv_file


def main():
    setup_logging()
    args = parse_args()
    cfg = resolve_config(args)
    validate_args(cfg)
    log_run_header(cfg)
    if cfg.deterministic:
        enable_determinism()
    seed_everything(cfg.seed)
    output_dir = cfg.output_dir

    accelerator = build_accelerator()
    device = accelerator.device
    log_device(log, device, f"training ({cfg.stage})")

    # Build dataset
    log.info("Loading dataset...")
    datasets = build_dataset(cfg, cfg.num_hist, cfg.frameskip)
    train_loader, val_loader = build_dataloaders(cfg, datasets, accelerator)

    # Build model
    log.info("Building model...")
    model = build_model(cfg, datasets)

    # The only recipe switch left in the whole pipeline: which history the predictor's ViT sees.
    # It is a property of the training recipe (and is stored in the checkpoint so that evaluation
    # can reproduce the same forward path), hence it is set on the model here and not derived from
    # any further config key.
    model.history_fusion_disabled = bool(cfg.history_fusion_disabled) and cfg.stage == "predictor"

    # Action / proprio encoders are trained in the classifier stage and frozen afterwards, so every
    # later stage inherits them from the classifier checkpoint (a predictor started without them
    # would train against random action/proprio embeddings, and the frozen localizer would then see
    # an out-of-distribution input and produce a corrupted mask).
    inherit_encoders_from_classifier(cfg, model)

    # Optimizer (only on trainable params)
    trainable_params = get_trainable_params(model)
    # Match old code: freeze action/proprio encoder during predictor and LRM training
    freeze_inherited_encoders(cfg, accelerator.unwrap_model(model))

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.01)

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Report the recipe of the requested stage (the recipe itself is fixed in the model code)
    log_stage_recipe(cfg, accelerator)

    # ========== EVAL-ONLY MODE ==========
    if cfg.eval_ckpt:
        run_eval_only(cfg, model, val_loader, accelerator)
        return  # Exit after eval

    csv_writer, csv_file = open_csv_log(cfg, output_dir, accelerator)

    # Training-CSV frequency: logging.log_every_n_steps (or --log-every-n-steps)
    LOG_EVERY = max(1, int(cfg.log_every))
    # End-of-epoch checkpoint frequency: logging.save_every_n_epochs (0 = never)
    SAVE_EVERY_N_EPOCHS = int(cfg.save_every_n_epochs)

    # Warm start (--predictor-ckpt) and optional LRM-only restore (--lrm-ckpt)
    load_warm_start_checkpoints(cfg, model, accelerator, device)
    # Register signal handlers: SIGTERM (-15) and SIGINT (Ctrl+C)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    if accelerator.is_main_process:
        log.info("Registered SIGTERM/SIGINT handlers for graceful shutdown")

    # Restore training state (--resume)
    start_epoch, start_epoch_step, global_step = resume_from_checkpoint(
        cfg, model, optimizer, accelerator, device
    )

    save_n = cfg.save_every_n_steps  # 0 disables periodic mid-epoch saves

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, global_step, aborted = train_one_epoch(
            cfg,
            model,
            optimizer,
            train_loader,
            accelerator,
            device,
            output_dir,
            csv_writer,
            csv_file,
            epoch,
            start_epoch,
            start_epoch_step,
            global_step,
            LOG_EVERY,
            save_n,
        )
        if aborted:
            break

        val_loss = validate_epoch(model, val_loader, accelerator, epoch, cfg.epochs)

        if accelerator.is_main_process:
            log.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if csv_writer:
                csv_writer.writerow(
                    [epoch, global_step, f"{train_loss:.6f}", f"{val_loss:.6f}", "EPOCH_END"]
                )
                csv_file.flush()
            if SAVE_EVERY_N_EPOCHS > 0 and epoch % SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(accelerator.unwrap_model(model), epoch, output_dir)
            else:
                log.info(
                    f"Epoch {epoch}: skipping the end-of-epoch checkpoint "
                    f"(save_every_n_epochs={SAVE_EVERY_N_EPOCHS})"
                )

    # Cleanup
    if csv_file:
        csv_file.close()

    log.info("Training complete!")


if __name__ == "__main__":
    main()
