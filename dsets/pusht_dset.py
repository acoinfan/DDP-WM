# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
import logging
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .traj_dset import TrajDataset, TrajSlicerDataset, train_val_dicts

log = logging.getLogger(__name__)
# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027, 2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027, 2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])


class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: int | None = None,
        transform: Callable | None = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True,  # agent's velocity
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth", weights_only=True)
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth", weights_only=True)
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth", weights_only=True)
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ["T"] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth", weights_only=True)
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        log.info(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[: self.state_dim]
            self.state_std = STATE_STD[: self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[: self.proprio_dim]
            self.proprio_std = PROPRIO_STD[: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        self.obs_img_dir = self.data_path / "obses_png"

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):
        """Read the requested frames of trajectory ``idx``.

        Two on-disk layouts are supported: pre-extracted PNG frames
        (``obses_png/episode_XXX/0000.png``, which is what the PushT data this repository trains on
        provides) and the reference implementation's ``obses/episode_XXX.mp4`` videos.
        """
        if self.obs_img_dir.is_dir():
            return self.get_frames_img(idx, frames)
        return self.get_frames_video(idx, frames)

    def get_frames_video(self, idx, frames):
        # decord is an optional dependency (the extra `[datasets]`): it is only needed for this
        # mp4 layout, and its wheels are not available on every platform. The PNG layout above needs
        # nothing but Pillow.
        try:
            import decord
        except ImportError as exc:  # pragma: no cover - depends on the data layout
            raise ImportError(
                "this PushT dataset stores `obses/*.mp4` videos, which need decord; install it with "
                "`pip install 'ddpwm[datasets]'` (or use the pre-extracted PNG layout)"
            ) from exc
        decord.bridge.set_bridge("torch")  # hand out torch tensors
        vid_dir = self.data_path / "obses"
        reader = decord.VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {"shape": shape}

    def get_frames_img(self, idx, frames):
        episode_dir = self.obs_img_dir / f"episode_{idx:03d}"

        image_list = []
        # iterate over the requested frames
        for frame_idx in frames:
            frame_path = episode_dir / f"{frame_idx:04d}.png"
            with Image.open(frame_path) as img:
                img_np = np.array(img)
                if img_np.ndim == 2:  # Grayscale image
                    img_np = np.expand_dims(img_np, axis=-1)
                image_list.append(img_np)
        if not image_list:
            raise RuntimeError(f"No frames could be loaded for episode {idx} with frame indices.")

        image_np = np.stack(image_list, axis=0)
        image = torch.from_numpy(image_np).float() / 255.0
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        shape = self.shapes[idx]

        # (T, H, W, C) -> (T, C, H, W)
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {"shape": shape}  # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_pusht_slice_train_val(
    transform: Callable,
    n_rollout: int | None = 50,
    data_path: str = "data/pusht_dataset",
    normalize_action: bool = True,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 0,
    with_velocity: bool = True,
) -> tuple[dict, dict]:
    """PushT train/val datasets.

    The split comes from the directory layout (`<data_path>/train`, `<data_path>/val`), so unlike the
    other environments there is no `split_ratio` here: a train/val fraction would have no effect.
    """
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    return train_val_dicts(train_dset, val_dset, train_slices, val_slices)
