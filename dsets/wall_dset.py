# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced, train_val_dicts

log = logging.getLogger(__name__)

# precomputed dataset stats
ACTION_MEAN = torch.tensor([0.0006, 0.0015])
ACTION_STD = torch.tensor([0.4395, 0.4684])
STATE_MEAN = torch.tensor([0.7518, 0.9239, -3.9702e-05, 3.1550e-04])
STATE_STD = torch.tensor([1.0964, 1.2390, 1.3819, 1.5407])


class WallDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/wall_single",
        n_rollout: int | None = None,
        transform: Callable | None = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        log.info("Loading wall dataset...")
        states = torch.load(self.data_path / "states.pth", weights_only=True)
        self.states = states
        self.proprios = self.states.clone()
        self.actions = torch.load(self.data_path / "actions.pth", weights_only=True)
        self.actions = self.actions / action_scale
        self.door_locations = torch.load(self.data_path / "door_locations.pth", weights_only=True)
        self.wall_locations = torch.load(self.data_path / "wall_locations.pth", weights_only=True)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)
            log.info(f"Loaded {n} rollouts")

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.proprios = self.proprios[:n]
        self.door_locations = self.door_locations[:n]
        self.wall_locations = self.wall_locations[:n]

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        self.traj_len = self.actions.shape[1]
        if normalize_action:
            self.action_mean = self.actions.mean(dim=(0, 1))
            self.action_std = self.actions.std(dim=(0, 1))
            self.state_mean = self.states.mean(dim=(0, 1))
            self.state_std = self.states.std(dim=(0, 1))
            self.proprio_mean = self.proprios.mean(dim=(0, 1))
            self.proprio_std = self.proprios.std(dim=(0, 1))
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
        return self.traj_len

    def get_frames(self, idx, frames):
        """Read the requested frames of trajectory ``idx``.

        Two on-disk layouts are supported: pre-extracted PNG frames
        (``obses_png/episode_XXX/0000.png``) and stacked ``obses/episode_XXX.pth`` arrays.
        """
        if self.obs_img_dir.is_dir():
            return self.get_frames_img(idx, frames)
        return self.get_frames_stacked(idx, frames)

    def get_frames_stacked(self, idx, frames):
        obs_dir = self.data_path / "obses"
        image = torch.load(obs_dir / f"episode_{idx:03d}.pth", weights_only=True)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]

        image = image[frames] / 255
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return (
            obs,
            act,
            state,
            {"fix_door_location": door_location[0], "fix_wall_location": wall_location[0]},
        )

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
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]
        # (T, H, W, C) -> (T, C, H, W)
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return (
            obs,
            act,
            state,
            {"fix_door_location": door_location[0], "fix_wall_location": wall_location[0]},
        )  # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.states.shape[0] if not self.n_rollout else self.n_rollout


def load_wall_slice_train_val(
    transform: Callable,
    n_rollout: int | None = 50,
    data_path: str = "data/wall_single",
    normalize_action: bool = False,
    split_ratio: float = 0.8,
    split_mode: str = "random",
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 0,
) -> tuple[dict, dict]:
    if split_mode == "random":
        dset = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
        )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset,
            train_fraction=split_ratio,
            num_frames=num_hist + num_pred,
            frameskip=frameskip,
        )
    elif split_mode == "folder":
        dset_train = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/train",
            normalize_action=normalize_action,
        )
        dset_val = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/val",
            normalize_action=normalize_action,
        )
        num_frames = num_hist + num_pred
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)

    return train_val_dicts(dset_train, dset_val, train_slices, val_slices)
