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

from .traj_dset import TrajDataset, get_train_val_sliced, train_val_dicts

log = logging.getLogger(__name__)


class PointMazeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/point_maze",
        n_rollout: int | None = None,
        transform: Callable | None = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        states = torch.load(self.data_path / "states.pth", weights_only=True).float()
        self.states = states
        self.actions = torch.load(self.data_path / "actions.pth", weights_only=True).float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth", weights_only=True)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        log.info(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths
            )
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

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

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
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        image = image[frames]  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}  # env_info

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

        # (T, H, W, C) -> (T, C, H, W)
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}  # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_point_maze_slice_train_val(
    transform: Callable,
    n_rollout: int | None = 50,
    data_path: str = "data/pusht_dataset",
    normalize_action: bool = False,
    split_ratio: float = 0.8,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 0,
) -> tuple[dict, dict]:
    dset = PointMazeDataset(
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
    return train_val_dicts(dset_train, dset_val, train_slices, val_slices)
