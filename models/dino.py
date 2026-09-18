# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""Frozen DINOv2 visual encoder used by every stage of the pipeline."""

import logging
import os

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# torch.hub coordinates of the encoder. `DDPWM_DINOV2_HUB` overrides them, which is useful to
#   * pin a revision for bit-reproducible runs:  DDPWM_DINOV2_HUB=facebookresearch/dinov2:<commit>
#   * use a local checkout on a machine without network access:  DDPWM_DINOV2_HUB=/path/to/dinov2
# The weights are cached under ~/.cache/torch/hub/checkpoints/ (they are never written into this
# repository).
DEFAULT_DINOV2_HUB = "facebookresearch/dinov2"


class DinoV2Encoder(nn.Module):
    """DINOv2 backbone in inference mode, exposing its patch tokens (or the CLS token).

    Args:
        name: torch.hub entry point, e.g. ``dinov2_vits14`` (ViT-S/14, 384 dims, 14x14 patches).
        feature_key: ``x_norm_patchtokens`` (per-patch features, what the pipeline uses) or
            ``x_norm_clstoken`` (a single global token).
    """

    def __init__(
        self,
        name: str = "dinov2_vits14",
        feature_key: str = "x_norm_patchtokens",
        hub: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.feature_key = feature_key
        if feature_key not in ("x_norm_patchtokens", "x_norm_clstoken"):
            raise ValueError(f"unsupported feature_key: {feature_key}")

        repo = hub or os.environ.get("DDPWM_DINOV2_HUB") or DEFAULT_DINOV2_HUB
        if os.path.isdir(repo):  # local checkout (offline machines)
            log.info(f"[DinoV2Encoder] loading {name} from local checkout {repo}")
            self.base_model = torch.hub.load(repo, name, source="local", verbose=False)
        else:  # 'owner/repo' or 'owner/repo:<revision>'
            log.info(
                f"[DinoV2Encoder] loading {name} from torch.hub {repo} "
                f"(downloaded/cached under ~/.cache/torch/hub)"
            )
            self.base_model = torch.hub.load(
                repo, name, source="github", trust_repo=True, verbose=False
            )
        for p in self.base_model.parameters():  # frozen for every stage
            p.requires_grad_(False)

        self.emb_dim = self.base_model.num_features
        self.patch_size = self.base_model.patch_size
        # number of feature vectors the encoder returns per frame
        self.latent_ndim = 1 if feature_key == "x_norm_clstoken" else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, emb_dim) for patch tokens, (B, 1, emb_dim) for the CLS token."""
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dimension
        return emb
