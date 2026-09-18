# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
import torch.nn as nn


class ProprioceptiveEmbedding(nn.Module):
    """Per-token action / proprioception embedding: a 1-D convolution over the vector's time axis.

    Input  [B, T, in_chans]  ->  Output  [B, T, emb_dim]   (kernel/stride = tubelet_size)
    """

    def __init__(
        self,
        tubelet_size=1,
        in_chans=8,  # action_dim
        emb_dim=384,  # output_dim
    ):
        super().__init__()

        # Map input to predictor dimension
        self.patch_embed = nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        # x: proprioceptive vectors of shape [B T D]
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x
