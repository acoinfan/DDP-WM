# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""
DDP-WM Predictor: Clean three-stage training implementation.

Architecture (matching paper exactly):
  Stage 1: Historical Information Fusion (cross-attention to history)
  Stage 2: Dynamic Localization Network (lightweight ViT classifier)
  Stage 3: Sparse Primary Dynamics Predictor (full ViT on foreground tokens only)
  Stage 4: Low-Rank Correction Module (cross-attention from BG to FG)

Training stages:
  'classifier'        -> Train Stage 1+2 jointly (history fusion + localization)
  'predictor' -> Freeze 1+2, train Stage 3
  'lrm'              -> Freeze 1+2+3, train Stage 4
  'inference'        -> All frozen, full forward pass

Usage:
  predictor = DDP_Predictor(training_stage='classifier', ...)
  predictor = DDP_Predictor(training_stage='predictor', classifier_ckpt='path/to/cls.pth', ...)
  predictor = DDP_Predictor(training_stage='lrm', predictor_ckpt='path/to/pred.pth', ...)
"""

import logging
import math
import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .ckpt_io import load_checkpoint
from .ckpt_layout import assert_current_layout
from .vit import ViTPredictor

log = logging.getLogger(__name__)


# ==============================================================================
# Configuration Constants
# ==============================================================================

GRID_SIZE = 14  # DINOv2 ViT-S/14 produces 14x14 patch grid
N_PATCHES = GRID_SIZE * GRID_SIZE  # 196
D_VISUAL = 384  # DINOv2 ViT-S feature dimension
D_ACTION_EMB = 10  # Action embedding dimension after ProprioceptiveEmbedding
D_PROPRIO_EMB = 10  # Proprio embedding dimension
D_MODEL = D_VISUAL + D_ACTION_EMB + D_PROPRIO_EMB  # 404, full token dimension

# Classifier (Stage 2) architecture
CLS_REDUCED_DIM = 192  # Dimensionality reduction before small ViT
CLS_NUM_LAYERS = 6  # Small ViT depth
CLS_NUM_HEADS = 3  # Small ViT heads
CLS_MLP_DIM = 768  # Small ViT FFN dimension
CLS_PARTITION = 4  # Each 14x14 patch predicts 2x2=4 sub-region change probs

# Primary Predictor (Stage 3) architecture
PRED_NUM_LAYERS = 6  # Main ViT depth
PRED_NUM_HEADS = 16  # Main ViT heads
PRED_MLP_DIM = 2048  # Main ViT FFN dimension
K_MAX = 32  # Fixed number of foreground tokens after mask processing

# LRM (Stage 4) architecture
LRM_NUM_HEADS = 4  # Cross-attention heads

# Label generation
PIXEL_THRESHOLD = 0.1  # Threshold for pixel-diff based GT mask generation


# ==============================================================================
# Utility Functions
# ==============================================================================


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module and set to eval mode."""
    if module is None:
        return
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def dilate_mask_2d(mask: Tensor, connectivity: int = 0) -> Tensor:
    """
    Dilate a binary mask on 14x14 grid.
    Args:
        mask: (B, 196) float, 0 or 1
        connectivity: 0=no dilation, 4=cross, 8=square
    Returns:
        (B, 196) dilated mask
    """
    if connectivity not in [4, 8]:
        return mask

    b, n = mask.shape
    if n != N_PATCHES:
        raise ValueError(f"mask has {n} patches, expected {N_PATCHES}")

    mask_2d = mask.view(b, 1, GRID_SIZE, GRID_SIZE)

    if connectivity == 4:
        kernel = torch.tensor(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=mask.dtype, device=mask.device
        ).view(1, 1, 3, 3)
    else:
        kernel = torch.ones(1, 1, 3, 3, dtype=mask.dtype, device=mask.device)

    dilated = F.conv2d(mask_2d, kernel, padding=1)
    return (dilated > 0).float().view(b, n)


def force_fixed_k(mask: Tensor, k: int = K_MAX) -> Tensor:
    """
    Adjust mask so each row has exactly k True values.
    Uses score-based selection: existing True positions get priority.
    """
    B, N = mask.shape
    noise = torch.rand(B, N, device=mask.device)
    scores = mask.float() * 2.0 + noise  # True positions score 2-3, False score 0-1
    _, indices = torch.topk(scores, k, dim=1)
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask.scatter_(1, indices, True)
    return new_mask


# ==============================================================================
# Building Blocks
# ==============================================================================


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n_in, n_out in zip([input_dim] + h, h + [output_dim], strict=False):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU(inplace=True))
        layers.pop()  # Remove last ReLU
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual and LayerNorm (post-norm)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
    ) -> Tensor:
        """
        Args: All tensors in (L, B, D) format.
        """
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if key_pos is None else memory + key_pos
        attn_out = self.cross_attn(query=q, key=k, value=memory)[0]
        tgt = tgt + self.dropout(attn_out)
        tgt = self.norm(tgt)
        return tgt


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer decoder layer: self-attn + cross-attn + FFN (post-norm)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
    ) -> Tensor:
        # Self-attention
        q = k = tgt if query_pos is None else tgt + query_pos
        sa_out = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(sa_out)
        tgt = self.norm1(tgt)
        # Cross-attention
        q2 = tgt if query_pos is None else tgt + query_pos
        k2 = memory if key_pos is None else memory + key_pos
        ca_out = self.cross_attn(q2, k2, memory)[0]
        tgt = tgt + self.dropout2(ca_out)
        tgt = self.norm2(tgt)
        # FFN
        ffn_out = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ffn_out)
        tgt = self.norm3(tgt)
        return tgt


# ==============================================================================
# Stage 1: Historical Information Fusion
# ==============================================================================


class HistoricalInformationFusion(nn.Module):
    """
    Paper Stage 1: Fuse history frames into current frame via cross-attention.
    Query = current frame tokens, Key/Value = history frame tokens.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = 4,
        num_frames: int = 3,
        num_patches: int = N_PATCHES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)
        self.query_pos = nn.Parameter(torch.randn(num_patches, d_model))
        self.mem_pos = nn.Parameter(torch.randn(num_patches, d_model))
        self.time_embeds = (
            nn.Parameter(torch.zeros(max(num_frames - 1, 1), d_model)) if num_frames > 1 else None
        )

    def forward(self, z_history: Tensor) -> Tensor:
        """
        Args:
            z_history: (B, T, N, D) - history sequence, last frame is current
        Returns:
            z_t_prime: (B, 1, N, D) - current frame with history fused in
        """
        B, T, N, D = z_history.shape

        if T <= 1:
            # No history to fuse, ensure params in computation graph
            dummy = 0
            if self.time_embeds is not None:
                dummy = (
                    self.time_embeds.sum() * 0 + self.query_pos.sum() * 0 + self.mem_pos.sum() * 0
                )
            return z_history + dummy

        # Current frame as Query: (N, B, D)
        current = z_history[:, -1].permute(1, 0, 2)
        q_pos = self.query_pos.unsqueeze(1).expand(-1, B, -1)

        # History frames as Key/Value: ((T-1)*N, B, D)
        history = rearrange(z_history[:, :-1], "b t n d -> (t n) b d")

        # Build memory position with temporal encoding
        hist_len = T - 1
        mem_pos_spatial = self.mem_pos.unsqueeze(0).expand(hist_len, -1, -1)  # (T-1, N, D)
        if self.time_embeds is not None:
            time_pe = self.time_embeds[:hist_len].unsqueeze(1)  # (T-1, 1, D)
            mem_pos_spatial = mem_pos_spatial + time_pe
        mem_pos_flat = rearrange(mem_pos_spatial, "t n d -> (t n) d").unsqueeze(1).expand(-1, B, -1)

        # Cross-attention
        encoded = self.cross_attn(current, history, query_pos=q_pos, key_pos=mem_pos_flat)

        return encoded.permute(1, 0, 2).unsqueeze(1)  # (B, 1, N, D)


# ==============================================================================
# Stage 2: Dynamic Localization Network (Classifier)
# ==============================================================================


class DynamicLocalizationNetwork(nn.Module):
    """
    Paper Stage 2: Lightweight ViT that predicts which patches will change.
    Predicts 4 sub-region change probabilities per 14x14 patch (= 28x28 resolution).
    """

    def __init__(
        self,
        d_visual: int = D_VISUAL,
        d_action: int = D_ACTION_EMB,
        d_proprio: int = D_PROPRIO_EMB,
        reduced_dim: int = CLS_REDUCED_DIM,
        num_layers: int = CLS_NUM_LAYERS,
        num_heads: int = CLS_NUM_HEADS,
        mlp_dim: int = CLS_MLP_DIM,
        partition: int = CLS_PARTITION,
    ):
        super().__init__()
        self.dim_reduce = nn.Linear(d_visual, reduced_dim)

        vit_dim = reduced_dim + d_action + d_proprio  # 192 + 10 + 10 = 212
        self.vit = ViTPredictor(
            dim=vit_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            num_frames=1,
            num_patches=N_PATCHES,
            dropout=0.1,
            emb_dropout=0,
        )
        self.cls_head = MLP(vit_dim, vit_dim, partition, num_layers=3)

    def forward(self, z_t_prime: Tensor) -> Tensor:
        """
        Args:
            z_t_prime: (B, 1, N, D_MODEL) - fused current frame
        Returns:
            logits: (B, N, 4) - per-patch sub-region change logits
        """
        vis = z_t_prime[..., :D_VISUAL]  # (B, 1, N, 384)
        prio = z_t_prime[..., D_VISUAL : D_VISUAL + D_PROPRIO_EMB]  # (B, 1, N, 10)
        act = z_t_prime[..., -D_ACTION_EMB:]  # (B, 1, N, 10)

        vis_reduced = self.dim_reduce(vis)  # (B, 1, N, 192)
        x = torch.cat([vis_reduced, prio, act], dim=-1)  # (B, 1, N, 212)
        x = rearrange(x, "b t n d -> b (t n) d")  # (B, N, 212)
        x = self.vit(x)  # (B, N, 212)
        logits = self.cls_head(x)  # (B, N, 4)
        return logits


# ==============================================================================
# Stage 3: Sparse Primary Dynamics Predictor
# ==============================================================================


class SparsePrimaryPredictor(nn.Module):
    """
    Paper Stage 3: Full-power ViT that processes ONLY foreground tokens.
    Uses ViTPredictor with masking to achieve sparse computation.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = PRED_NUM_LAYERS,
        num_heads: int = PRED_NUM_HEADS,
        mlp_dim: int = PRED_MLP_DIM,
    ):
        super().__init__()
        self.vit = ViTPredictor(
            dim=d_model,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            num_frames=1,
            num_patches=N_PATCHES,
            dropout=0.1,
            emb_dropout=0,
        )

        # === Hungarian-matching heads of the reference pipeline ===
        # This reproduction does not use the Hungarian loss, so nothing consumes pred_coords /
        # pred_cls: these two heads get no gradient in any stage and are kept only so that a
        # checkpoint of the reference implementation still loads key-for-key. They are frozen here
        # (and therefore never enter the optimizer) so that "trainable parameters" means what it
        # says; the evaluation's completeness check skips them for the same reason.
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),  # Output normalized [0,1] coordinates
        )
        self.cls_head_pred = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1))
        for head in (self.coord_head, self.cls_head_pred):
            freeze_module(head)

    def forward(self, z_t_prime: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            z_t_prime: (B, N, D_MODEL) - fused current frame (squeezed)
            mask: (B, N) boolean - True = foreground
        Returns:
            pred_fg: (B, K, D_MODEL) - predicted foreground features
        """
        # ViTPredictor with mask: processes only True tokens
        pred_fg = self.vit(z_t_prime, mask)  # (B, K, D)
        pred_coords = self.coord_head(pred_fg)  # (B, K, 2)
        pred_cls = self.cls_head_pred(pred_fg).squeeze(-1)  # (B, K)
        return pred_fg, pred_coords, pred_cls


# ==============================================================================
# Stage 4: Low-Rank Correction Module (LRM)
# ==============================================================================


class LowRankCorrectionModule(nn.Module):
    """
    Paper Stage 4: Single cross-attention layer.
    Background tokens (Query) attend to predicted foreground tokens (Key/Value).
    Uses learnable APE assigned by mask to provide spatial information.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = LRM_NUM_HEADS,
        n_patches: int = N_PATCHES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ape = nn.Parameter(torch.randn(n_patches, d_model))
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)

    @staticmethod
    def mask_to_indices(mask: Tensor) -> tuple[Tensor, Tensor]:
        """Split mask into foreground and background indices."""
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        B, N = mask.shape
        pos = torch.arange(N, device=mask.device).unsqueeze(0).expand(B, -1)
        idx_fg = pos[mask].view(B, -1)
        idx_bg = pos[~mask].view(B, -1)
        return idx_fg, idx_bg

    def forward(self, z_t_prime: Tensor, mask: Tensor, pred_fg: Tensor) -> Tensor:
        """
        Args:
            z_t_prime: (B, N, D) - current frame features
            mask: (B, N) boolean - True = foreground
            pred_fg: (B, K, D) - predicted next-frame foreground features
        Returns:
            updated_bg: (B, N-K, D) - updated background features
        """
        B, N, D = z_t_prime.shape
        K = pred_fg.shape[1]
        num_bg = N - K

        # Extract background tokens
        bg_tokens = z_t_prime[~mask].view(B, num_bg, D)

        # Get position indices
        idx_fg, idx_bg = self.mask_to_indices(mask)

        # Assign APE by mask
        ape_fg = self.ape.index_select(0, idx_fg.reshape(-1)).view(B, K, D)
        ape_bg = self.ape.index_select(0, idx_bg.reshape(-1)).view(B, num_bg, D)

        # Cross-attention: BG queries FG (L, B, D format)
        tgt = bg_tokens.transpose(0, 1)  # (N-K, B, D)
        mem = pred_fg.transpose(0, 1)  # (K, B, D)
        q_pos = ape_bg.transpose(0, 1)  # (N-K, B, D)
        k_pos = ape_fg.transpose(0, 1)  # (K, B, D)

        updated = self.cross_attn(tgt, mem, query_pos=q_pos, key_pos=k_pos)
        return updated.transpose(0, 1)  # (B, N-K, D)


# ==============================================================================
# Label Generator: Creates GT masks from pixel differences
# ==============================================================================


class LabelGenerator(nn.Module):
    """
    Generates binary foreground masks from pixel-level frame differences.
    Used as GT supervision for the classifier (Stage 2).
    Not trainable - purely deterministic.
    """

    def __init__(
        self,
        threshold: float = PIXEL_THRESHOLD,
        grid_h: int = GRID_SIZE,
        grid_w: int = GRID_SIZE,
        partition: int = CLS_PARTITION,
    ):
        super().__init__()
        self.threshold = threshold
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.partition = partition  # 4 = 2x2 sub-patches
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, frames_current: Tensor, frames_next: Tensor) -> Tensor:
        """
        Args:
            frames_current: (B, C, H, W) - current frame images (after encoder_transform)
            frames_next: (B, C, H, W) - next frame images
        Returns:
            gt_mask: (B, N, 4) float - per sub-patch binary labels (0/1)
        """
        B, C, H, W = frames_current.shape
        pixel_diff = frames_next - frames_current  # (B, C, H, W)

        # Pixel L2 norm squared
        norms_sq = torch.sum(pixel_diff**2, dim=1, keepdim=True)  # (B, 1, H, W)

        # Pool to sub-patch resolution (28x28 for partition=4 on 14x14 grid)
        sub_h = int(math.sqrt(self.partition))  # 2
        patch_h = H // (self.grid_h * sub_h)
        patch_w = W // (self.grid_w * sub_h)

        pooled = F.avg_pool2d(norms_sq, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        # pooled: (B, 1, grid_h*2, grid_w*2) = (B, 1, 28, 28)

        rms = torch.sqrt(pooled).squeeze(1)  # (B, 28, 28)
        binary = (rms > self.threshold).float()  # (B, 28, 28)

        # Reshape to (B, N, 4): group 2x2 sub-patches per parent patch
        # (B, 28, 28) -> (B, 14, 2, 14, 2) -> (B, 14, 14, 2, 2) -> (B, 196, 4)
        gt_mask = rearrange(
            binary,
            "b (h p1) (w p2) -> b (h w) (p1 p2)",
            h=self.grid_h,
            w=self.grid_w,
            p1=sub_h,
            p2=sub_h,
        )
        return gt_mask


# ==============================================================================
# Main Assembly: DDP_Predictor
# ==============================================================================


class DDP_Predictor(nn.Module):
    """
    Complete DDP-WM predictor with staged training support.

    Args:
        training_stage: One of 'classifier', 'predictor', 'lrm', 'inference'
        classifier_ckpt: Path to trained classifier checkpoint (for stages after classifier)
        predictor_ckpt: Path to trained primary predictor checkpoint (for lrm stage)
    """

    def __init__(
        self,
        training_stage: Literal["classifier", "predictor", "lrm", "inference"] = "classifier",
        classifier_ckpt: str | None = None,
        predictor_ckpt: str | None = None,
    ):
        super().__init__()
        self.training_stage = training_stage

        log.info(f"[DDP_Predictor] Initializing in '{training_stage}' stage")

        # --- Always create Stage 1 + 2 (needed by all stages) ---
        # history_fusion always uses num_frames=3 (its internal context window)
        # regardless of the overall num_hist setting (which affects data window for rollout)
        HISTORY_FUSION_FRAMES = 3
        self.history_fusion = HistoricalInformationFusion(
            d_model=D_MODEL, nhead=4, num_frames=HISTORY_FUSION_FRAMES, num_patches=N_PATCHES
        )
        self.localizer = DynamicLocalizationNetwork()
        self.label_generator = LabelGenerator()

        # --- Stage 3: needed for predictor, lrm, inference ---
        self.primary_predictor: SparsePrimaryPredictor | None = None
        if training_stage in ["predictor", "lrm", "inference"]:
            self.primary_predictor = SparsePrimaryPredictor()

        # --- Stage 4: needed for lrm, inference ---
        self.lrm: LowRankCorrectionModule | None = None
        if training_stage in ["lrm", "inference"]:
            self.lrm = LowRankCorrectionModule()

        # --- Load pre-trained weights ---
        if training_stage in ["predictor", "lrm", "inference"] and classifier_ckpt:
            self._load_classifier(classifier_ckpt)

        if training_stage in ["lrm", "inference"] and predictor_ckpt:
            self._load_predictor(predictor_ckpt)

        # --- Freeze stages that should not be trained ---
        self._freeze_stages()

    def _load_classifier(self, ckpt_path: str):
        """Load trained classifier (history_fusion + localizer) weights.

        Only the current checkpoint layout is accepted: the tensors are already prefixed with
        'history_fusion.' / 'localizer.'. Legacy layouts raise, convert them offline first
        (see third_party/README.md).
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        if "predictor" in ckpt:
            state = ckpt["predictor"]
        else:
            state = ckpt

        # Only the current checkpoint layout is supported (legacy keys raise, see models/ckpt_layout.py).
        assert_current_layout(state, ckpt_path, source="DDP_Predictor")

        own_state = self.state_dict()
        loaded = 0
        for k, v in state.items():
            if k.startswith(("history_fusion.", "localizer.")):
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k].copy_(v)
                    loaded += 1
        log.info(f"[DDP_Predictor] Loaded {loaded} classifier params from {ckpt_path}")

    def _load_predictor(self, ckpt_path: str):
        """Load trained primary predictor weights (includes classifier + primary_predictor)."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Predictor checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        if "predictor" in ckpt:
            state = ckpt["predictor"]
        else:
            state = ckpt
        own_state = self.state_dict()
        loaded = 0
        for k, v in state.items():
            if k.startswith(("history_fusion.", "localizer.", "primary_predictor.")):
                if k in own_state:
                    own_state[k].copy_(v)
                    loaded += 1
        log.info(f"[DDP_Predictor] Loaded {loaded} predictor parameters from {ckpt_path}")

    def _freeze_stages(self):
        """Freeze modules based on training stage."""
        if self.training_stage == "classifier":
            pass  # Train everything (history_fusion + localizer)

        elif self.training_stage == "predictor":
            freeze_module(self.history_fusion)
            freeze_module(self.localizer)
            log.info("[DDP_Predictor] Frozen: history_fusion, localizer")

        elif self.training_stage == "lrm":
            freeze_module(self.history_fusion)
            freeze_module(self.localizer)
            freeze_module(self.primary_predictor)
            log.info("[DDP_Predictor] Frozen: history_fusion, localizer, primary_predictor")

        elif self.training_stage == "inference":
            freeze_module(self)
            log.info("[DDP_Predictor] Frozen: ALL modules")

    def train(self, mode: bool = True):
        """Keep the frozen classifier in eval() mode, as the old code did.

        Old MotionPredictor overrode train() with `super().train(False)`, so during the
        predictor / LRM stages the classifier ran with dropout DISABLED while the shared history
        fusion and the predictor ViT ran in train mode. A plain nn.Module.train() would switch the
        (frozen) localizer back to train and re-enable its dropout, which changes the sampled
        foreground mask during training (and is a train/eval mismatch at inference time).
        """
        super().train(mode)
        if self.training_stage != "classifier" and self.localizer is not None:
            self.localizer.eval()
        return self

    def _process_mask(self, logits: Tensor) -> Tensor:
        """Convert classifier logits to binary mask (B, N)."""
        # logits: (B, N, 4) - sigmoid > 0.5 per sub-region, any active = patch active
        probs = logits.sigmoid()
        patch_active = (
            (probs > 0.5).float().sum(dim=-1)
        )  # 0-4 sub-region count, matching old code  # (B, N)
        # No dilation for PushT (the paper uses connectivity=0, i.e. no mask dilation)
        mask = force_fixed_k(patch_active, k=K_MAX)
        return mask  # (B, N) bool

    def _process_mask_from_active(self, patch_active: Tensor) -> Tensor:
        """Same as _process_mask but starting from an already-computed (B, N) activity map.

        Used by the GT-filled rollout, which mirrors the reference ``tools._infer()``:
        activity -> force_fixed_k(32).
        """
        return force_fixed_k(patch_active, k=K_MAX)

    def forward(
        self,
        z_history: Tensor,
        images_current: Tensor | None = None,
        images_next: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass with stage-dependent outputs.

        Args:
            z_history: (B, T, N, D_MODEL) - encoded feature sequence
            images_current: (B, C, H, W) - for GT mask generation (classifier training)
            images_next: (B, C, H, W) - for GT mask generation (classifier training)

        Returns:
            Dict with stage-dependent keys:
              classifier stage: {'mask_logits', 'gt_mask'}
              predictor stage: {'pred_fg', 'mask', 'gt_fg'}
              lrm stage: {'pred_bg', 'mask', 'gt_bg'}
              inference stage: {'prediction'}  (B, 1, N, D_MODEL)
        """
        # --- Stage 1: History Fusion ---
        # history_fusion is designed for exactly 3 frames (2 history + 1 current).
        # If z_history has more frames (e.g., from rollout with num_hist=5),
        # only take the last 3 to match time_embeds dimension.
        FUSION_CONTEXT = 3
        z_for_fusion = (
            z_history[:, -FUSION_CONTEXT:] if z_history.shape[1] > FUSION_CONTEXT else z_history
        )

        with torch.no_grad() if self.training_stage != "classifier" else torch.enable_grad():
            z_t_prime = self.history_fusion(z_for_fusion)  # (B, 1, N, D)

        # --- Stage 2: Dynamic Localization ---
        with torch.no_grad() if self.training_stage != "classifier" else torch.enable_grad():
            mask_logits = self.localizer(z_t_prime)  # (B, N, 4)

        if self.training_stage == "classifier":
            # Generate GT mask for supervision
            gt_mask = None
            if images_current is not None and images_next is not None:
                gt_mask = self.label_generator(images_current, images_next)  # (B, N, 4)
            return {"mask_logits": mask_logits, "gt_mask": gt_mask}

        # --- Process mask for downstream stages ---
        mask = self._process_mask(mask_logits)  # (B, N) bool

        # --- Stage 3: Sparse Primary Dynamics Prediction ---
        # Predictor uses history-fused features when available.
        # Matches old code (MotionPredictor/DETRStylePredictor) where z_history
        # is overwritten by fusion output before reaching theViT.
        # - T=1: fusion returns identity → z_t_prime = raw features (no change)
        # - T>1: fusion applies cross-attention → z_t_prime = fused features
        # This gives the ViT temporal context for better h2+ predictions.
        z_current = z_t_prime.squeeze(1)  # (B, N, D) - fused features (raw when T=1)

        with torch.no_grad() if self.training_stage == "lrm" else torch.enable_grad():
            # The ViT sees only the K=32 classifier-masked foreground tokens.
            pred_fg, pred_coords, pred_cls = self.primary_predictor(
                z_current, mask
            )  # (B,K,D), (B,K,2), (B,K)

        if self.training_stage == "predictor":
            # GT: extract foreground features from next frame (provided externally)
            return {
                "pred_fg": pred_fg,
                "pred_coords": pred_coords,
                "pred_cls": pred_cls,
                "mask": mask,
                "z_current": z_current,
            }

        # --- Stage 4: Low-Rank Correction ---
        updated_bg = self.lrm(z_current, mask, pred_fg)  # (B, N-K, D)

        if self.training_stage == "lrm":
            return {"pred_bg": updated_bg, "pred_fg": pred_fg, "mask": mask, "z_current": z_current}

        # --- Inference: Combine foreground and background ---
        z_next = torch.zeros_like(z_current)
        z_next[mask] = pred_fg.reshape(-1, D_MODEL)
        z_next[~mask] = updated_bg.reshape(-1, D_MODEL)

        return {"prediction": z_next.unsqueeze(1)}  # (B, 1, N, D)
