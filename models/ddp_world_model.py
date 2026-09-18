# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""
DDP-WM World Model: Orchestrates encoding, prediction, and loss computation.

Wraps:
  - Frozen DINOv2 encoder
  - ProprioceptiveEmbedding for action/proprio
  - DDP_Predictor (with training_stage support)
  - Stage-specific loss functions

Each training stage has its own loss:
  - classifier: BCE on mask logits vs pixel-diff GT
  - predictor: MSE on predicted foreground features vs GT
  - lrm: MSE on predicted background features vs GT
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import transforms

from .ddp_predictor import D_ACTION_EMB, D_PROPRIO_EMB, D_VISUAL, DDP_Predictor

log = logging.getLogger(__name__)


class DDPWorldModel(nn.Module):
    """
    DDP-WM World Model.

    Handles three training stages with different loss functions:
      - classifier: BCE loss on 28x28 sub-patch classification
      - predictor: MSE loss on foreground token features
      - lrm: MSE loss on background token features

    Args:
        training_stage: 'classifier', 'predictor', 'lrm', or 'inference'
        num_hist: Number of history frames (3 for classifier, 5 for primary with rollout)
        classifier_ckpt: Path to trained classifier checkpoint
        predictor_ckpt: Path to trained primary predictor checkpoint
        cls_pos_weight: Positive class weight for BCE in classifier training
    """

    def __init__(
        self,
        image_size: int = 224,
        num_hist: int = 3,
        training_stage: str = "classifier",
        encoder=None,
        action_encoder=None,
        proprio_encoder=None,
        classifier_ckpt: str | None = None,
        predictor_ckpt: str | None = None,
        cls_pos_weight: float = 10.0,
        num_action_repeat: int = 1,
        num_proprio_repeat: int = 1,
        proprio_dim: int = 0,
        action_dim: int = 0,
    ):
        super().__init__()

        self.training_stage = training_stage
        self.num_hist = num_hist

        # Store dimensions
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat

        # External modules (built by build_model / load_ddpwm_model and passed in)
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder

        # Encoder transform
        if encoder is not None and hasattr(encoder, "name") and "dino" in encoder.name:
            decoder_scale = 16
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            self.encoder_transform = lambda x: x

        # Create predictor
        self.predictor = DDP_Predictor(
            training_stage=training_stage,
            classifier_ckpt=classifier_ckpt,
            predictor_ckpt=predictor_ckpt,
        )

        # Loss functions per stage
        self.cls_pos_weight = cls_pos_weight

        log.info(f"[DDPWorldModel] stage={training_stage}, num_hist={num_hist}")

    def train(self, mode=True):
        super().train(mode)
        # The DINOv2 encoder is frozen for every stage, so it stays in eval() mode.
        if self.encoder is not None:
            self.encoder.eval()
        if self.predictor is not None:
            self.predictor.train(mode)
        return self

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode visual observations and proprioception."""
        visual = obs["visual"]
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        with torch.no_grad():
            visual_embs = self.encoder(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs["proprio"]
        proprio_emb = self.proprio_encoder(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def encode_action(self, act: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(act)

    def build_z(self, obs: dict[str, torch.Tensor], act: torch.Tensor) -> torch.Tensor:
        """
        Build the full feature tensor z by encoding and concatenating.
        Returns: (B, T, N, D_MODEL) where D_MODEL = 384 + 10 + 10 = 404
        """
        z_dict = self.encode_obs(obs)
        act_emb = self.encode_action(act)  # (B, T, D_act)

        # Tile action and proprio to all patches, then concatenate on the feature dimension
        visual_embs = z_dict["visual"]  # (B, T, N, 384)
        proprio_emb = z_dict["proprio"]  # (B, T, D_pro)

        N = visual_embs.shape[2]
        proprio_tiled = repeat(proprio_emb.unsqueeze(2), "b t 1 d -> b t n d", n=N)
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 d -> b t n d", n=N)

        z = torch.cat([visual_embs, proprio_tiled, act_tiled], dim=-1)  # (B, T, N, 404)
        return z

    def _rollout_fill_gt(self, z: torch.Tensor, visual_frames: torch.Tensor) -> torch.Tensor:
        """Predictor-stage rollout: fill the intermediate frames with GROUND TRUTH.

        The reference trainer sets `_gt_diff_rollout = 1`, so its `tools._get_rollout_z` takes the
        `preds = gt_diff[::-1]` branch and never calls the model:
            gt_diff = _diff_encode(z[:, i-1:i+1], ...)   -> DifferenceEncoder on the real frames
            z[:, i][..., :394] = _apply_diff(z[:, i-1], preds)
        i.e. frame i is rebuilt as  (previous frame's state)  with the GT next-frame features
        written at the GT pixel-diff foreground mask. No self-prediction at all.

        Args:
            z: (B, T, N, D) encoded sequence (will be replaced in-place on a clone)
            visual_frames: (B, T, C, H, W) encoder-transformed frames (for the pixel-diff mask)
        """
        B, T, N, D = z.shape
        d_state = D_VISUAL + D_PROPRIO_EMB
        z_real = z.detach().clone()
        z = z.clone()
        for t in range(1, T - 1):
            mask = self._generate_gt_mask_from_pixels(visual_frames[:, t - 1], visual_frames[:, t])
            z_next = z[:, t].clone()
            # start from the previous (already rebuilt) frame's state, then paste GT foreground
            z_next[..., :d_state] = z[:, t - 1, :, :d_state]
            gt_state = z_real[:, t, :, :d_state]
            z_next[..., :d_state][mask] = gt_state[mask]
            z[:, t] = z_next
        return z

    def forward(self, obs: dict[str, torch.Tensor], act: torch.Tensor):
        """
        Training forward pass. Returns (None, None, None, loss, loss_components).
        Compatible with existing orgtrain.py interface.
        """
        B, T = obs["visual"].shape[:2]

        # 1. Encode everything
        z = self.build_z(obs, act)  # (B, T, N, D_MODEL)

        # 2. Rollout augmentation (predictor stage): the intermediate frames are filled with the
        #    GROUND TRUTH next-frame features at the GT pixel-diff top-32 mask -- pure teacher
        #    forcing, matching the reference trainer's `_gt_diff_rollout = 1` path. The LRM stage
        #    instead uses self-prediction (see _lrm_rollout_augment).
        if self.training_stage == "predictor":
            _B, _T = obs["visual"].shape[:2]
            _vf = self.encoder_transform(rearrange(obs["visual"], "b t ... -> (b t) ..."))
            _vf = rearrange(_vf, "(b t) ... -> b t ...", b=_B)
            z = self._rollout_fill_gt(z, _vf)

        # 3. Prepare images for label generation (classifier stage needs pixel-level GT)
        images_current = None
        images_next = None
        if self.training_stage in ["classifier", "predictor"]:
            visual = obs["visual"]  # (B, T, C, H, W)
            visual_transformed = self.encoder_transform(rearrange(visual, "b t ... -> (b t) ..."))
            visual_transformed = rearrange(visual_transformed, "(b t) ... -> b t ...", b=B)
            images_current = visual_transformed[:, -2]  # (B, C, H, W)
            images_next = visual_transformed[:, -1]  # (B, C, H, W)

        # 4. Prepare predictor input (which may include rollout-replaced frames)
        #    - history_fusion_disabled: the predictor has no history encoder, so it must see a
        #      single frame (the last frame before the target); history_fusion is then an identity.
        #    - otherwise: exactly the 3 frames that the frozen history fusion consumes
        #      (the reference code sliced z to its last 4 frames and used the first 3 as history).
        if getattr(self, "history_fusion_disabled", False):
            z_history = z[:, -2:-1]  # (B, 1, N, D)
        else:
            z_history = z[:, -4:-1]  # (B, 3, N, D)
        z_next_gt = z[:, -1]  # (B, N, D) - ground truth next frame (never replaced)

        # 5. Call predictor (NOT for the LRM stage: its outputs would be discarded, but the call
        #    still burns compute -- a full ViT+LRM pass -- and consumes RNG for a mask that is
        #    never used, which shifts every subsequent dropout/mask realisation.)
        outputs = None
        if self.training_stage != "lrm":
            outputs = self.predictor(z_history, images_current, images_next)

        # 5. Compute stage-specific loss
        loss = torch.tensor(0.0, device=z.device)
        loss_components = {}

        if self.training_stage == "classifier":
            loss, loss_components = self._compute_classifier_loss(outputs)

        elif self.training_stage == "predictor":
            # Predictor recipe: MSE over all 196 positions
            # (background is zero in both the GT and the prediction)
            loss, loss_components = self._compute_dense_mse_loss(outputs, z_next_gt)

        elif self.training_stage == "lrm":
            # Multi-step rollout augmentation (the reference code enables it for this stage)
            if z.shape[1] > 2:
                z_next_gt_orig = z[:, -1].detach().clone()  # Save real target before modification
                # the LRM rollout uses the classifier mask, as the reference LRM stage does
                z = self._lrm_rollout_augment(z)
            else:
                z_next_gt_orig = z_next_gt

            # Final prediction using direct module calls (NO no_grad, for gradient flow)
            z_history_final = z[:, max(0, z.shape[1] - 1 - self.num_hist) : z.shape[1] - 1]
            _, outputs = self._predict_for_rollout(z_history_final, return_components=True)

            # Compute loss against REAL target (never replaced by rollout)
            loss, loss_components = self._compute_lrm_loss(outputs, z_next_gt_orig)

        loss_components["loss"] = loss
        return None, None, None, loss, loss_components

    def _compute_classifier_loss(self, outputs: dict[str, torch.Tensor]):
        """BCE loss on mask logits vs pixel-diff GT."""
        mask_logits = outputs["mask_logits"]  # (B, N, 4)
        gt_mask = outputs["gt_mask"]  # (B, N, 4) or None

        if gt_mask is None:
            return torch.tensor(0.0), {}

        loss = F.binary_cross_entropy_with_logits(
            mask_logits,
            gt_mask,
            pos_weight=torch.tensor(self.cls_pos_weight, device=mask_logits.device),
            reduction="mean",
        )
        return loss, {"loss_cls_bce": loss}

    def _compute_dense_mse_loss(self, outputs, z_next_gt):
        """Full MSE at ALL 196 positions (no mask on GT).

        The pretrained ViT predicts absolute next-frame features at ALL positions.
        It was NEVER trained to output zeros at background positions.
        Loss = MSE(pred_all[:394], z_next_gt[:394]) over all 196*394 elements.

        Verified: MSE(pred, gt_full) = 0.07 vs MSE(pred, gt*mask) = 5.2
        """
        # Construct pred and GT with mask: BG=zero at both (matching old code exactly)
        z_current = outputs["z_current"]
        mask = outputs["mask"]  # (B, N) bool
        pred_fg = outputs["pred_fg"]  # (B, K, D)
        B, N, D = z_current.shape
        pred_all = torch.zeros(B, N, D, device=z_current.device, dtype=z_current.dtype)
        pred_all[mask] = pred_fg.reshape(-1, D)

        d_state = D_VISUAL + D_PROPRIO_EMB  # 394

        # Full MSE: predict ALL positions' absolute features
        pred_state = pred_all[..., :d_state]  # (B, N, d_state)
        # Mask GT too: BG=zero (matching old code)
        gt_full = torch.zeros_like(z_next_gt[..., :d_state])
        mask_for_gt = mask.unsqueeze(-1).expand(-1, -1, d_state)
        gt_full[mask_for_gt] = z_next_gt[..., :d_state][mask_for_gt]

        loss = F.mse_loss(pred_state, gt_full.detach())

        return loss, {"loss_dense_mse": loss, "loss": loss}

    def _compute_lrm_loss(self, outputs: dict[str, torch.Tensor], z_next_gt: torch.Tensor):
        """MSE loss on full 196-token prediction vs GT (matching old code).

        Old code computes MSE over ALL 196 tokens (FG from frozen ViT + BG from LRM).
        FG positions have no gradient (ViT is frozen) but contribute to loss value.
        Gradients only flow to LRM through BG positions.
        """
        pred_bg = outputs["pred_bg"]  # (B, N-K, D)
        pred_fg = outputs["pred_fg"]  # (B, K, D)
        mask = outputs["mask"]  # (B, N) bool

        B, N, D = z_next_gt.shape
        d_state = D_VISUAL + D_PROPRIO_EMB  # 394

        # Combine FG + BG into full 196-token prediction (matching old code)
        pred_full = torch.zeros(B, N, D, device=z_next_gt.device, dtype=z_next_gt.dtype)
        pred_full[mask] = pred_fg.reshape(-1, D)
        pred_full[~mask] = pred_bg.reshape(-1, D)

        # MSE on all 196 tokens, first 394 dims (matching old code exactly)
        loss = F.mse_loss(pred_full[..., :d_state], z_next_gt[..., :d_state].detach())

        return loss, {"loss_lrm_mse": loss}

    def _predict_for_rollout(
        self, z_hist, return_components=False, single_frame=False, mask_dilate=0
    ):
        """Single-step prediction for LRM rollout training.

        NO torch.no_grad() on frozen modules - allows gradient to flow through
        for multi-step chains (matching old code where frozen params still allow
        input gradients to propagate).

        Args:
            z_hist: (B, h, N, D) history frames (h <= 3)
            return_components: if True, also return dict with pred_fg, pred_bg, mask
            single_frame: feed only the last frame (the LRM rollout recipe: history fusion is an
                identity); otherwise the last 3 frames go through the fusion.
            mask_dilate: connectivity used to dilate the (binarised) classifier mask before
                force_fixed_k; 0 = no dilation (the LRM rollout uses 4, the reference _infer value)
        Returns:
            z_next: (B, N, D) predicted next frame
            [optional] outputs: dict with pred_fg, pred_bg, mask, z_current
        """
        FUSION_CONTEXT = 3
        if single_frame:
            z_for_fusion = z_hist[:, -1:]
        else:
            z_for_fusion = (
                z_hist[:, -FUSION_CONTEXT:] if z_hist.shape[1] > FUSION_CONTEXT else z_hist
            )

        # History fusion (frozen params, gradient flows through input)
        z_t_prime = self.predictor.history_fusion(z_for_fusion)

        # Localization: the frozen Localizer, optionally with the reference _infer's mask
        # processing (binarise -> dilate(connectivity) -> force_fixed_k).
        with torch.no_grad():
            mask_logits = self.predictor.localizer(z_t_prime)
        if mask_dilate:
            # old tools._infer(): (pred_cls > 0.1) -> dilate(connectivity=4) -> force_fixed_k(32)
            from .ddp_predictor import dilate_mask_2d

            # Binarise first, then dilate: dilating the raw sub-region counts would change the
            # magnitudes that force_fixed_k ranks by, i.e. pick a different patch set.
            _active = ((mask_logits.sigmoid() > 0.5).sum(dim=-1) > 0).float()
            mask = self.predictor._process_mask_from_active(
                dilate_mask_2d(_active, connectivity=mask_dilate)
            )
        else:
            mask = self.predictor._process_mask(mask_logits)

        # Primary predictor (frozen params, gradient flows through input!)
        z_current = z_t_prime.squeeze(1)
        pred_fg = self.predictor.primary_predictor.vit(z_current, mask)

        # LRM (trainable!)
        updated_bg = self.predictor.lrm(z_current, mask, pred_fg)

        # Combine FG + BG
        B, N, D = z_current.shape
        z_next = torch.zeros(B, N, D, device=z_current.device, dtype=z_current.dtype)
        z_next[mask] = pred_fg.reshape(-1, D)
        z_next[~mask] = updated_bg.reshape(-1, D)

        if return_components:
            return z_next, {
                "pred_fg": pred_fg,
                "pred_bg": updated_bg,
                "mask": mask,
                "z_current": z_current,
            }
        return z_next

    def _lrm_rollout_augment(self, z):
        """Replace intermediate frames with the model's own predictions for LRM rollout training.

        Matches the reference `_get_rollout_z` with `_gt_diff_rollout = 0`: frames 1..T-2 are
        replaced with predictions (frame 0 and the target frame T-1 are kept real), the fallback
        source is the already-rebuilt previous frame, and every rollout step is predicted from the
        SINGLE real frame i-1 (`_infer(model, z_vis[:, i-1:i], ...)`, so the history fusion is an
        identity) with the mask going through binarise -> dilate(4) -> force_fixed_k(32).

        Safety net: tokens with L2 norm <= 10 fall back to the previous frame (matching _apply_diff).

        Gradient flows through the entire chain via LRM and the frozen-but-graph-tracked ViT: the
        reference code writes the rolled frames with in-place index assignment into a tensor that
        does not require grad, which installs a CopySlices grad_fn and back-propagates into the
        assigned value (verified on torch 2.3), so the LRM does see BPTT through the rollout chain.
        """
        B, T, N, D = z.shape
        if T <= 2:
            return z

        d_state = D_VISUAL + D_PROPRIO_EMB

        frames = [z[:, 0]]

        for i in range(1, T - 1):
            # NOTE the input frame: the reference `_infer` receives `z_vis[:, i-1:i]`, i.e. the
            # ORIGINAL (real) frame i-1 -- only the *fallback* frame (below, `prev_frame`) is the
            # already-rebuilt one.
            z_pred = self._predict_for_rollout(z[:, i - 1 : i], single_frame=True, mask_dilate=4)

            prev_frame = frames[-1]
            pred_state = z_pred[..., :d_state]
            prev_state = prev_frame[..., :d_state]
            norms = torch.norm(pred_state, p=2, dim=-1)
            keep_pred = (norms > 10.0).unsqueeze(-1)
            merged_state = torch.where(keep_pred, pred_state, prev_state)

            action_dims = z[:, i, :, d_state:]
            z_merged = torch.cat([merged_state, action_dims], dim=-1)

            # Gradient flow through the rolled frames (see the docstring).
            frames.append(z_merged)

        frames.append(z[:, -1])
        return torch.stack(frames, dim=1)

    @staticmethod
    def _apply_safety_net(z_pred_4d, z_current_4d):
        """
        Old code _apply_diff safety net: L2 norm <= 10 on state dims
        falls back to current frame. Truncates to d_state=394 then zero-pads.
        """
        d_state = D_VISUAL + D_PROPRIO_EMB  # 394
        z_pred = z_pred_4d.squeeze(1)
        z_curr = z_current_4d.squeeze(1)
        pred_state = z_pred[..., :d_state]
        curr_state = z_curr[..., :d_state]
        norms = torch.norm(pred_state, p=2, dim=-1)
        keep_pred = (norms > 10.0).unsqueeze(-1)
        merged = torch.where(keep_pred, pred_state, curr_state)
        zeros = torch.zeros(
            *merged.shape[:-1], D_ACTION_EMB, device=merged.device, dtype=merged.dtype
        )
        return torch.cat([merged, zeros], dim=-1).unsqueeze(1)

    def _generate_gt_mask_from_pixels(self, frame_current, frame_next, k=32):
        """Generate binary patch mask from GT pixel differences (matching old code DifferenceEncoder).

        Args:
            frame_current: (B, C, H, W) - encoder-transformed current frame
            frame_next: (B, C, H, W) - encoder-transformed next frame
            k: number of foreground patches to select
        Returns:
            mask: (B, N=196) bool tensor with exactly k True values per row
        """
        import torch.nn.functional as F_local

        pixel_diff = frame_next - frame_current  # (B, C, H, W)
        norms_sq = torch.sum(pixel_diff**2, dim=1, keepdim=True)  # (B, 1, H, W)
        # Pool to 14x14 patch grid (H=196 for 224px image with patch_size=16)
        H, W = frame_current.shape[2], frame_current.shape[3]
        patch_h = H // 14
        patch_w = W // 14
        pooled = F_local.avg_pool2d(
            norms_sq, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w)
        )
        rms = torch.sqrt(pooled).view(frame_current.shape[0], -1)  # (B, 196)
        # Select top-k patches by change magnitude, then force exactly k
        from models.ddp_predictor import force_fixed_k

        # Use threshold-based activation + force_fixed_k (matching old code)
        threshold = 0.1
        active = (rms > threshold).float()
        mask = force_fixed_k(active, k=k)
        return mask

    def rollout(self, obs_0, act):
        """
        Open-loop rollout for MPC evaluation.
        Sparse path: the frozen 32-token ViT predicts the foreground, the LRM the background.
        """
        if self.training_stage != "inference":
            raise RuntimeError(f"rollout() requires the inference stage, got {self.training_stage}")

        self.eval()
        num_obs_init = obs_0["visual"].shape[1]
        act_0 = act[:, :num_obs_init]
        z = self.build_z(obs_0, act_0)  # (B, num_obs_init, N, D)

        future_actions = act[:, num_obs_init:]
        T_future = future_actions.shape[1]

        for t in range(T_future):
            # The predictor stage decides how much history the network sees: 3 frames go through the
            # (frozen) history fusion, or a single frame when the model was trained with
            # history_fusion_disabled (the fusion is then an identity).
            z_hist = (
                z[:, -1:]
                if getattr(self, "history_fusion_disabled", False)
                else z[:, -self.num_hist :]
            )
            outputs = self.predictor(z_hist)
            z_pred_next = outputs["prediction"]  # (B, 1, N, D)

            # Safety net: low-norm tokens fall back to current frame
            z_pred_next = self._apply_safety_net(z_pred_next, z[:, -1:])

            # Inject next action
            next_act = future_actions[:, t : t + 1]
            act_emb = self.encode_action(next_act)
            N = z_pred_next.shape[2]
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 d -> b t n d", n=N)
            z_pred_next[..., -self.action_dim :] = act_tiled

            z = torch.cat([z, z_pred_next], dim=1)

        # One more prediction without new action
        z_hist = (
            z[:, -1:] if getattr(self, "history_fusion_disabled", False) else z[:, -self.num_hist :]
        )
        outputs = self.predictor(z_hist)
        z_pred_final = self._apply_safety_net(outputs["prediction"], z[:, -1:])
        z = torch.cat([z, z_pred_final], dim=1)

        # Separate visual embeddings
        z_visual = z[..., :D_VISUAL]
        z_proprio = z[:, :, 0, D_VISUAL : D_VISUAL + self.proprio_dim]
        z_obses = {"visual": z_visual, "proprio": z_proprio}
        return z_obses, z
