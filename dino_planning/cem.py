# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
import logging
import time

import numpy as np
import torch
from einops import repeat

from .base_planner import BasePlanner
from .objectives import compute_patch_mask, masked_visual_loss
from .utils import move_to_device

log = logging.getLogger(__name__)


class _RngRecorder:
    """Record `torch.rand` / `torch.randn` calls so they can be replayed later.

    Why this is needed: the world model draws from torch's *global* RNG inside its forward pass
    (`models.ddp_predictor.force_fixed_k` calls `torch.rand(B, N)`), so simply skipping the rollout
    of an episode would shift the stream and change the trajectories of every episode after it.
    `eval_episodes` replays the recorded draw pattern instead, which keeps a partial evaluation
    numerically identical to the full batch it was cut out of.
    """

    def __init__(self):
        self.calls = []
        self._orig = {}

    def __enter__(self):
        for name in ("rand", "randn"):
            orig = getattr(torch, name)
            self._orig[name] = orig
            setattr(torch, name, self._make_spy(name, orig))
        return self.calls

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(torch, name, orig)
        self._orig.clear()
        return False

    def _make_spy(self, name, orig):
        def spy(*args, **kwargs):
            out = orig(*args, **kwargs)
            self.calls.append((name, args, dict(kwargs)))
            return out

        return spy

    @staticmethod
    def replay(calls):
        for name, args, kwargs in calls:
            getattr(torch, name)(*args, **kwargs)


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        use_masked_heatmap=True,
        pixel_diff_thresh=0.1,
        mask_dilation=4,
        eval_episodes=None,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        # Masked (sparse) cost of the paper: score the visual term only on the patches where the
        # start and the goal frame differ. The mask is recomputed from the *current* start frame on
        # every CEM call (i.e. on every MPC round), exactly as the reference CEMPlanner does.
        self.use_masked_heatmap = use_masked_heatmap
        self.pixel_diff_thresh = pixel_diff_thresh
        self.mask_dilation = mask_dilation
        # Debug/eval aid: plan only these batch indices (see `eval_episodes` in the plan config).
        self.eval_episodes = None if eval_episodes is None else sorted({int(i) for i in eval_episodes})
        # Draw pattern of one rollout, learned once so the skipped episodes can replay it (see
        # _RngRecorder). `None` until the first skipped episode has been rolled out for real.
        self._rng_pattern = None

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        t_cem0 = time.perf_counter()
        t_prev = t_cem0

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                # `eval_episodes`: the candidate noise is still drawn for every episode of the full
                # batch (cheap, CPU side) so the draw order stays identical to a full run, while the
                # expensive world-model rollout is skipped for the episodes we do not evaluate. The
                # selected episodes therefore reproduce the trajectories of the full batch exactly.
                if self.eval_episodes is not None and traj not in self.eval_episodes:
                    action = (
                        torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                        * sigma[traj]
                        + mu[traj]
                    )
                    if self._rng_pattern is None:
                        # First skipped episode: run the rollout once to learn how much RNG a rollout
                        # consumes (the model draws internally), then replay that pattern from then on.
                        recorder = _RngRecorder()
                        with recorder as pattern:
                            self._rollout_candidates(
                                self._expand_trajectory(trans_obs_0, traj), action
                            )
                        self._rng_pattern = list(pattern)
                        log.info("[SUBSET] recorded %d RNG calls per rollout", len(self._rng_pattern))
                    else:
                        _RngRecorder.replay(self._rng_pattern)
                    continue
                cur_trans_obs_0 = self._expand_trajectory(trans_obs_0, traj)
                cur_z_obs_g = self._expand_trajectory(z_obs_g, traj)
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses = self._rollout_candidates(cur_trans_obs_0, action)

                if self.use_masked_heatmap:
                    # Patch mask from (this trajectory's current start frame, goal frame), broadcast
                    # over the candidate samples: only the regions that still differ are scored.
                    patch_mask = compute_patch_mask(
                        trans_obs_0["visual"][traj : traj + 1],
                        trans_obs_g["visual"][traj : traj + 1],
                        threshold=self.pixel_diff_thresh,
                        dilation=self.mask_dilation,
                    ).repeat(
                        self.num_samples, 1
                    )  # (num_samples, P)
                    loss_visual = masked_visual_loss(i_z_obses, cur_z_obs_g, patch_mask)

                    # The proprioceptive term keeps the objective's own weighting: the visual part is
                    # neutralised by scoring the goal features, so what is left is alpha * proprio.
                    goal_visual_full = cur_z_obs_g["visual"].expand(
                        -1, i_z_obses["visual"].shape[1], -1, -1
                    )
                    i_z_obses_for_proprio = dict(i_z_obses)
                    i_z_obses_for_proprio["visual"] = goal_visual_full
                    loss = loss_visual + self.objective_fn(i_z_obses_for_proprio, cur_z_obs_g)
                else:
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            t_now = time.perf_counter()
            log.info(
                f"[CEM {self.logging_prefix}] step {i+1}/{self.opt_steps} "
                f"dt={t_now - t_prev:.1f}s cum={t_now - t_cem0:.1f}s "
                f"mean_loss={np.mean(losses):.4f} |sigma|={float(sigma.norm()):.2f}"
            )
            t_prev = t_now
            if self.evaluator is not None and i % self.eval_every == 0:
                t_ev0 = time.perf_counter()
                logs, successes, _, _ = self.evaluator.eval_actions(mu)
                log.info(
                    f"[CEM {self.logging_prefix}] env-eval at step {i+1}: "
                    f"{time.perf_counter() - t_ev0:.1f}s success={np.mean(successes):.3f}"
                )
                t_prev = time.perf_counter()
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid

    def _expand_trajectory(self, per_eval_dict, traj):
        """Repeat every modality of one evaluation's observations for `num_samples` candidates."""
        return {
            key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
            for key, arr in per_eval_dict.items()
        }

    def _rollout_candidates(self, trans_obs_0, action):
        """Roll the world model forward for a batch of candidate action sequences.

        Candidates are independent, so the whole batch can be split into chunks that fit in GPU
        memory without changing the result (the world model sets `_rollout_chunk_size` on small
        GPUs). Returns the encoded trajectory dict of the rollout.
        """
        chunk = getattr(self.wm, "_rollout_chunk_size", None)
        if not (chunk and self.num_samples > chunk):
            i_z_obses, _ = self.wm.rollout(obs_0=trans_obs_0, act=action)
            return i_z_obses

        parts = []
        for start in range(0, self.num_samples, chunk):
            end = min(start + chunk, self.num_samples)
            chunk_obs = {k: v[start:end] for k, v in trans_obs_0.items()}
            z_part, _ = self.wm.rollout(obs_0=chunk_obs, act=action[start:end])
            parts.append(z_part)
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
