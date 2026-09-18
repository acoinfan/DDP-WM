# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
import logging
import os

import numpy as np
import torch
from einops import rearrange

from common.aggregate import aggregate_dct

from .utils import move_to_device

log = logging.getLogger(__name__)


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        # Set by PlanWorkspace when only a subset of the batch is planned (`eval_episodes`): the
        # reported success rate is then computed over these indices only.
        self.report_indices = None

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(1)
        else:
            traj_data = np.expand_dims(traj_data[np.arange(traj_data.shape[0]), last_index], axis=1)
        return traj_data

    def eval_actions(self, actions, action_len=None):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(self.obs_0), self.device)
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip)
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        if self.report_indices:
            # Subset mode (`eval_episodes`): simulate only the selected episodes. The others get a
            # cheap copy of their initial frame, which keeps every downstream shape unchanged and
            # skips ~50 environment renders per CEM step. This cannot change the selected episodes:
            # each scene has its own seeded environment, and the world-model rollout above (which is
            # what consumes the shared torch RNG) still runs on the full batch.
            idx = np.asarray(self.report_indices)
            sub_obses = []
            sub_states = []
            for i in idx:
                obs_i, state_i = self.env.envs[int(i)].rollout(
                    self.seed[int(i)], self.state_0[int(i)], exec_actions[int(i)]
                )
                sub_obses.append(obs_i)
                sub_states.append(state_i)
            sub_obses = aggregate_dct(sub_obses)
            sub_states = np.stack(sub_states)
            length = sub_states.shape[1]
            e_obses = {key: np.repeat(self.obs_0[key][:, :1], length, axis=1) for key in sub_obses}
            e_states = np.repeat(self.state_0[:, None], length, axis=1)
            for j, i in enumerate(idx):
                e_states[i] = sub_states[j]
                for key in e_obses:
                    e_obses[key][i] = sub_obses[key][j]
        else:
            e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # Optional visual comparison of the imagined rollout against the environment rollout
        # (frame stacks with a goal column, optionally as mp4), kept as an empty hook.
        #
        # The reference implementation rendered the imagined latent trajectory back to pixels via
        # `self.wm.decode_obs(i_z_obses)` and plotted it in `_plot_rollout_compare(...)`; that needs
        # a VQVAE decoder on the world model, and this reproduction runs without one:
        # `load_ddpwm_model()` sets `result["decoder"] = None` for both model types, and
        # DDPWorldModel has no decoder module at all (the latent->pixel probe lives in
        # our internal tooling, not in this repository). Fill this in if a decoder is ever wired up.
        pass

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results["success"]

        logs = {
            "success_rate" if key == "success" else f"mean_{key}": (
                np.mean(value) if key != "success" else np.mean(value.astype(float))
            )
            for key, value in eval_results.items()
        }

        log.info(f"Success rate: {logs['success_rate']}")
        log.info(eval_results)

        if os.environ.get("DDPWM_LOG_MARGINS"):
            # Success requires pos_diff < 20 and angle_diff < pi/9 (see env/pusht/pusht_wrapper.py).
            # Log both components per episode so a failure can be compared against the threshold.
            state_g = np.asarray(self.state_g)
            e_state_arr = np.asarray(e_state)
            pos_diff = np.linalg.norm(state_g[:, :4] - e_state_arr[:, :4], axis=1)
            angle_diff = np.abs(state_g[:, 4] - e_state_arr[:, 4])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            log.info("[MARGINS] pos_diff=%s", np.round(pos_diff, 3).tolist())
            log.info("[MARGINS] angle_deg=%s", np.round(np.degrees(angle_diff), 3).tolist())

        if self.report_indices:
            idx = np.asarray(self.report_indices)
            selected_success = np.asarray(successes)[idx]
            log.info(
                "[SUBSET] episodes=%s success=%s sr=%.3f",
                list(self.report_indices),
                selected_success.tolist(),
                float(np.mean(selected_success.astype(float))),
            )

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update(
            {
                "mean_visual_dist": mean_visual_dist,
                "mean_proprio_dist": mean_proprio_dist,
                "mean_div_visual_emb": div_visual_emb,
                "mean_div_proprio_emb": div_proprio_emb,
            }
        )

        return logs, successes
