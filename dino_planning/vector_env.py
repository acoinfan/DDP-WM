# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
"""Serial (single-process) vector environment used by the planning evaluator."""

import numpy as np

from common.aggregate import aggregate_dct


class SerialVectorEnv:
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def sample_random_init_goal_states(self, seed):
        init_state, goal_state = zip(
            *(self.envs[i].sample_random_init_goal_states(seed[i]) for i in range(self.num_envs)),
            strict=False,
        )
        return np.stack(init_state), np.stack(goal_state)

    def update_env(self, env_info):
        for i in range(self.num_envs):
            self.envs[i].update_env(env_info[i])

    def eval_state(self, goal_state, cur_state):
        eval_result = []
        for i in range(self.num_envs):
            eval_result.append(self.envs[i].eval_state(goal_state[i], cur_state[i]))
        return aggregate_dct(eval_result)

    def prepare(self, seed, init_state):
        obs = []
        state = []
        for i in range(self.num_envs):
            o, s = self.envs[i].prepare(seed[i], init_state[i])
            obs.append(o)
            state.append(s)
        return aggregate_dct(obs), np.stack(state)

    def rollout(self, seed, init_state, actions):
        obses = []
        states = []
        for i in range(self.num_envs):
            obs, state = self.envs[i].rollout(seed[i], init_state[i], actions[i])
            obses.append(obs)
            states.append(state)
        return aggregate_dct(obses), np.stack(states)
