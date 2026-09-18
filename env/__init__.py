# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
# Gym registration for the simulated environment used by planning/evaluation.
#
# Only PushT ships a gym environment in this repository (env/pusht/). The other datasets in
# conf/env/<env>.yaml (point_maze / wall / deformable_env) can be trained on -- their dataloaders
# live in dsets/ -- but their gym wrappers were not carried over, so there is nothing to register
# for them and plan-time evaluation is supported for PushT only.
from gym.envs.registration import register

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
