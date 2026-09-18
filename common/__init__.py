# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Small helpers shared by the environment and the planner layers.

This package exists so that neither layer has to import the other: `env/` (the gym environment) and
`dino_planning/` (the planner) both depend on `common/`, which itself depends on nothing but numpy
and torch.
"""
