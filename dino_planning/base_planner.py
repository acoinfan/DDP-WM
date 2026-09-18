# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
import json
from abc import ABC, abstractmethod

import numpy as np


class BasePlanner(ABC):
    def __init__(
        self,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        log_filename,
        **kwargs,
    ):
        self.wm = wm
        self.action_dim = action_dim
        self.objective_fn = objective_fn
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device

        self.evaluator = evaluator
        self.log_filename = log_filename  # do not log if None

    def dump_logs(self, logs):
        logs_entry = {
            key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value)
            for key, value in logs.items()
        }
        if self.log_filename is not None:
            with open(self.log_filename, "a") as file:
                file.write(json.dumps(logs_entry) + "\n")

    @abstractmethod
    def plan(self):
        pass
