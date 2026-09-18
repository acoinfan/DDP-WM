# conf/ - seven files, all of them in use

| File | Read by | Purpose |
|---|---|---|
| `ddpwm.yaml` | `train_ddpwm.py` (default `--config`) | Training configuration: `common` (env / data_dir / batch_size / num_workers / frameskip), the three stages (`classifier`, `predictor`, `lrm`) and `logging`. Only keys that are actually read are listed; constants that are hard-coded in the code are documented in the file header. |
| `env/pusht.yaml` | `train_ddpwm.load_env_config()` | Dataset wiring for one environment: `data_root` / `data_subdir` / `dataset._target_` + arguments. (Only PushT also has a gym environment, so `python -m dino_planning.plan` evaluates PushT only.) |
| `env/point_maze.yaml` | same | Same, for point_maze. |
| `env/wall.yaml` | same | Same, with `split_mode=random`. |
| `env/deformable_env.yaml` | same | Same, with `object_name=rope` (override via `--object-name`). |
| `plan_pusht.yaml` | `python -m dino_planning.plan` (hydra `config_name`) | Planning/evaluation: planner (MPC + CEM), objective, sparse cost, goal sampling. `ddpwm_ckpt` deliberately has no default, so stale weights cannot be evaluated by accident. |
| `example_run.yaml` | not loaded automatically | Fully commented example of a run configuration (equivalent to `ddpwm.yaml`), plus the full commands for all three stages and the list of hard-coded constants. |

## Conventions

1. **Training**: `ddpwm.yaml` provides stage defaults, the command line overrides them. Dataset
   wiring comes from `env/<env>.yaml`; a run configuration can override individual keys with its
   optional top-level `env:` block.
2. **Evaluation**: run `python -m dino_planning.plan ddpwm_ckpt=<ckpt> seed=<n> n_evals=<n>
   use_sparse_cost=true eval_dir=<dir> hydra.run.dir=<dir>`; give every evaluation its own
   directory (that is where `eval.log`, `result.json`, `targets.pkl` and `logs.json` land).
3. **No decorative keys**: every key left in `ddpwm.yaml` is read by `train_ddpwm.py`. Values that
   are hard-coded in the code (`image_size=224`, frozen DINOv2 ViT-S/14, 10-dim embeddings,
   `k_max=32`, `n_patches=196`, per-stage depth/heads, ...) are only documented in comments;
   changing them in YAML has no effect.
4. Legacy configurations of the original reference pipeline are archived outside the repository
   (see `third_party/README.md` and the reproduction notes); they are intentionally not shipped here.
