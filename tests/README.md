# `tests/` — what to run, and how

Two kinds of checks live here:

* **fast tests** (`test_*.py`) — CPU only, no dataset, no DINOv2 download, a few seconds. They cover
  the contracts that break silently: stage configs, checkpoint layout, module imports and the
  forward/backward path of all three training stages.
* **the end-to-end consistency check** (`consistency_check.sh`) — trains each stage for a minute and
  runs one evaluation, printing a small summary you can compare before/after a change. This is the
  check to run after touching anything in `models/`, `dsets/`, `dino_planning/` or the trainer.

## Fast tests

```bash
python tests/run_all.py          # no pytest needed
pytest tests/                    # or, if pytest is installed
python tests/run_all.py --list   # what would run
```

| file | what it checks |
|---|---|
| `test_imports.py` | every module the pipeline imports imports cleanly; `env` registers the `pusht` gym id |
| `test_configs.py` | `conf/ddpwm.yaml` has all three stage sections with the right `num_hist`; every `conf/env/<env>.yaml` has the keys `build_dataset` reads and no dead `gym:` block; `resolve_config` rejects a config without the requested stage and the old `primary_predictor:` section name |
| `test_model_forward.py` | one training step (forward + backward) for classifier / predictor (both history-fusion variants) / lrm with a stub encoder; the rollout sees 1 frame or 3 frames according to the recipe |
| `test_ckpt_layout.py` | every checkpoint you drop into `pretrained/` (e.g. a released one you want to re-verify) uses the current key layout, carries the recipe flag, and loads into the model without unexpected keys; skipped when that directory is empty, which is the normal case |
| `test_ckpt_override.py` | the evaluation override order (classifier > predictor > LRM) implemented by `dino_planning.plan.merge_checkpoint_layers` |
| `test_ckpt_load_policy.py` | every `torch.load` in the tree passes `weights_only` explicitly (so the behaviour cannot change with the installed torch version), `models/ckpt_io.load_checkpoint` fills it in, and only the DINO-WM reference loader asks for `False` |
| `test_planning_math.py` | the maths that decides the success rate, on CPU: the sparse patch mask + dilation, the sparse objective, CEM's `init_mu_sigma`, the MPC success mask, and the `env/` -> `dino_planning/` layering rule |

## Consistency check (needs a GPU + the dataset)

```bash
PY=<python> bash tests/consistency_check.sh --seconds 60 --gpu 0
EVAL_CKPT=<lrm.pth> PY=<python> bash tests/consistency_check.sh --eval-only   # only the evaluation
# for a checkpoint that does not contain the localizer itself (e.g. a converted reference LRM
# checkpoint) add the classifier it was trained with:
EVAL_CKPT=<lrm.pth> EVAL_CLS_CKPT=<classifier.pth> PY=<python> bash tests/consistency_check.sh --eval-only
```

What it does: one short run of each training stage **from a random initialisation**, chained
(classifier feeds the predictor, the predictor feeds the LRM; each writes a CSV of the
per-history-length losses), then one 1-scene `dino_planning` evaluation of the LRM it just trained,
and prints a compact summary. It needs no checkpoint of ours -- only the dataset (the `train` split
for training, the `val` split for the evaluation).

Reference run (seed 0, batch 8, single GPU), for scale:

* predictor stage, step 100 → `1,100,0.327659,...`
* evaluation of a trained LRM (seed 1, 1 scene) → SR 1.0, `state_dist` 14.28966

Absolute values differ between machines, so compare only within one machine.

A 60-second-per-stage run from scratch is a *smoke*: the numbers it prints have no reference value,
they are meant to be compared against the same run before/after a code change.

The predictor stage and the evaluation are bit-reproducible; the classifier and LRM stages drift
(cuDNN picks algorithms per run). To make the run bit-comparable add `--deterministic`, which the
script forwards to every training stage and to the evaluation:

```bash
PY=<python> bash tests/consistency_check.sh --seconds 60 --gpu 0 --deterministic
```

The LRM keeps a ~1e-5 spread even then (a few of its ops have no deterministic kernel).

`smoke_predictor.py` is the older, narrower version of the predictor part of that check (it prints
the mask size and the ViT gradient norm); it is kept because it isolates the predictor stage when
only that stage is in question:

```bash
python tests/smoke_predictor.py --mode single|fusion \
    --classifier-ckpt runs/run_XXXX/checkpoints/classifier/model_latest.pth
```
