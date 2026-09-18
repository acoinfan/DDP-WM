#!/bin/bash
# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
# Train/eval consistency check: one short run of each training stage, chained together
# (classifier -> predictor -> LRM, all from a random initialisation) plus one evaluation of the LRM
# it just trained, printing a compact summary that can be compared before/after a refactor.
#
#   bash tests/consistency_check.sh [--seconds 60] [--gpu 0] [--eval-seed 1]
#   EVAL_CKPT=<ckpt> bash tests/consistency_check.sh --eval-only     # only the evaluation
#   (add EVAL_CLS_CKPT=<classifier.pth> when <ckpt> does not contain the localizer itself, e.g. a
#    converted reference checkpoint; checkpoints trained here bundle it)
#
# What it checks:
#   * classifier stage runs and writes a CSV with h1..h3 losses
#   * predictor  stage (--history-fusion-disabled, warm-started from that classifier) -> h1..h5
#   * LRM        stage (warm-started from that predictor)                              -> h1..h5
#   * evaluation (sparse cost, 1 scene) of the LRM checkpoint this run produced
#
# It needs no external checkpoint (everything is trained here), only the dataset: training reads the
# `train` split, the evaluation the `val` split of the PushT data configured in
# `conf/plan_pusht.yaml` / `$DDPWM_PUSHT_DATA`.
#
# Reproducibility (measured on titan, seed 0, batch 8): without --deterministic the PREDICTOR stage
# and the evaluation are bit-reproducible, while the CLASSIFIER and LRM stages drift between runs
# (cuDNN picks algorithms per run) - two runs of the same tree differ by ~0.01 at step 100 for the
# classifier. Compare numbers only within the same machine.
# Pass --deterministic (torch deterministic kernels) when comparing two code revisions, and collect
# the whole chain in one call so both stages read the same dataset on the same GPU.
set -u
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
P=${PY:-${DDPWM_PYTHON:-python3}}

"$P" -c "import torch, gym, hydra" 2>/dev/null || {
  echo "ERROR: '$P' cannot import torch/gym/hydra." >&2
  echo "       Activate the project environment, or run with PY=<path to its python>." >&2
  exit 2
}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
GPU=${GPU:-0}
EVAL_SEED=${EVAL_SEED:-1}
EVAL_ONLY=0
DETERMINISTIC=${DETERMINISTIC:-false}

while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_PER_RUN="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --eval-seed) EVAL_SEED="$2"; shift 2;;
    --deterministic) DETERMINISTIC=true; shift;;
    --eval-only) EVAL_ONLY=1; shift;;   # machines without the full training data (e.g. a laptop)
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

# training stages take --deterministic as a flag, the evaluation as `deterministic=true`
if [ "$DETERMINISTIC" = "true" ]; then DET_ARG=(--deterministic); else DET_ARG=(); fi

RUN_ROOT=$(mktemp -d /tmp/ddpwm_consistency.XXXXXX)
trap 'rm -rf "$RUN_ROOT"' EXIT

run_stage() {                    # $1=tag  $2=stage-log-name  (rest: extra args)
  local tag=$1 logname=$2; shift 2
  mkdir -p "$RUN_ROOT/$tag/checkpoints/stage" "$RUN_ROOT/$tag/csv_log"
  # --save-every-n-steps: the run is killed before the epoch ends, and the next stage needs the
  # checkpoint this one produced (50 steps is reached even by the slowest stage within the budget).
  CUDA_VISIBLE_DEVICES=$GPU timeout $((SECONDS_PER_RUN + 20)) "$P" train_ddpwm.py \
      "$@" ${DET_ARG[@]+"${DET_ARG[@]}"} --epochs 1 --batch-size 8 --save-every-n-steps 50 \
      --output-dir "$RUN_ROOT/$tag/checkpoints/stage" \
      > "$RUN_ROOT/$tag.log" 2>&1
  local csv="$RUN_ROOT/$tag/csv_log/train_log_${logname}.csv"
  if [ -f "$csv" ]; then
    # Report the row at a fixed step so before/after refactor comparisons are stable
    # (the number of steps completed within the time budget varies slightly).
    echo "  $tag: $(wc -l < "$csv") rows | step100: $(awk -F, '$2==100' "$csv" | head -1)"
    echo "  $tag: last: $(tail -1 "$csv")"
  else
    echo "  $tag: FAILED - see $RUN_ROOT/$tag.log"
    grep -m3 -E "Error|Traceback|error" "$RUN_ROOT/$tag.log" | sed 's/^/      /'
  fi
}

echo "== consistency check (gpu=$GPU, ${SECONDS_PER_RUN}s per stage) =="
CLS_CKPT="$RUN_ROOT/classifier/checkpoints/stage/model_latest.pth"
PRED_CKPT="$RUN_ROOT/predictor/checkpoints/stage/model_latest.pth"
LRM_CKPT="$RUN_ROOT/lrm/checkpoints/stage/model_latest.pth"
if [ "$EVAL_ONLY" = "1" ]; then
  echo "training stages: skipped (--eval-only)"
else
  echo "training stages (from scratch; each stage is fed by the previous one):"
  run_stage classifier  classifier          --stage classifier
  run_stage predictor   predictor           --stage predictor --history-fusion-disabled \
                                            --classifier-ckpt "$CLS_CKPT"
  run_stage lrm         lrm                 --stage lrm \
                                            --classifier-ckpt "$CLS_CKPT" \
                                            --predictor-ckpt "$PRED_CKPT"
fi

echo "evaluation:"
# What to evaluate: the LRM trained above, or -- with --eval-only -- $EVAL_CKPT.
EVAL_CKPT="${EVAL_CKPT:-$LRM_CKPT}"
if [ ! -f "$EVAL_CKPT" ]; then
  echo "  skipped: no checkpoint to evaluate (pass EVAL_CKPT=<path> with --eval-only)"
else
  EVAL_DIR="$RUN_ROOT/eval"
  mkdir -p "$EVAL_DIR"
  CUDA_VISIBLE_DEVICES="$GPU" "$P" -m dino_planning.plan \
      ddpwm_ckpt="$EVAL_CKPT" seed="$EVAL_SEED" n_evals=1 \
      ${EVAL_CLS_CKPT:+ddpwm_cls_ckpt="$EVAL_CLS_CKPT"} \
      deterministic="$DETERMINISTIC" \
      use_sparse_cost=true eval_dir="$EVAL_DIR" hydra.run.dir="$EVAL_DIR" \
      > "$EVAL_DIR/eval.log" 2>&1
  # numpy 2.x prints values as np.float64(1.0) / np.float32(14.28); strip that wrapper so the same
  # parsing works on every machine.
  extract() { grep -o "$1'[:,][^,}]*" "$EVAL_DIR/eval.log" | tail -1 | sed 's/np\.float[0-9]*(//; s/)//'; }
  echo "  ckpt $EVAL_CKPT"
  SR=$(extract success_rate)
  STATE_DIST=$(extract mean_state_dist)
  echo "  SR $SR"
  echo "  state_dist $STATE_DIST"
  # A missing number means the evaluation died (wrong/missing data path, ...). Report it as a
  # failure instead of printing empty values that look like a completed run.
  if [ -z "$SR" ] || [ -z "$STATE_DIST" ]; then
    echo "  FAIL: the evaluation produced no result -- see $EVAL_DIR/eval.log"
    grep -m3 -E "Error|Traceback|error" "$EVAL_DIR/eval.log" | sed 's/^/      /'
    echo "  (training stages above may still be fine; this is the evaluation step)"
    echo "== FAILED =="
    exit 1
  fi
fi
echo "== done =="
