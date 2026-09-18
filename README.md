# **DDP-WM**: Disentangled Dynamics Prediction for Efficient World Models
[[Paper]](https://arxiv.org/abs/2602.01780) [[Code]](https://github.com/HCPLab-SYSU/DDP-WM) [[Data]](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)

Shicheng Yin, Kaixuan Yin, Weixing Chen, Yang Liu, Guanbin Li and Liang Lin, Sun Yat-sen University

![DDP-WM Framework Overview](assets/model.png)

This repository is the official implementation of **DDP-WM (Disentangled Dynamics Prediction World Model)**, a novel framework designed to tackle the efficiency-performance bottleneck in existing world models. We observe that in most physical interaction scenarios, scene evolution can be decomposed into **sparse primary dynamics** driven by physical interactions and **context-driven background updates**.

Unlike dense models such as DINO-WM, DDP-WM employs a four-stage decoupled process for efficient modeling: it identifies foreground changing regions via **dynamic localization**, focuses computational resources on them using a **primary predictor** for high-precision forecasting, and leverages an innovative **Low-Rank Correction Module (LRM)** to update the background at a minimal cost. This design significantly improves computational efficiency while also providing a smoother optimization landscape for the planner, leading to superior planning success rates across various tasks.

## Acknowledgement

Our codebase is refactored and developed upon the excellent [DINO-WM](https://github.com/gaoyuezhou/dino-wm) project. We sincerely thank the authors of DINO-WM for their great work and for open-sourcing their code.

## Project Status

- [x] Core Model Architecture (`DDP_Predictor`, `DDPWorldModel`)
- [x] Staged Training Script (`train_ddpwm.py`)
- [x] Hydra Configuration Files (`conf/`)
- [x] Planning and Evaluation Scripts (`dino_planning/plan.py`)
- [ ] Pre-trained Model Checkpoints

## Getting Started

1.  [Installation](#installation)
2.  [Datasets](#datasets)
3.  [Train a DDP-WM Model](#train-a-ddp-wm-model)
4.  [Plan with a DDP-WM Model](#plan-with-a-ddp-wm-model)

## Installation

Our codebase is refactored from the DINO-WM project, but this repository is self-contained:
everything here runs on PushT, so no Conda, MuJoCo or PyFlex setup is required and a plain virtual
environment is enough. You need Python ≥ 3.10 and a CUDA build of PyTorch

```bash
git clone https://github.com/HCPLab-SYSU/DDP-WM.git
cd DDP-WM

pip install -e .               # the only line you need to train
pip install -e '.[env]'        # + what the PushT gym environment needs, for planning/evaluation
pip install -e '.[env,dev]'    # + the test runner: python tests/run_all.py
```

Pick the line that matches what you want to do: `pip install -e .` is enough to **train**, `[env]`
adds the dependencies the planner and the evaluator need, `[dev]` adds pytest for `tests/run_all.py`.
The install is editable, so the code keeps running from this tree. If your PushT data is stored as
`obses/*.mp4` videos rather than pre-extracted PNG frames, also add the `[datasets]` extra
(`pip install -e '.[env,dev,datasets]'`), which pulls `decord` for video decoding.

## Datasets

The datasets used in this project are the same as those for DINO-WM and can be downloaded from
**[here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)**.

After downloading and unzipping, point a single environment variable at the dataset root:

```bash
export DDPWM_DATA_DIR=/path/to/data     # the directory that contains pusht_noise/
```

Training reads `<root>/pusht_noise/{train,val}`; planning and evaluation use the same directory and
only its `val` split (the goal states are sampled from it). There is nothing separate to configure
for evaluation. If your PushT directory lives somewhere else, override it explicitly with
`DDPWM_PUSHT_DATA=<.../pusht_noise>` or `data_path=<.../pusht_noise>` on the planning command line.

The expected directory structure is as follows:
```
data
├── deformable
│   ├── granular
│   └── rope
├── point_maze
├── pusht_noise
│   ├── train
│   └── val
└── wall_single
```

## Train a DDP-WM Model

The training of DDP-WM is conducted in a staged, decoupled manner to ensure stability and reproducibility. You need to train the different components of the model sequentially. Every stage is trained on top of the previous one, so the checkpoint of a stage also carries the weights of all earlier stages; pass them with `--classifier-ckpt` / `--predictor-ckpt`.

### Stage 1: Train the Dynamic Localization Network

This stage trains the **Historical Information Fusion Module** and the **Dynamic Localization Network**, together with the action and proprioception encoders that every later stage inherits.
```bash
python -m torch.distributed.run --nproc_per_node=8 train_ddpwm.py \
    --stage classifier --epochs 2 --lr 7e-4 \
    --output-dir runs/run_XXXX/checkpoints/classifier
```
Checkpoints are saved under the `--output-dir` you give: the trainer writes `model_latest.pth`, and with `--save-every-n-steps` also resumable mid-epoch saves that `--resume` can pick up.

### Stage 2: Train the Sparse Primary Dynamics Predictor

This stage trains the **Primary Dynamics Predictor**. We freeze the weights from the previous stage and use the generated sparse masks to guide the predictor. It is run in two steps: first the ViT is bootstrapped on a single frame with a randomly initialised backbone, then it is warm-started from that checkpoint and fed through the frozen 3-frame history fusion inherited from Stage 1. You need to specify the checkpoint from Stage 1 via `--classifier-ckpt`.

```bash
# 2a) single frame, random ViT initialisation
python -m torch.distributed.run --nproc_per_node=8 train_ddpwm.py \
    --stage predictor --history-fusion-disabled \
    --classifier-ckpt runs/run_XXXX/checkpoints/classifier/model_latest.pth \
    --epochs 2 --lr 7e-4 --save-every-n-steps 1000 \
    --output-dir runs/run_XXXX/checkpoints/predictor_single_frame

# 2b) frozen 3-frame history fusion, warm start from 2a
python -m torch.distributed.run --nproc_per_node=8 train_ddpwm.py \
    --stage predictor \
    --classifier-ckpt runs/run_XXXX/checkpoints/classifier/model_latest.pth \
    --predictor-ckpt runs/run_XXXX/checkpoints/predictor_single_frame/model_latest.pth \
    --epochs 2 --lr 7e-4 --save-every-n-steps 5000 \
    --output-dir runs/run_YYYY/checkpoints/predictor
```

### Stage 3: Train the Low-Rank Correction Module (LRM)

Finally, we train the **LRM** to update the background. In this stage, all modules from the previous two stages are frozen.

```bash
python -m torch.distributed.run --nproc_per_node=8 train_ddpwm.py \
    --stage lrm \
    --classifier-ckpt runs/run_XXXX/checkpoints/classifier/model_latest.pth \
    --predictor-ckpt runs/run_YYYY/checkpoints/predictor/model_latest.pth \
    --epochs 2 --lr 7e-4 --save-every-n-steps 5000 \
    --output-dir runs/run_ZZZZ/checkpoints/lrm
```

After these three stages, you will have a fully trained DDP-WM model.

## Plan with a DDP-WM Model

Planning is model-predictive control with CEM (`MPCPlanner` + `CEMPlanner`) and the sparse cost of the paper. Fifty scenes per seed is the protocol we report:

```bash
python -m dino_planning.plan \
    ddpwm_ckpt=runs/run_ZZZZ/checkpoints/lrm/model_latest.pth \
    seed=99 n_evals=50 use_sparse_cost=true \
    eval_dir=runs/run_ZZZZ/planning/s99 hydra.run.dir=runs/run_ZZZZ/planning/s99
```

The per-episode and mean success rate are printed at the end of the run (and written to `result.json` in `eval_dir`). Three notes:

* the checkpoints we trained are not distributed with this repository; the evaluation above takes a
  checkpoint produced by the training stages (or one you converted yourself);
* a checkpoint trained with this repository already contains the localizer and encoders, so no extra argument is needed; a checkpoint converted from another codebase only carries the LRM, in which case pass the classifier it was trained with via `ddpwm_cls_ckpt=<classifier.pth>`;
* pass `deterministic=true` (and `--deterministic` when training) to make a run bit-comparable with another run on the same machine.

## Citation

If you find our work useful, please consider citing our paper:
```bibtex
@misc{yin2026ddpwmdisentangleddynamicsprediction,
      title={DDP-WM: Disentangled Dynamics Prediction for Efficient World Models}, 
      author={Shicheng Yin and Kaixuan Yin and Weixing Chen and Yang Liu and Guanbin Li and Liang Lin},
      year={2026},
      eprint={2602.01780},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.01780}, 
}
```

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)). The license texts and the
attribution of the projects it builds on — the DINO-WM reference implementation, DINOv2 and the
PushT environment — are collected in [`third_party/`](third_party/README.md).
