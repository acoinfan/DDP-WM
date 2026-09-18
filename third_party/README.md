# third_party

Attribution and license texts of the third-party components this repository builds on. This file
and the license texts next to it are the attribution record; the repository has no separate NOTICE
file.

| Component | License | Where it is used | Redistributed here? |
|---|---|---|---|
| DDP-WM reference implementation ("dino_wm") | MIT (`dino_wm/LICENSE`, Copyright (c) 2025 gaoyuezhou) | the pipeline this repository reproduces; the code in `models/`, `dsets/`, `env/` and `dino_planning/` is derived from it | Yes, as derived code |
| [DINOv2](https://github.com/facebookresearch/dinov2) (Meta Platforms) | Apache-2.0 (`dinov2/LICENSE`) | frozen visual encoder (`models/dino.py`, `dinov2_vits14`) | No - loaded at runtime via `torch.hub` |
| [PushT env](https://github.com/huggingface/gym-pusht) (Zac Wellmer) | MIT | `env/pusht/` (evaluation only) | Yes, as part of this repository |
| torch / torchvision / accelerate / einops / hydra-core / omegaconf / numpy / gym / pygame / pymunk / shapely / scikit-image / opencv-python / matplotlib / tqdm | various | training + evaluation | No - installed via `pip` |

The DINOv2 encoder is downloaded on first use into `~/.cache/torch/hub/facebookresearch_dinov2_main`.
If you redistribute this repository together with a DINOv2 checkout, keep `dinov2/LICENSE` next to it.
