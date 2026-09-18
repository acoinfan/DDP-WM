# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Fast smoke test for the two predictor recipes (--history-fusion-disabled / default).

Builds the real model + a small real data batch, runs a few optimisation steps of the
per-history-length loop exactly like train_ddpwm.py does, and prints diagnostics:
  - history tensor length seen by the predictor (1 with --history-fusion-disabled, else 3)
  - classifier mask size (must be K_MAX=32)
  - loss value, and gradient norm reaching primary_predictor.vit
"""

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dsets.img_transforms import default_transform
from dsets.pusht_dset import load_pusht_slice_train_val
from models.ddp_world_model import DDPWorldModel
from models.dino import DinoV2Encoder
from models.proprio import ProprioceptiveEmbedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["single", "fusion"],
        required=True,
        help="single = predictor without history fusion (1 frame), fusion = 3 frames",
    )
    ap.add_argument(
        "--classifier-ckpt",
        required=True,
        help="a classifier checkpoint you trained (e.g. "
        "runs/run_XXXX/checkpoints/classifier/model_latest.pth)",
    )
    ap.add_argument("--predictor-ckpt", default=None)
    ap.add_argument(
        "--data-path",
        default=os.environ.get("DDPWM_PUSHT_DATA"),
        help="PushT dataset directory (defaults to $DDPWM_PUSHT_DATA); required, the dataset is "
        "not shipped with the code",
    )
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-hist", type=int, default=5)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not args.data_path:
        ap.error("--data-path (or $DDPWM_PUSHT_DATA) is required: pass the PushT dataset directory")

    dev = torch.device(args.device)

    datasets, _ = load_pusht_slice_train_val(
        data_path=args.data_path,
        with_velocity=True,
        n_rollout=None,
        normalize_action=True,
        transform=default_transform(224),
        num_hist=args.num_hist,
        num_pred=1,
        frameskip=args.frameskip,
    )
    loader = torch.utils.data.DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    enc = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
    for p in enc.parameters():
        p.requires_grad = False
    ae = ProprioceptiveEmbedding(in_chans=datasets["train"].action_dim, emb_dim=10, tubelet_size=1)
    pe = ProprioceptiveEmbedding(in_chans=datasets["train"].proprio_dim, emb_dim=10, tubelet_size=1)

    model = DDPWorldModel(
        image_size=224,
        num_hist=args.num_hist,
        training_stage="predictor",
        encoder=enc,
        action_encoder=ae,
        proprio_encoder=pe,
        classifier_ckpt=os.path.join(_ROOT, args.classifier_ckpt),
        predictor_ckpt=args.predictor_ckpt,
    ).to(dev)

    # inherit trained action/proprio encoders from the classifier (same as train_ddpwm.py)
    cls = torch.load(
        os.path.join(_ROOT, args.classifier_ckpt), map_location="cpu", weights_only=True
    )
    model.action_encoder.load_state_dict(cls["action_encoder"])
    model.proprio_encoder.load_state_dict(cls["proprio_encoder"])

    model.history_fusion_disabled = args.mode == "single"

    for p in model.action_encoder.parameters():
        p.requires_grad_(False)
    for p in model.proprio_encoder.parameters():
        p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(
        f"trainable params: {n_tr:,} "
        f"(vit={sum(p.numel() for p in model.predictor.primary_predictor.parameters()):,})"
    )
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    model.train()
    seen_hist = set()
    it = iter(loader)
    for step in range(args.steps):
        obs, act, _state = next(it)
        obs = {k: v.to(dev) for k, v in obs.items()}
        act = act.to(dev)
        T_full = obs["visual"].shape[1]
        opt.zero_grad()
        total = 0.0
        for hist_len in range(1, T_full):
            Ts = hist_len + 1
            obs_s = {k: v[:, :Ts] for k, v in obs.items()}
            act_s = act[:, :Ts]
            # instrument: capture the history length actually fed to the predictor
            orig_forward = model.predictor.forward
            captured = {}

            def spy(z_history, *a, _captured=captured, _orig_forward=orig_forward, **kw):
                _captured["T"] = z_history.shape[1]
                return _orig_forward(z_history, *a, **kw)

            model.predictor.forward = spy
            _, _, _, loss_h, comp = model(obs_s, act_s)
            model.predictor.forward = orig_forward
            seen_hist.add(captured.get("T"))
            loss_h.backward()
            total += loss_h.detach().item()
        opt.step()
        gn = None
        for n, p in model.named_parameters():
            if "primary_predictor" in n and p.grad is not None:
                gn = p.grad.norm().item()
                break
        print(
            f"step {step}: mean_h_loss={total/(T_full-1):.5f} | hist lengths seen={sorted(seen_hist)} "
            f"| first-vit-grad-norm={gn:.3e} | mask in loss-components={list(comp.keys())}"
        )

    # mask size sanity check
    obs, act, _ = next(it)
    obs = {k: v.to(dev) for k, v in obs.items()}
    act = act.to(dev)
    with torch.no_grad():
        z = model.build_z(obs, act)
        if args.mode == "single":
            zh = z[:, -2:-1]
        else:
            zh = z[:, -4:-1]
        ztp = model.predictor.history_fusion(zh)
        mask = model.predictor._process_mask(model.predictor.localizer(ztp))
    print(f"mask True per row (should be {32}): {mask.sum(1).tolist()}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
