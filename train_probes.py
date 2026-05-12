"""Train linear evaluation probes for contact and success classification.

Four probes are trained and saved:
    head_contact_aligned.pt    — ContactProbe  using the aligned encoder checkpoint
    head_contact_baseline.pt   — ContactProbe  using a freshly-initialised encoder
    head_success_aligned.pt    — SuccessProbe  using the aligned encoder checkpoint
    head_success_baseline.pt   — SuccessProbe  using a freshly-initialised encoder

A ``probe_meta.json`` is written to ``output_dir`` recording the checkpoint path
and proprio z-scoring stats (mean_p / std_p) so that evaluate_probes.py can
reconstruct identical preprocessing without touching the training data.

Usage
-----
Full dataset, using best checkpoint from sweep:
    python train_probes.py --checkpoint best_run00_..._mse.pt

Custom options:
    python train_probes.py \\
        --checkpoint  best_run00_l0.10-1.00-1.00_lr0.0005_wd0.0001_mse.pt \\
        --dataset     dataset \\
        --output_dir  probes/ \\
        --epochs      15 \\
        --batch       64
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data import (
    ContactProbeDataset,
    SuccessProbeDataset,
    collect_episodes,
    compute_window_stats,
    split_episodes,
)
from training.probes import ContactProbe, SuccessProbe, build_encoders, train_probe


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear evaluation probes")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to EncoderTrainingModel state_dict (the aligned model checkpoint)",
    )
    parser.add_argument(
        "--dataset",    default="dataset",
        help="Path to the dataset root directory used to train the probes",
    )
    parser.add_argument("--output_dir", default="probes")
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--wd",         type=float, default=1e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect episodes and compute stats ────────────────────────────────────
    episodes = collect_episodes(args.dataset)
    stats    = compute_window_stats(episodes)
    train_eps, val_eps, _ = split_episodes(episodes, seed=args.seed)

    # ── Contact datasets (use all windows from each episode) ──────────────────
    contact_train_ds = ContactProbeDataset(train_eps, stats)
    contact_val_ds   = ContactProbeDataset(val_eps,   stats)

    # ── Success datasets (one window per episode — the final one) ─────────────
    success_train_ds = SuccessProbeDataset(train_eps, stats)
    success_val_ds   = SuccessProbeDataset(val_eps,   stats)

    print(
        f"\nContact dataset : train={len(contact_train_ds)} windows"
        f"  val={len(contact_val_ds)} windows"
        f"  (contact rate train="
        f"{100*contact_train_ds.n_contact/max(len(contact_train_ds),1):.1f}%)"
    )
    print(
        f"Success dataset : train={len(success_train_ds)} episodes"
        f"  val={len(success_val_ds)} episodes"
        f"  (success rate train="
        f"{100*success_train_ds.n_success/max(len(success_train_ds),1):.1f}%)"
    )

    pin = torch.cuda.is_available()

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=args.batch, shuffle=shuffle,
            num_workers=0, pin_memory=pin,
        )

    contact_train_loader = make_loader(contact_train_ds, shuffle=True)
    contact_val_loader   = make_loader(contact_val_ds,   shuffle=False)
    success_train_loader = make_loader(success_train_ds, shuffle=True)
    success_val_loader   = make_loader(success_val_ds,   shuffle=False)

    # ── Build aligned and baseline encoder models (frozen) ───────────────────
    aligned_model  = build_encoders(args.checkpoint, device)
    baseline_model = build_encoders(None, device)

    configs = [
        ("contact", "aligned",  aligned_model,  contact_train_loader, contact_val_loader),
        ("contact", "baseline", baseline_model, contact_train_loader, contact_val_loader),
        ("success", "aligned",  aligned_model,  success_train_loader, success_val_loader),
        ("success", "baseline", baseline_model, success_train_loader, success_val_loader),
    ]

    results = {}
    for task, model_name, enc_model, tr_loader, va_loader in configs:
        label = f"{task}_{model_name}"
        save_path = os.path.join(args.output_dir, f"head_{label}.pt")

        if task == "contact":
            probe = ContactProbe(enc_model)
        else:
            probe = SuccessProbe(enc_model)

        print(f"\n{'#'*60}")
        print(f"  Training probe: {label}")
        print(f"{'#'*60}")

        metrics = train_probe(
            probe        = probe,
            train_loader = tr_loader,
            val_loader   = va_loader,
            task         = task,
            save_path    = save_path,
            epochs       = args.epochs,
            lr           = args.lr,
            weight_decay = args.wd,
            patience     = args.patience,
            device       = device,
        )
        results[label] = metrics
        print(f"  Saved head → {save_path}")

    # ── Persist meta for evaluate_probes.py ──────────────────────────────────
    meta = {
        "checkpoint":    os.path.abspath(args.checkpoint),
        "dataset":       os.path.abspath(args.dataset),
        "mean_p":        stats["mean_p"].tolist(),
        "std_p":         stats["std_p"].tolist(),
        "seed":          args.seed,
        "probe_results": {
            k: {
                "best_f1":      v["best_f1"],
                "best_primary": v["best_primary"],
                "best_epoch":   v["best_epoch"],
            }
            for k, v in results.items()
        },
    }
    meta_path = os.path.join(args.output_dir, "probe_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nProbe meta saved → {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Probe':<28} {'best_f1':>8}  {'best_primary':>13}  epoch")
    print(f"  {'-'*56}")
    for label, m in results.items():
        task = label.split("_")[0]
        pname = "recall" if task == "contact" else "precision"
        print(
            f"  {label:<28} {m['best_f1']:>8.4f}  "
            f"{m['best_primary']:>13.4f}  {m['best_epoch']}"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
