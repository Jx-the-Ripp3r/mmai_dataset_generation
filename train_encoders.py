"""Top-level training script for the windowed encoder model.

Usage
-----
Full dataset, default sweep:
    python train_encoders.py

Override dataset root (e.g., in Colab after mounting Drive):
    python train_encoders.py --dataset /content/drive/MyDrive/dataset

Single run with specific lambdas (skips the sweep):
    python train_encoders.py --lambdas 0.1 1.0 1.0 --lr 5e-4 --wd 1e-4

Other flags:
    --epochs     30          max epochs per run
    --patience   7           early-stopping patience
    --max_min    30          wall-clock budget per run (minutes)
    --batch      64          batch size
    --output_dir .           where to save checkpoints
    --seed       42
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data import (
    collect_episodes,
    compute_window_stats,
    split_episodes,
    WindowedRoboticsDataset,
)
from training.train import run_lambda_sweep, train_encoder_model, SWEEP


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    dataset_root: str,
    batch_size:   int,
    seed:         int,
):
    """Collect episodes → compute stats → split → build DataLoaders."""
    episodes = collect_episodes(dataset_root)
    stats    = compute_window_stats(episodes)
    train_eps, val_eps, eval_eps = split_episodes(episodes, seed=seed)

    train_ds = WindowedRoboticsDataset(train_eps, stats)
    val_ds   = WindowedRoboticsDataset(val_eps,   stats)

    eval_ds = WindowedRoboticsDataset(eval_eps, stats)
    print(
        f"\nDataset sizes:  train={len(train_ds)} windows  "
        f"val={len(val_ds)} windows  eval(unused)={len(eval_ds)}"
    )
    print(
        f"Contact windows: train={train_ds.n_contact} "
        f"({100*train_ds.n_contact/max(len(train_ds),1):.1f}%)  "
        f"val={val_ds.n_contact} "
        f"({100*val_ds.n_contact/max(len(val_ds),1):.1f}%)"
    )

    # num_workers=0 for Windows / Colab compatibility; pin_memory only on CUDA
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train windowed encoder model")
    parser.add_argument("--dataset",    default="dataset",
                        help="Path to the dataset root directory")
    parser.add_argument("--output_dir", default=".",
                        help="Directory to save checkpoints")
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--max_min",    type=float, default=30.0,
                        help="Wall-clock budget per run (minutes)")
    parser.add_argument("--seed",       type=int,   default=42)
    # Single-run overrides (if both supplied, skips the full sweep)
    parser.add_argument("--lambdas",    type=float, nargs=3, default=None,
                        metavar=("L1", "L2", "L3"),
                        help="Override lambdas for a single run")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--wd",         type=float, default=None)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(args.dataset, args.batch, args.seed)

    if args.lambdas is not None:
        # Single run
        lambdas = tuple(args.lambdas)
        lr      = args.lr      if args.lr  is not None else 5e-4
        wd      = args.wd      if args.wd  is not None else 1e-4
        name    = f"single_l{lambdas[0]:.2f}-{lambdas[1]:.2f}-{lambdas[2]:.2f}_lr{lr}_wd{wd}"
        train_encoder_model(
            train_loader   = train_loader,
            val_loader     = val_loader,
            lambdas        = lambdas,
            lr             = lr,
            weight_decay   = wd,
            epochs         = args.epochs,
            patience       = args.patience,
            max_minutes    = args.max_min,
            save_path      = os.path.join(args.output_dir, f"best_{name}.pt"),
            model_name     = name,
            device         = device,
        )
    else:
        # Full hyperparameter sweep
        run_lambda_sweep(
            train_loader   = train_loader,
            val_loader     = val_loader,
            sweep          = SWEEP,
            epochs         = args.epochs,
            patience       = args.patience,
            max_minutes    = args.max_min,
            output_dir     = args.output_dir,
            device         = device,
        )


if __name__ == "__main__":
    main()
