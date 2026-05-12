"""Generate a series of noisy evaluation splits for probe degradation curves.

Each split ("level") runs the full PyBullet simulation with noise parameters
scaled per a configurable multiplier ``m``.

Per-parameter linear scheduling (NOT uniform multiplication)
------------------------------------------------------------
Each noise parameter has its own ``(value_at_m1, value_at_m10)`` schedule and
is linearly interpolated for arbitrary ``m``.  Force noise is fixed at 0 —
force readings drive the contact LABEL (``|F| > contact_force_threshold``) and
provide the force-direction supervision used during encoder training; adding
noise corrupts the labels themselves rather than the observations.

Layout written to disk::

    <output>/
        level_1/episode_0000/ ...
        level_2/episode_0000/ ...
        ...

Usage
-----
Default (10 levels × 100 noisy episodes):
    python generate_noise_sweep.py

Custom levels and episode count:
    python generate_noise_sweep.py \\
        --levels 1 2 3 4 5 6 7 8 9 10 \\
        --episodes_per_level 200 \\
        --output dataset_noise_sweep \\
        --seed 123

Integer or float multipliers are both accepted:
    python generate_noise_sweep.py --levels 1 1.5 2 3 5 10
"""

import argparse
import copy
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

from config import DatasetConfig
from generate_dataset import ensure_assets, run_dataset
from sim.controller import SimpleDownwardController
from sim.environment import PegInsertionEnv


# ── Per-parameter noise schedule ──────────────────────────────────────────────
# Each entry is (value_at_m1, value_at_m10). Linear interpolation in m. Values
# outside [1, 10] extrapolate linearly. Force noise is intentionally absent —
# it is held at 0 because corrupting force corrupts the contact label itself.
NOISE_SCHEDULE: Dict[str, tuple] = {
    "image_noise_sigma_lo":  (5.0,    25.0),
    "image_noise_sigma_hi":  (25.0,   50.0),
    "light_direction_range": (0.20,   2.00),
    "light_color_jitter":    (0.05,   0.50),
    "joint_pos_noise_sigma": (0.001,  0.005),
    "joint_vel_noise_sigma": (0.005,  0.025),
}


def _interp(low: float, high: float, m: float) -> float:
    """Linear interpolation: m=1 → low, m=10 → high. Clipped at 0 below."""
    t = (m - 1.0) / 9.0
    return max(0.0, low + t * (high - low))


def _scale_noise_config(base_cfg: DatasetConfig, m: float) -> DatasetConfig:
    """Return a *new* DatasetConfig with noise parameters set for level ``m``.

    Each parameter is interpolated per its own schedule (see NOISE_SCHEDULE).
    Force noise is fixed at 0 so the contact label / force-direction targets
    remain ground-truth across all levels.  All episodes are flagged noisy.
    """
    cfg = copy.deepcopy(base_cfg)

    img_lo = _interp(*NOISE_SCHEDULE["image_noise_sigma_lo"],  m)
    img_hi = _interp(*NOISE_SCHEDULE["image_noise_sigma_hi"],  m)
    cfg.image_noise_sigma_range = (img_lo, img_hi)

    cfg.light_direction_range = _interp(*NOISE_SCHEDULE["light_direction_range"], m)
    cfg.light_color_jitter    = _interp(*NOISE_SCHEDULE["light_color_jitter"],    m)
    cfg.joint_pos_noise_sigma = _interp(*NOISE_SCHEDULE["joint_pos_noise_sigma"], m)
    cfg.joint_vel_noise_sigma = _interp(*NOISE_SCHEDULE["joint_vel_noise_sigma"], m)

    cfg.force_noise_sigma = 0.0
    cfg.num_clean         = 0

    return cfg


# ── Per-level summary stats ───────────────────────────────────────────────────

def _summarize_level(level_dir: str) -> Dict[str, float]:
    """Compute window-level contact rate and per-episode success rate.

    Reads ``c_windows.npy`` and ``metadata.json`` from each episode directory
    under ``level_dir``. Window-level contact rate matches what the encoder
    and probes see; step-level rate is included as a sanity check.
    """
    episode_dirs = sorted(glob.glob(os.path.join(level_dir, "episode_*")))
    n_eps             = 0
    n_success         = 0
    n_windows_total   = 0
    n_contact_windows = 0
    step_rate_sum     = 0.0

    for ep_dir in episode_dirs:
        meta_path = os.path.join(ep_dir, "metadata.json")
        cw_path   = os.path.join(ep_dir, "c_windows.npy")
        if not (os.path.isfile(meta_path) and os.path.isfile(cw_path)):
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        cw = np.load(cw_path)

        n_eps             += 1
        n_success         += int(meta.get("success", 0))
        n_windows_total   += int(cw.shape[0])
        n_contact_windows += int(cw.sum())
        step_rate_sum     += float(meta.get("contact_ratio", 0.0))

    return {
        "n_eps":               n_eps,
        "n_windows":           n_windows_total,
        "window_contact_rate": n_contact_windows / max(n_windows_total, 1),
        "step_contact_rate":   step_rate_sum     / max(n_eps, 1),
        "success_rate":        n_success         / max(n_eps, 1),
    }


def _print_schedule_for_level(m: float, cfg: DatasetConfig) -> None:
    """Pretty-print the scaled noise values for visibility."""
    lo, hi = cfg.image_noise_sigma_range
    print(
        f"  image σ ∈ [{lo:.2f}, {hi:.2f}]  "
        f"light_dir={cfg.light_direction_range:.3f}  "
        f"light_jit={cfg.light_color_jitter:.3f}\n"
        f"  joint_pos σ={cfg.joint_pos_noise_sigma:.4f}  "
        f"joint_vel σ={cfg.joint_vel_noise_sigma:.4f}  "
        f"force σ={cfg.force_noise_sigma:.2f} (fixed)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate noise-sweep evaluation splits for probe evaluation"
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=list(range(1, 11)),
        metavar="M",
        help="Noise multipliers to sweep (default: 1 2 … 10)",
    )
    parser.add_argument(
        "--episodes_per_level",
        type=int,
        default=100,
        help="Number of noisy episodes per level (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="dataset_noise_sweep",
        help="Root directory for all sweep splits (default: dataset_noise_sweep)",
    )
    parser.add_argument(
        "--gui",  action="store_true",
        help="Launch PyBullet GUI (slow; useful for visual inspection)",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)

    levels = sorted(set(args.levels))
    print(
        f"Noise sweep: {len(levels)} levels × {args.episodes_per_level} episodes"
        f" = {len(levels)*args.episodes_per_level} total episodes"
    )
    print(f"Multipliers : {levels}")
    print(f"Output root : {args.output}\n")

    base_cfg = DatasetConfig()
    ensure_assets(base_cfg)

    level_summaries: List[Dict] = []

    for m in levels:
        level_dir = os.path.join(args.output, f"level_{m:g}")
        print(f"\n{'='*60}")
        print(f"  Level m={m:g}  →  {level_dir}")
        print(f"{'='*60}")

        cfg = _scale_noise_config(base_cfg, m)
        cfg.num_noisy = args.episodes_per_level
        # Inject the output path via the override mechanism in DatasetConfig
        cfg.dataset_dir_override = level_dir  # type: ignore[attr-defined]
        _print_schedule_for_level(m, cfg)

        env = PegInsertionEnv(cfg, gui=args.gui)
        cfg.num_joints = env.num_joints
        controller = SimpleDownwardController(env)

        # Per-episode seeding (independent of m) keeps episode N's initial
        # conditions identical across every level — only the noise SCALE
        # varies, so the underlying physics rollouts are matched.
        run_dataset(cfg, env, controller, per_episode_seed_base=args.seed)

        env.close()

        stats = _summarize_level(level_dir)
        stats["m"] = m
        level_summaries.append(stats)
        print(
            f"  Level m={m:g} done — {stats['n_eps']} eps, "
            f"{stats['n_windows']} windows  |  "
            f"contact (windows)={100*stats['window_contact_rate']:.1f}%  "
            f"contact (steps)={100*stats['step_contact_rate']:.1f}%  "
            f"success={100*stats['success_rate']:.1f}%"
        )

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(
        f"  {'m':>5}  {'eps':>6}  {'windows':>8}  "
        f"{'win_contact%':>13}  {'step_contact%':>14}  {'success%':>9}"
    )
    print(f"  {'-'*70}")
    for s in level_summaries:
        print(
            f"  {s['m']:>5g}  {s['n_eps']:>6d}  {s['n_windows']:>8d}  "
            f"{100*s['window_contact_rate']:>12.1f}%  "
            f"{100*s['step_contact_rate']:>13.1f}%  "
            f"{100*s['success_rate']:>8.1f}%"
        )
    print(f"{'='*72}")
    print(f"All {len(levels)} levels generated under '{args.output}'.")


if __name__ == "__main__":
    main()
