"""Generate a series of noisy evaluation splits for probe degradation curves.

Each split ("level") runs the full PyBullet simulation with noise parameters
scaled by a configurable multiplier ``m``.  The baseline defaults in
``config.py`` correspond to ``m = 1.0``.

Layout written to disk::

    <output>/
        level_1.0/episode_0000/ ...
        level_2.0/episode_0000/ ...
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
import os
import sys

import numpy as np

from config import DatasetConfig
from generate_dataset import ensure_assets, run_dataset
from sim.controller import SimpleDownwardController
from sim.environment import PegInsertionEnv


def _scale_noise_config(base_cfg: DatasetConfig, multiplier: float) -> DatasetConfig:
    """Return a *new* DatasetConfig with all noise parameters scaled by ``multiplier``.

    Only noisy parameters are touched; geometry, robot, and camera settings are
    identical to ``base_cfg``.  ``num_clean`` is set to 0 so that every episode
    receives noise injection.
    """
    cfg = copy.deepcopy(base_cfg)

    # Scale image noise range at both ends
    lo, hi = base_cfg.image_noise_sigma_range
    cfg.image_noise_sigma_range = (lo * multiplier, hi * multiplier)

    cfg.force_noise_sigma      = base_cfg.force_noise_sigma      * multiplier
    cfg.joint_pos_noise_sigma  = base_cfg.joint_pos_noise_sigma  * multiplier
    cfg.joint_vel_noise_sigma  = base_cfg.joint_vel_noise_sigma  * multiplier
    cfg.light_direction_range  = base_cfg.light_direction_range  * multiplier
    cfg.light_color_jitter     = base_cfg.light_color_jitter     * multiplier

    # All episodes are noisy (no clean episodes in evaluation splits)
    cfg.num_clean = 0

    return cfg


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

    for m in levels:
        level_dir = os.path.join(args.output, f"level_{m:g}")
        print(f"\n{'='*60}")
        print(f"  Level m={m:g}  →  {level_dir}")
        print(f"{'='*60}")

        cfg = _scale_noise_config(base_cfg, m)
        cfg.num_noisy = args.episodes_per_level
        # Inject the output path via the override mechanism in DatasetConfig
        cfg.dataset_dir_override = level_dir  # type: ignore[attr-defined]

        # Seed per-level for reproducibility while keeping levels independent
        level_seed = args.seed + int(m * 1000)
        np.random.seed(level_seed)

        env = PegInsertionEnv(cfg, gui=args.gui)
        cfg.num_joints = env.num_joints
        controller = SimpleDownwardController(env)

        run_dataset(cfg, env, controller)

        env.close()
        print(f"  Level m={m:g} done — {args.episodes_per_level} episodes in {level_dir}")

    print(f"\nAll {len(levels)} levels generated under '{args.output}'.")


if __name__ == "__main__":
    main()
