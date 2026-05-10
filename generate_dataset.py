"""Generate the peg-in-hole dataset.

Usage
-----
Full dataset (1500 episodes, 840 clean / 660 noisy):
    python generate_dataset.py

Debug run (100 episodes, 56 clean / 44 noisy, diagnostics):
    python generate_dataset.py --debug

With GUI visualisation:
    python generate_dataset.py --debug --gui
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from config import DatasetConfig
from setup_assets import setup_all
from sim.environment import PegInsertionEnv
from sim.controller import SimpleDownwardController
from sim.noise import (
    randomize_lighting,
    apply_image_noise,
    apply_force_noise,
    apply_joint_noise,
)
from utils.data_io import save_episode
from utils.diagnostics import run_all as run_diagnostics


def ensure_assets(cfg: DatasetConfig) -> None:
    """Generate URDF assets if they don't exist yet."""
    if not os.path.isfile(cfg.robot_urdf):
        setup_all(cfg)


def compute_windows(
    proprio: np.ndarray,
    force: np.ndarray,
    cfg: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-step arrays into windowed data points aligned with RGB frames.

    Window w covers steps [w*k, ..., w*k+k-1]. RGB is saved at the last step
    of each window, so frame_000 corresponds to window 0, frame_001 to window 1,
    etc.

    Returns:
        proprio_windows:  (N, k, proprio_dim) float32
        force_directions: (N, 3)             float32 — fwindow unit vector
        c_windows:        (N,)               int8    — cwindow 0/1
    """
    k = cfg.window_size
    eps = cfg.contact_force_threshold
    T = proprio.shape[0]
    N = T // k
    proprio_dim = proprio.shape[1]

    proprio_windows = np.zeros((N, k, proprio_dim), dtype=np.float32)
    force_directions = np.zeros((N, 3), dtype=np.float32)
    c_windows = np.zeros(N, dtype=np.int8)

    for w in range(N):
        start = w * k
        end = start + k

        proprio_windows[w] = proprio[start:end].astype(np.float32)

        fi = force[start:end, :3].astype(np.float32)   # (k, 3) — Fx, Fy, Fz
        norms = np.linalg.norm(fi, axis=1)              # (k,)
        ci = (norms > eps).astype(np.float32)           # (k,)
        c_windows[w] = np.int8(np.max(ci))

        safe_norms = np.where(norms > eps, norms, 1.0)
        f_hat = np.where(ci[:, None] > 0, fi / safe_norms[:, None], 0.0)

        fsum = np.sum(ci[:, None] * f_hat, axis=0)     # (3,)
        fsum_norm = np.linalg.norm(fsum)
        if fsum_norm > 0:
            force_directions[w] = fsum / fsum_norm

    return proprio_windows, force_directions, c_windows


def run_episode(
    env: PegInsertionEnv,
    controller: SimpleDownwardController,
    cfg: DatasetConfig,
    is_noisy: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, bool]:
    """Execute one episode and return (rgb_frames, proprio, force, success)."""
    rgb_frames: list[np.ndarray] = []
    proprio = np.zeros((cfg.max_steps, cfg.num_joints * 2), dtype=np.float64)
    force = np.zeros((cfg.max_steps, 6), dtype=np.float64)

    success = False
    bore_bottom_z = cfg.table_height + cfg.hole_bottom_thickness
    target_depth = getattr(env, "target_depth", 0.02)
    required_depth_m = float(
        cfg.bore_opening_z
        - (bore_bottom_z + (1.0 - cfg.success_depth_fraction) * target_depth)
    )
    lateral_high_steps = 0
    n_consecutive = getattr(
        cfg, "angular_jam_consecutive_steps", 12
    )
    lat_threshold = getattr(
        cfg, "angular_jam_lateral_force_threshold", 5.0
    )
    use_early_stop = getattr(cfg, "angular_jam_early_stop", True)

    for step in range(cfg.max_steps):
        controller.step()
        env.step_simulation()

        # proprioception
        jpos, jvel = env.get_joint_state()
        if is_noisy:
            jpos, jvel = apply_joint_noise(jpos, jvel, cfg)
        proprio[step] = np.concatenate([jpos, jvel])

        # force / torque
        ft = env.get_contact_force_torque()
        if is_noisy:
            ft = apply_force_noise(ft, cfg)
        force[step] = ft

        # RGB at end of each window (captures state after force events)
        if (step + 1) % cfg.save_rgb_every == 0:
            rgb = env.get_rgb()
            if is_noisy:
                rgb = apply_image_noise(rgb, cfg)
            rgb_frames.append(rgb)

        if not success:
            success = env.check_success()

        # Early stop: peg at intermediate depth + sustained high lateral force → angular jam
        if use_early_stop and not success:
            peg_tip_z = float(env.get_peg_tip_pos()[2])
            end_depth_m = float(cfg.bore_opening_z - peg_tip_z)
            lateral_mag = float(np.linalg.norm(ft[:2]))
            if 0.005 < end_depth_m < required_depth_m - 0.001:
                if lateral_mag > lat_threshold:
                    lateral_high_steps += 1
                    if lateral_high_steps >= n_consecutive:
                        # Truncate to this step and return as failure (angular jam)
                        return (
                            rgb_frames,
                            proprio[: step + 1].copy(),
                            force[: step + 1].copy(),
                            False,
                        )
                else:
                    lateral_high_steps = 0
            else:
                lateral_high_steps = 0

    return rgb_frames, proprio, force, success


def main() -> None:
    parser = argparse.ArgumentParser(description="Peg-in-hole dataset generation")
    parser.add_argument("--debug", action="store_true",
                        help="Quick run: 20 episodes, 50 steps, with diagnostics")
    parser.add_argument("--gui", action="store_true",
                        help="Launch PyBullet GUI (slow, useful for visual check)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = DatasetConfig.debug() if args.debug else DatasetConfig()

    ensure_assets(cfg)

    env = PegInsertionEnv(cfg, gui=args.gui)
    # store num_joints on config so run_episode can reference it
    cfg.num_joints = env.num_joints
    controller = SimpleDownwardController(env)

    os.makedirs(cfg.dataset_dir, exist_ok=True)

    total = cfg.total_episodes
    t0 = time.time()

    for episode_id in tqdm(range(total), desc="episodes", file=sys.stdout):
        is_noisy = episode_id >= cfg.num_clean
        is_hard = np.random.random() < cfg.hard_episode_fraction

        # --- sample randomisation parameters ---
        xy_range = cfg.peg_xy_offset_range_hard if is_hard else cfg.peg_xy_offset_range
        rot_range = cfg.peg_rotation_range_deg_hard if is_hard else cfg.peg_rotation_range_deg

        peg_offset = np.random.uniform(-xy_range, xy_range, 2)
        peg_rotation = np.random.uniform(-rot_range, rot_range, 2)
        hole_offset = np.random.uniform(-cfg.hole_xy_offset_range,
                                         cfg.hole_xy_offset_range, 2)
        target_depth = np.random.uniform(*cfg.target_depth_range)

        # --- reset & optionally randomise lighting ---
        env.reset(peg_offset, peg_rotation, hole_offset, target_depth)
        controller.reset(env.get_ee_pos())
        if is_noisy:
            randomize_lighting(env.physics_client, cfg)

        # --- run episode ---
        rgb_frames, proprio, force_data, success = run_episode(
            env, controller, cfg, is_noisy
        )
        num_steps = force_data.shape[0]
        # Early stop with peg at intermediate depth → synthetic angular jam
        angular_jam = (
            getattr(cfg, "angular_jam_early_stop", False)
            and not success
            and num_steps < cfg.max_steps
        )

        # --- compute windowed data points ---
        proprio_windows, force_directions, c_windows = compute_windows(
            proprio, force_data, cfg
        )
        n_windows = proprio_windows.shape[0]

        # --- compute per-episode stats ---
        force_mag = np.linalg.norm(force_data[:, :3], axis=1)
        contact_ratio = float(np.mean(force_mag > cfg.contact_force_threshold))
        max_force = float(np.max(force_mag))

        # --- compute final insertion depth and tip XY error ---
        final_peg_tip = env.get_peg_tip_pos()
        final_peg_tip_z = float(final_peg_tip[2])
        bore_bottom_z = cfg.table_height + cfg.hole_bottom_thickness
        # Positive = inside the bore; 0 = at bore opening; ~30 mm = at bore floor
        end_depth_m = float(cfg.bore_opening_z - final_peg_tip_z)
        # Minimum depth required for the success check (bore_opening - target_z)
        required_depth_m = float(
            cfg.bore_opening_z
            - (bore_bottom_z + (1.0 - cfg.success_depth_fraction) * target_depth)
        )
        # Positive deficit = peg fell short of success threshold
        depth_deficit_m = float(required_depth_m - end_depth_m)
        # XY distance of peg tip from hole centre at episode end (success needs < 5 mm)
        hole_xy = env.hole_world_pos[:2]
        end_xy_dist_m = float(np.linalg.norm(final_peg_tip[:2] - hole_xy))

        metadata = {
            "episode_id": episode_id,
            "peg_offset": peg_offset.tolist(),
            "peg_rotation": peg_rotation.tolist(),
            "hole_offset": hole_offset.tolist(),
            "target_depth": float(target_depth),
            "end_depth_m": end_depth_m,
            "required_depth_m": required_depth_m,
            "depth_deficit_m": depth_deficit_m,
            "end_xy_dist_m": end_xy_dist_m,
            "is_noisy": is_noisy,
            "is_hard": is_hard,
            "success": int(success),
            "angular_jam": angular_jam,
            "max_contact_force": max_force,
            "contact_ratio": contact_ratio,
            "n_windows": n_windows,
            "window_size": cfg.window_size,
        }

        save_episode(
            cfg.dataset_dir, episode_id, rgb_frames,
            proprio_windows, force_directions, c_windows, metadata,
        )

    elapsed = time.time() - t0
    print(f"\nGenerated {total} episodes in {elapsed:.1f}s "
          f"({elapsed/total:.2f}s / ep)")

    env.close()

    if args.debug:
        run_diagnostics(cfg.dataset_dir, cfg.diagnostics_dir,
                        cfg.contact_force_threshold)


if __name__ == "__main__":
    main()
