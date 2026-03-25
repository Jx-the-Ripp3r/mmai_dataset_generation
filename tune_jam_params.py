"""Search for hole_radius + lateral_friction that yield rim, success, and angular jam."""
import json
import os
import glob
import tempfile
import shutil
import numpy as np
from tqdm import tqdm

from config import DatasetConfig
from setup_assets import setup_all
from sim.environment import PegInsertionEnv
from sim.controller import SimpleDownwardController
from sim.noise import randomize_lighting, apply_image_noise, apply_force_noise, apply_joint_noise
from utils.data_io import save_episode


def run_episode(env, controller, cfg, is_noisy):
    rgb_frames = []
    proprio = np.zeros((cfg.max_steps, cfg.num_joints * 2), dtype=np.float64)
    force = np.zeros((cfg.max_steps, 6), dtype=np.float64)
    success = False
    for step in range(cfg.max_steps):
        controller.step()
        env.step_simulation()
        jpos, jvel = env.get_joint_state()
        if is_noisy:
            jpos, jvel = apply_joint_noise(jpos, jvel, cfg)
        proprio[step] = np.concatenate([jpos, jvel])
        ft = env.get_contact_force_torque()
        if is_noisy:
            ft = apply_force_noise(ft, cfg)
        force[step] = ft
        if step % cfg.save_rgb_every == 0:
            rgb = env.get_rgb()
            if is_noisy:
                rgb = apply_image_noise(rgb, cfg)
            rgb_frames.append(rgb)
        if not success:
            success = env.check_success()
    return rgb_frames, proprio, force, success


def classify_episodes(dataset_dir):
    """Return (n_success, n_rim, n_angular_jam, n_xy_only)."""
    n_success = n_rim = n_angular_jam = n_xy_only = 0
    for path in sorted(glob.glob(os.path.join(dataset_dir, "episode_*", "metadata.json"))):
        with open(path) as f:
            m = json.load(f)
        end_mm = m.get("end_depth_m", 0) * 1000
        req_mm = m.get("required_depth_m", 30) * 1000
        if m.get("success"):
            n_success += 1
            continue
        if end_mm < 5:
            n_rim += 1
        elif end_mm < req_mm - 1:
            n_angular_jam += 1
        else:
            n_xy_only += 1
    return n_success, n_rim, n_angular_jam, n_xy_only


def run_trial(cfg, dataset_dir, seed=42):
    np.random.seed(seed)
    if not os.path.isfile(cfg.robot_urdf):
        setup_all(cfg)
    env = PegInsertionEnv(cfg, gui=False)
    cfg.num_joints = env.num_joints
    controller = SimpleDownwardController(env)
    os.makedirs(dataset_dir, exist_ok=True)
    n = cfg.total_episodes
    for episode_id in tqdm(range(n), desc="episodes", leave=False):
        is_noisy = episode_id >= cfg.num_clean
        is_hard = np.random.random() < cfg.hard_episode_fraction
        xy_range = cfg.peg_xy_offset_range_hard if is_hard else cfg.peg_xy_offset_range
        rot_range = cfg.peg_rotation_range_deg_hard if is_hard else cfg.peg_rotation_range_deg
        peg_offset = np.random.uniform(-xy_range, xy_range, 2)
        peg_rotation = np.random.uniform(-rot_range, rot_range, 2)
        hole_offset = np.random.uniform(-cfg.hole_xy_offset_range, cfg.hole_xy_offset_range, 2)
        target_depth = np.random.uniform(*cfg.target_depth_range)
        env.reset(peg_offset, peg_rotation, hole_offset, target_depth)
        controller.reset(env.get_ee_pos())
        if is_noisy:
            randomize_lighting(env.physics_client, cfg)
        rgb_frames, proprio, force_data, success = run_episode(env, controller, cfg, is_noisy)
        force_mag = np.linalg.norm(force_data[:, :3], axis=1)
        contact_ratio = float(np.mean(force_mag > cfg.contact_force_threshold))
        max_force = float(np.max(force_mag))
        final_peg_tip = env.get_peg_tip_pos()
        final_peg_tip_z = float(final_peg_tip[2])
        bore_bottom_z = cfg.table_height + cfg.hole_bottom_thickness
        end_depth_m = float(cfg.bore_opening_z - final_peg_tip_z)
        required_depth_m = float(
            cfg.bore_opening_z
            - (bore_bottom_z + (1.0 - cfg.success_depth_fraction) * target_depth)
        )
        depth_deficit_m = float(required_depth_m - end_depth_m)
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
            "max_contact_force": max_force,
            "contact_ratio": contact_ratio,
        }
        save_episode(dataset_dir, episode_id, rgb_frames, proprio, force_data, metadata)
    env.close()
    return classify_episodes(dataset_dir)


def main():
    # (hole_radius, lateral_friction, max_joint_force or None for default 56)
    # (hole_radius, lateral_friction, max_joint_force, optional substeps).
    # Used to search for physics params; angular jams are now mainly from
    # angular_jam_early_stop in config (see generate_dataset + config).
    candidates = [
        (0.008, 0.50, None),
        (0.0075, 0.55, None),
        (0.007, 0.60, 25),
    ]
    print("Searching for hole_radius (m) + lateral_friction that give rim + success + angular jam...")
    print("Candidates:", candidates)
    best = None
    best_params = None
    with tempfile.TemporaryDirectory(prefix="peg_tune_") as root:
        for item in candidates:
            hr, lf = item[0], item[1]
            mjf = item[2] if len(item) > 2 else None
            substeps = item[3] if len(item) > 3 else None
            cfg = DatasetConfig.debug()
            cfg.hole_radius = hr
            cfg.lateral_friction = lf
            if mjf is not None:
                cfg.max_joint_force = mjf
            if substeps is not None:
                cfg.substeps_per_step = substeps
            path_suffix = f"hr{hr}_lf{lf}" + (f"_mjf{mjf}" if mjf else "") + (f"_ss{substeps}" if substeps else "")
            cfg.dataset_dir_override = os.path.join(root, path_suffix)
            n_succ, n_rim, n_jam, n_xy = run_trial(cfg, cfg.dataset_dir)
            has_all = n_rim >= 1 and n_succ >= 1 and n_jam >= 1
            label = f"hr={hr} lf={lf}" + (f" mjf={mjf}" if mjf else "") + (f" substeps={substeps}" if substeps else "")
            print(f"  {label}  -> success={n_succ} rim={n_rim} angular_jam={n_jam} xy_only={n_xy}  has_all_3={has_all}")
            if has_all and (best is None or n_jam > best[2]):
                best = (n_succ, n_rim, n_jam, n_xy)
                best_params = (hr, lf, mjf)
    if best is not None:
        hr, lf, mjf = best_params
        print(f"\nBest params: hole_radius={hr}, lateral_friction={lf}" + (f", max_joint_force={mjf}" if mjf else ""))
        print(f"  success={best[0]} rim={best[1]} angular_jam={best[2]} xy_only={best[3]}")
        return hr, lf, mjf
    print("\nNo candidate had all three (rim + success + angular jam). Try more extreme params.")
    return None, None, None


if __name__ == "__main__":
    main()
