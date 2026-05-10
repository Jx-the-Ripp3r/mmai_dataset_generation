"""Configuration for peg-in-hole dataset generation."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig:
    # ---- Geometry (metres) ----
    peg_radius: float = 0.005
    peg_length: float = 0.080
    peg_mass: float = 0.05
    # Tighter bore + higher friction give true angular jams (peg wedges, whole
    # assembly stops). 8 mm keeps rim + success mix; try 7 mm or μ≈0.55 if you
    # want more mid-insertion jams (may reduce success rate).
    hole_radius: float = 0.008
    hole_block_size: float = 0.060
    hole_depth: float = 0.030
    hole_bottom_thickness: float = 0.005

    # ---- Robot / workspace ----
    robot_base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hole_base_xy: Tuple[float, float] = (0.25, 0.0)
    table_height: float = 0.0
    peg_start_height_above_hole: float = 0.020

    # ---- Randomisation ranges ----
    peg_xy_offset_range: float = 0.003
    peg_xy_offset_range_hard: float = 0.010
    hole_xy_offset_range: float = 0.002
    peg_rotation_range_deg: float = 2.0
    peg_rotation_range_deg_hard: float = 5.0
    hard_episode_fraction: float = 0.25
    target_depth_range: Tuple[float, float] = (0.015, 0.025)

    # ---- Episode ----
    max_steps: int = 80
    substeps_per_step: int = 40
    save_rgb_every: int = 5
    window_size: int = 5  # must be a multiple of save_rgb_every
    sim_timestep: float = 1.0 / 240.0

    # ---- Controller ----
    descent_rate: float = 0.001
    max_joint_force: float = 56.0

    # ---- Camera (fixed across all episodes) ----
    cam_distance: float = 0.35
    cam_yaw: float = 45.0
    cam_pitch: float = -25.0
    cam_target: Tuple[float, float, float] = (0.25, 0.0, 0.06)
    cam_width: int = 128
    cam_height: int = 128
    cam_fov: float = 50.0
    cam_near: float = 0.01
    cam_far: float = 2.0

    # ---- Splits ----
    # Train/eval split: 80% train (70% clean / 30% noisy), 20% eval (100% noisy).
    # That works out to 44% noisy episodes overall (660/1500 or 44/100 for debug).
    num_clean: int = 840
    num_noisy: int = 660

    # ---- Success thresholds ----
    success_depth_fraction: float = 0.90
    success_xy_tolerance: float = 0.005

    # ---- Contact detection ----
    contact_force_threshold: float = 0.5

    # ---- Angular-jam early stop (synthetic mid-insertion jams) ----
    # When peg is at intermediate depth and lateral force is high for N steps,
    # stop the episode early so we get a distinct force profile (stall).
    angular_jam_early_stop: bool = True
    angular_jam_lateral_force_threshold: float = 5.0  # N (Fx,Fy magnitude)
    angular_jam_consecutive_steps: int = 12

    # ---- Noise (noisy episodes only) ----
    image_noise_sigma_range: Tuple[float, float] = (5.0, 25.0)
    force_noise_sigma: float = 0.2
    joint_pos_noise_sigma: float = 0.001
    joint_vel_noise_sigma: float = 0.005
    light_direction_range: float = 1.0
    light_color_jitter: float = 0.20

    # ---- Friction / dynamics ----
    # Higher friction allows wedge jams when peg contacts bore wall; ~0.45–0.55
    # balances jams vs successes and rim-contact failures.
    lateral_friction: float = 0.50
    restitution: float = 0.1

    # ---- Derived paths ----
    @property
    def project_root(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def asset_dir(self) -> str:
        return os.path.join(self.project_root, "assets")

    @property
    def dataset_dir(self) -> str:
        override = getattr(self, "dataset_dir_override", None)
        if override is not None:
            return override
        return os.path.join(self.project_root, "dataset")

    @property
    def diagnostics_dir(self) -> str:
        return os.path.join(self.project_root, "diagnostics")

    @property
    def robot_urdf(self) -> str:
        return os.path.join(self.asset_dir, "ur3e", "ur3e.urdf")

    @property
    def hole_mesh_path(self) -> str:
        return os.path.join(self.asset_dir, "hole.obj")

    # ---- Computed geometry helpers ----
    @property
    def hole_block_height(self) -> float:
        return self.hole_depth + self.hole_bottom_thickness

    @property
    def bore_opening_z(self) -> float:
        return self.table_height + self.hole_block_height

    @property
    def total_episodes(self) -> int:
        return self.num_clean + self.num_noisy

    @property
    def num_rgb_frames(self) -> int:
        return self.max_steps // self.save_rgb_every

    @classmethod
    def debug(cls) -> "DatasetConfig":
        return cls(num_clean=56, num_noisy=44, max_steps=80)
