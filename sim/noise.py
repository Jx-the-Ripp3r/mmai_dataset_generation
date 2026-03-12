"""Noise injection utilities for noisy episodes."""

import numpy as np
import pybullet as p


def randomize_lighting(physics_client: int, cfg) -> None:
    """Randomise the debug-visualiser light direction (called once per episode)."""
    direction = np.random.uniform(-cfg.light_direction_range,
                                  cfg.light_direction_range, size=3)
    direction[2] = abs(direction[2]) + 0.3  # keep light roughly above
    p.configureDebugVisualizer(
        p.COV_ENABLE_SHADOWS, 1,
        lightPosition=direction.tolist(),
        physicsClientId=physics_client,
    )


def apply_image_noise(rgb: np.ndarray, cfg) -> np.ndarray:
    """Additive Gaussian noise + brightness shift on a uint8 RGB image."""
    sigma = np.random.uniform(*cfg.image_noise_sigma_range)
    noise = np.random.normal(0, sigma, rgb.shape)
    brightness = np.random.uniform(-15, 15)
    noisy = rgb.astype(np.float32) + noise + brightness
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_force_noise(ft: np.ndarray, cfg) -> np.ndarray:
    """Additive Gaussian noise on a 6-D force/torque vector."""
    return ft + np.random.normal(0, cfg.force_noise_sigma, ft.shape)


def apply_joint_noise(
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg,
) -> tuple:
    """Additive Gaussian noise on joint positions and velocities."""
    noisy_pos = positions + np.random.normal(0, cfg.joint_pos_noise_sigma,
                                             positions.shape)
    noisy_vel = velocities + np.random.normal(0, cfg.joint_vel_noise_sigma,
                                              velocities.shape)
    return noisy_pos, noisy_vel
