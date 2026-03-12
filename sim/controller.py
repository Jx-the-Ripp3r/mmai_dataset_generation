"""Simple downward controller — no lateral correction."""

import numpy as np
import pybullet as p


class SimpleDownwardController:
    """Move the end-effector straight down at a constant rate.

    Uses high-gain position control that respects contact physics while
    tracking the target trajectory closely in free space.
    """

    def __init__(self, env):
        self.env = env
        self.cfg = env.config
        self._down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        self._target: np.ndarray | None = None
        self._z_floor: float = 0.0

    def reset(self, start_pos: np.ndarray) -> None:
        self._target = start_pos.copy()
        cfg = self.cfg
        # Limit target so peg tip doesn't go more than 5 mm past bore bottom
        bore_bottom_z = cfg.table_height + cfg.hole_bottom_thickness
        self._z_floor = bore_bottom_z + cfg.peg_length - 0.005

    def step(self) -> None:
        if self._target is None:
            self.reset(self.env.get_ee_pos())

        self._target[2] = max(
            self._target[2] - self.cfg.descent_rate, self._z_floor
        )

        ik = p.calculateInverseKinematics(
            self.env.robot_id,
            self.env.ee_link_index,
            self._target.tolist(),
            self._down_orn,
            maxNumIterations=500,
            residualThreshold=1e-6,
            physicsClientId=self.env.physics_client,
        )

        for idx, ji in enumerate(self.env.joint_indices):
            p.setJointMotorControl2(
                self.env.robot_id,
                ji,
                p.POSITION_CONTROL,
                targetPosition=ik[idx],
                force=self.cfg.max_joint_force,
                positionGain=1.0,
                velocityGain=1.0,
                maxVelocity=2.0,
                physicsClientId=self.env.physics_client,
            )
