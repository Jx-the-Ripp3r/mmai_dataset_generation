"""PyBullet peg-in-hole simulation environment."""

import os
import numpy as np
import pybullet as p
import pybullet_data

from config import DatasetConfig


class PegInsertionEnv:
    """Manages the PyBullet world: UR3e robot, peg, hole fixture, camera."""

    _SEED_JOINTS = [0, 0, 0, 0, 0, 0]

    def __init__(self, config: DatasetConfig, gui: bool = False):
        self.config = config
        self.gui = gui

        # ---- physics client ----
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(config.sim_timestep, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(
            contactBreakingThreshold=0.001,
            numSolverIterations=100,
            enableConeFriction=True,
            physicsClientId=self.physics_client,
        )

        # ---- ground plane ----
        self.plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self.physics_client
        )

        # ---- UR3e robot ----
        self.robot_id = p.loadURDF(
            config.robot_urdf,
            basePosition=list(config.robot_base_position),
            useFixedBase=True,
            physicsClientId=self.physics_client,
        )

        self.joint_indices: list[int] = []
        num_joints = p.getNumJoints(self.robot_id,
                                     physicsClientId=self.physics_client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i,
                                  physicsClientId=self.physics_client)
            if info[2] != p.JOINT_FIXED:
                self.joint_indices.append(i)

        self.num_joints = len(self.joint_indices)
        # ee_link is child of the last joint (the fixed ee_fixed_joint)
        self.ee_link_index = num_joints - 1

        # Enable F/T sensor on wrist-3 joint (last revolute)
        p.enableJointForceTorqueSensor(
            self.robot_id, self.joint_indices[-1], True,
            physicsClientId=self.physics_client,
        )

        # ---- camera (constant across episodes) ----
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(config.cam_target),
            distance=config.cam_distance,
            yaw=config.cam_yaw,
            pitch=config.cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=config.cam_fov,
            aspect=config.cam_width / config.cam_height,
            nearVal=config.cam_near,
            farVal=config.cam_far,
        )

        # ---- pre-create reusable collision / visual shapes ----
        self.peg_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=config.peg_radius,
            height=config.peg_length,
            physicsClientId=self.physics_client,
        )
        self.peg_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=config.peg_radius,
            length=config.peg_length,
            rgbaColor=[0.82, 0.82, 0.82, 1.0],
            physicsClientId=self.physics_client,
        )

        # ---- mutable state (populated by reset()) ----
        self.peg_id: int | None = None
        self.hole_body_ids: list[int] = []
        self.peg_constraint: int | None = None
        self.hole_world_pos = np.zeros(3)
        self.target_depth: float = 0.0

    # ------------------------------------------------------------------
    # Hole fixture from box primitives (avoids concave-mesh issues)
    # ------------------------------------------------------------------
    def _create_box_hole(self, cx: float, cy: float, cfg) -> list[int]:
        """Build the hole fixture from five boxes: four walls + bottom plate.

        The bore has a square cross-section of side ``2 * hole_radius``.
        """
        hr = cfg.hole_radius
        bs = cfg.hole_block_size / 2.0
        bh = cfg.hole_block_height
        bt = cfg.hole_bottom_thickness
        base_z = cfg.table_height
        wall_h = bh            # full block height
        color = [0.30, 0.30, 0.30, 1.0]

        bodies: list[int] = []

        def _add_box(hx, hy, hz, px, py, pz):
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                physicsClientId=self.physics_client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                rgbaColor=color,
                physicsClientId=self.physics_client,
            )
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[px, py, pz],
                physicsClientId=self.physics_client,
            )
            p.changeDynamics(
                bid, -1,
                lateralFriction=cfg.lateral_friction,
                restitution=cfg.restitution,
                contactStiffness=1e6,
                contactDamping=1e3,
                physicsClientId=self.physics_client,
            )
            bodies.append(bid)

        # wall thickness = half_block - hole_radius
        wt = bs - hr  # half-thickness of each wall

        # +X wall
        _add_box(wt / 2, bs, wall_h / 2,
                 cx + hr + wt / 2, cy, base_z + wall_h / 2)
        # -X wall
        _add_box(wt / 2, bs, wall_h / 2,
                 cx - hr - wt / 2, cy, base_z + wall_h / 2)
        # +Y wall (spans only the opening width to avoid overlap at corners)
        _add_box(hr, wt / 2, wall_h / 2,
                 cx, cy + hr + wt / 2, base_z + wall_h / 2)
        # -Y wall
        _add_box(hr, wt / 2, wall_h / 2,
                 cx, cy - hr - wt / 2, base_z + wall_h / 2)
        # bottom plate (full block footprint)
        _add_box(bs, bs, bt / 2,
                 cx, cy, base_z + bt / 2)

        return bodies

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        peg_xy_offset: np.ndarray,
        peg_rotation_deg: np.ndarray,
        hole_xy_offset: np.ndarray,
        target_depth: float,
    ) -> None:
        cfg = self.config
        self.target_depth = target_depth

        # --- remove old dynamic bodies ---
        if self.peg_constraint is not None:
            p.removeConstraint(self.peg_constraint,
                               physicsClientId=self.physics_client)
            self.peg_constraint = None
        if self.peg_id is not None:
            p.removeBody(self.peg_id,
                         physicsClientId=self.physics_client)
            self.peg_id = None
        for bid in self.hole_body_ids:
            p.removeBody(bid, physicsClientId=self.physics_client)
        self.hole_body_ids.clear()

        # --- spawn hole fixture from box primitives ---
        hole_x = cfg.hole_base_xy[0] + hole_xy_offset[0]
        hole_y = cfg.hole_base_xy[1] + hole_xy_offset[1]
        self.hole_world_pos = np.array([hole_x, hole_y, cfg.table_height])

        self.hole_body_ids = self._create_box_hole(
            hole_x, hole_y, cfg
        )

        # --- reset robot joints to seed, then solve IK ---
        for i, ji in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, ji, self._SEED_JOINTS[i],
                              physicsClientId=self.physics_client)

        # desired EE position: above hole centre, offset by peg_xy_offset
        ee_x = hole_x + peg_xy_offset[0]
        ee_y = hole_y + peg_xy_offset[1]
        ee_z = cfg.bore_opening_z + cfg.peg_start_height_above_hole + cfg.peg_length

        down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        ik = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index,
            [ee_x, ee_y, ee_z], down_orn,
            maxNumIterations=500,
            residualThreshold=1e-6,
            physicsClientId=self.physics_client,
        )
        for i, ji in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, ji, ik[i],
                              physicsClientId=self.physics_client)
            p.setJointMotorControl2(
                self.robot_id, ji, p.POSITION_CONTROL,
                targetPosition=ik[i],
                force=cfg.max_joint_force,
                positionGain=1.0,
                velocityGain=1.0,
                physicsClientId=self.physics_client,
            )

        # --- create & attach peg ---
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index,
                                  physicsClientId=self.physics_client)
        ee_pos = np.array(ee_state[0])

        rot_x = np.deg2rad(peg_rotation_deg[0])
        rot_y = np.deg2rad(peg_rotation_deg[1])
        child_orn = p.getQuaternionFromEuler([rot_x, rot_y, 0])

        # Place peg centre so that the constraint attachment point (the peg's
        # local –z end) already coincides with the EE — avoids a large
        # correction impulse during the settle phase.
        peg_centre = [ee_pos[0], ee_pos[1], ee_pos[2] - cfg.peg_length / 2]

        self.peg_id = p.createMultiBody(
            baseMass=cfg.peg_mass,
            baseCollisionShapeIndex=self.peg_col,
            baseVisualShapeIndex=self.peg_vis,
            basePosition=peg_centre,
            baseOrientation=down_orn,
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            self.peg_id, -1,
            lateralFriction=cfg.lateral_friction,
            restitution=cfg.restitution,
            contactStiffness=1e6,
            contactDamping=1e3,
            physicsClientId=self.physics_client,
        )

        # child_frame_position places the peg's top at the EE origin
        self.peg_constraint = p.createConstraint(
            self.robot_id, self.ee_link_index,
            self.peg_id, -1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, -cfg.peg_length / 2],
            childFrameOrientation=child_orn,
            physicsClientId=self.physics_client,
        )
        p.changeConstraint(self.peg_constraint, maxForce=10000,
                           physicsClientId=self.physics_client)

        # disable peg ↔ robot collision
        for li in range(-1, p.getNumJoints(self.robot_id,
                                            physicsClientId=self.physics_client)):
            p.setCollisionFilterPair(
                self.robot_id, self.peg_id, li, -1, enableCollision=0,
                physicsClientId=self.physics_client,
            )

        # settle with kinematic hold to prevent gravity / constraint drift
        for _ in range(50):
            for i_j, ji in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, ji, ik[i_j],
                                  physicsClientId=self.physics_client)
            p.stepSimulation(physicsClientId=self.physics_client)

        # hand off to position control for the episode
        for i_j, ji in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id, ji, p.POSITION_CONTROL,
                targetPosition=ik[i_j],
                force=cfg.max_joint_force,
                positionGain=1.0,
                velocityGain=1.0,
                physicsClientId=self.physics_client,
            )

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step_simulation(self) -> None:
        for _ in range(self.config.substeps_per_step):
            p.stepSimulation(physicsClientId=self.physics_client)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def get_rgb(self) -> np.ndarray:
        """128×128×3 uint8 RGB image."""
        _, _, rgba, _, _ = p.getCameraImage(
            self.config.cam_width, self.config.cam_height,
            self.view_matrix, self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client,
        )
        img = np.array(rgba, dtype=np.uint8).reshape(
            self.config.cam_height, self.config.cam_width, 4
        )
        return img[:, :, :3]

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions [6], velocities [6])."""
        states = [
            p.getJointState(self.robot_id, ji,
                            physicsClientId=self.physics_client)
            for ji in self.joint_indices
        ]
        pos = np.array([s[0] for s in states], dtype=np.float64)
        vel = np.array([s[1] for s in states], dtype=np.float64)
        return pos, vel

    def get_ee_pos(self) -> np.ndarray:
        state = p.getLinkState(self.robot_id, self.ee_link_index,
                               physicsClientId=self.physics_client)
        return np.array(state[0], dtype=np.float64)

    def get_contact_force_torque(self) -> np.ndarray:
        """Compute net contact force/torque on the peg from all contacts.

        Returns [Fx, Fy, Fz, Tx, Ty, Tz] — pure contact contribution
        (gravity-free).  Torques are about the EE position.
        """
        if self.peg_id is None:
            return np.zeros(6, dtype=np.float64)

        ee_pos = self.get_ee_pos()
        total_f = np.zeros(3, dtype=np.float64)
        total_t = np.zeros(3, dtype=np.float64)

        contacts = p.getContactPoints(
            bodyA=self.peg_id,
            physicsClientId=self.physics_client,
        )
        for c in contacts:
            other_body = c[2]
            if other_body == self.robot_id:
                continue
            pos_on_peg = np.array(c[5], dtype=np.float64)
            normal = np.array(c[7], dtype=np.float64)
            normal_force = c[9]
            fric1_dir = np.array(c[11], dtype=np.float64)
            fric1_force = c[10]
            fric2_dir = np.array(c[13], dtype=np.float64)
            fric2_force = c[12]

            f = normal * normal_force
            f += fric1_dir * fric1_force
            f += fric2_dir * fric2_force
            total_f += f
            total_t += np.cross(pos_on_peg - ee_pos, f)

        return np.concatenate([total_f, total_t])

    # ------------------------------------------------------------------
    # Success criterion
    # ------------------------------------------------------------------
    def get_peg_tip_pos(self) -> np.ndarray:
        """Return the peg tip (bottom) world position from actual peg body."""
        if self.peg_id is None:
            return np.zeros(3, dtype=np.float64)
        pos, orn = p.getBasePositionAndOrientation(
            self.peg_id, physicsClientId=self.physics_client
        )
        pos = np.array(pos, dtype=np.float64)
        # Peg body origin is at cylinder centre. The peg is rotated π around
        # X, so local +Z maps to world -Z. Local [0,0,+L/2] gives the world
        # bottom (tip).
        local_tip = np.array([0, 0, self.config.peg_length / 2])
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos + rot @ local_tip

    def check_success(self) -> bool:
        cfg = self.config
        peg_tip = self.get_peg_tip_pos()

        bore_bottom_z = self.hole_world_pos[2] + cfg.hole_bottom_thickness
        target_z = bore_bottom_z + (1.0 - cfg.success_depth_fraction) * self.target_depth

        peg_xy = peg_tip[:2]
        hole_xy = self.hole_world_pos[:2]
        xy_dist = float(np.linalg.norm(peg_xy - hole_xy))

        return peg_tip[2] < target_z and xy_dist < cfg.success_xy_tolerance

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        p.disconnect(physicsClientId=self.physics_client)
