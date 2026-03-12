"""One-time asset generation: simplified UR3e URDF and hole fixture mesh.

Run once before dataset generation:
    python setup_assets.py
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# UR3e simplified URDF (correct DH kinematics, cylinder/box link geometry)
# ---------------------------------------------------------------------------

_UR3E_URDF = """\
<?xml version="1.0"?>
<robot name="ur3e">

  <!-- materials -->
  <material name="ur_blue">
    <color rgba="0.0 0.22 0.52 1.0"/>
  </material>
  <material name="ur_silver">
    <color rgba="0.70 0.70 0.70 1.0"/>
  </material>
  <material name="ur_dark">
    <color rgba="0.20 0.20 0.20 1.0"/>
  </material>

  <!-- ==================== base ==================== -->
  <link name="base_link">
    <visual>
      <geometry><cylinder radius="0.038" length="0.050"/></geometry>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <material name="ur_dark"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.038" length="0.050"/></geometry>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.025"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>

  <!-- ==================== shoulder pan ==================== -->
  <joint name="shoulder_pan_joint" type="revolute">
    <parent link="base_link"/>
    <child  link="shoulder_link"/>
    <origin xyz="0 0 0.15185" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.2832" upper="6.2832" effort="56" velocity="3.14"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="shoulder_link">
    <visual>
      <geometry><cylinder radius="0.038" length="0.060"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <material name="ur_blue"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.038" length="0.060"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>

  <!-- ==================== shoulder lift ==================== -->
  <joint name="shoulder_lift_joint" type="revolute">
    <parent link="shoulder_link"/>
    <child  link="upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 1.5707963 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-6.2832" upper="6.2832" effort="56" velocity="3.14"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="upper_arm_link">
    <visual>
      <geometry><box size="0.24355 0.050 0.050"/></geometry>
      <origin xyz="-0.121775 0 0" rpy="0 0 0"/>
      <material name="ur_silver"/>
    </visual>
    <collision>
      <geometry><box size="0.24355 0.050 0.050"/></geometry>
      <origin xyz="-0.121775 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="-0.121775 0 0"/>
      <inertia ixx="5e-3" ixy="0" ixz="0" iyy="5e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>

  <!-- ==================== elbow ==================== -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm_link"/>
    <child  link="forearm_link"/>
    <origin xyz="-0.24355 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-6.2832" upper="6.2832" effort="28" velocity="3.14"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="forearm_link">
    <visual>
      <geometry><box size="0.2132 0.040 0.040"/></geometry>
      <origin xyz="-0.1066 0 0" rpy="0 0 0"/>
      <material name="ur_blue"/>
    </visual>
    <collision>
      <geometry><box size="0.2132 0.040 0.040"/></geometry>
      <origin xyz="-0.1066 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="-0.1066 0 0"/>
      <inertia ixx="3e-3" ixy="0" ixz="0" iyy="3e-3" iyz="0" izz="5e-4"/>
    </inertial>
  </link>

  <!-- ==================== wrist 1 ==================== -->
  <joint name="wrist_1_joint" type="revolute">
    <parent link="forearm_link"/>
    <child  link="wrist_1_link"/>
    <origin xyz="-0.2132 0 0.13105" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-6.2832" upper="6.2832" effort="28" velocity="6.28"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <link name="wrist_1_link">
    <visual>
      <geometry><cylinder radius="0.025" length="0.050"/></geometry>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <material name="ur_silver"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.025" length="0.050"/></geometry>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="3e-4" ixy="0" ixz="0" iyy="3e-4" iyz="0" izz="3e-4"/>
    </inertial>
  </link>

  <!-- ==================== wrist 2 ==================== -->
  <joint name="wrist_2_joint" type="revolute">
    <parent link="wrist_1_link"/>
    <child  link="wrist_2_link"/>
    <origin xyz="0 -0.08535 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.2832" upper="6.2832" effort="28" velocity="6.28"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <link name="wrist_2_link">
    <visual>
      <geometry><cylinder radius="0.025" length="0.050"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <material name="ur_silver"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.025" length="0.050"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="3e-4" ixy="0" ixz="0" iyy="3e-4" iyz="0" izz="3e-4"/>
    </inertial>
  </link>

  <!-- ==================== wrist 3 ==================== -->
  <joint name="wrist_3_joint" type="revolute">
    <parent link="wrist_2_link"/>
    <child  link="wrist_3_link"/>
    <origin xyz="0 0.0921 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-6.2832" upper="6.2832" effort="28" velocity="6.28"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <link name="wrist_3_link">
    <visual>
      <geometry><cylinder radius="0.020" length="0.040"/></geometry>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <material name="ur_dark"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.020" length="0.040"/></geometry>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>

  <!-- ==================== end-effector (fixed) ==================== -->
  <joint name="ee_fixed_joint" type="fixed">
    <parent link="wrist_3_link"/>
    <child  link="ee_link"/>
    <origin xyz="0 0 0" rpy="0 0 1.5707963"/>
  </joint>

  <link name="ee_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

</robot>
"""


def generate_ur3e_urdf(output_path: str) -> None:
    """Write the simplified UR3e URDF to *output_path*."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(_UR3E_URDF)
    print(f"  UR3e URDF -> {output_path}")


# ---------------------------------------------------------------------------
# Hole fixture mesh  (block with cylindrical bore)
# ---------------------------------------------------------------------------

def generate_hole_mesh(
    output_path: str,
    hole_radius: float,
    block_size: float,
    hole_depth: float,
    bottom_thickness: float,
) -> None:
    """Create a block with a cylindrical bore via boolean difference and save
    as OBJ.  Requires *trimesh* and the *manifold3d* backend."""
    import trimesh

    block_height = hole_depth + bottom_thickness

    box = trimesh.creation.box(extents=[block_size, block_size, block_height])

    bore_height = hole_depth + 0.001
    bore_centre_z = (block_height / 2) - (hole_depth / 2)
    cylinder = trimesh.creation.cylinder(
        radius=hole_radius, height=bore_height, sections=64
    )
    cylinder.apply_translation([0, 0, bore_centre_z])

    hole_mesh = trimesh.boolean.difference([box, cylinder], engine="manifold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hole_mesh.export(output_path)
    print(f"  Hole mesh  -> {output_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def setup_all(cfg=None):
    """Generate every asset needed by the simulation."""
    if cfg is None:
        from config import DatasetConfig
        cfg = DatasetConfig()

    print("Generating assets …")
    generate_ur3e_urdf(cfg.robot_urdf)
    generate_hole_mesh(
        cfg.hole_mesh_path,
        hole_radius=cfg.hole_radius,
        block_size=cfg.hole_block_size,
        hole_depth=cfg.hole_depth,
        bottom_thickness=cfg.hole_bottom_thickness,
    )
    print("Done.")


if __name__ == "__main__":
    setup_all()
