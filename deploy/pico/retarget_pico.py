"""Convert PICO VR pose data to G1 robot qpos_full (36-D).

Since PICO only tracks head + 2 controllers, we cannot drive the full body:
  - Root (torso): orientation from head yaw; XY fixed at 0; Z fixed at 0.78 m.
  - Waist: yaw follows head yaw; roll/pitch stay near zero.
  - Arms: 7-DOF per arm solved via damped-least-squares Jacobian IK in MuJoCo.
  - Legs: held at default standing pose; driven by the walk policy separately.

Coordinate frames
-----------------
PICO world (OpenXR stage space):  +Y up,  -Z forward,  +X right
MuJoCo/Robot world:               +Z up,  +X forward,  +Y left

The rotation matrix from PICO→Robot is:
    R[0,:] = [0, 0, -1]   robot +X = -PICO Z  (forward)
    R[1,:] = [-1, 0,  0]  robot +Y = -PICO X  (left)
    R[2,:] = [0, 1,  0]   robot +Z =  PICO Y  (up)
"""

from __future__ import annotations

import numpy as np
import mujoco

from tracking import constants as consts
from deploy.pico.client import PicoFrame

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Default standing qpos_full (36-D: root_xyz + root_quat + 29 joints)
_DEFAULT_QPOS = consts.DEFAULT_QPOS.copy()          # shape (36,)

# Left arm joint indices inside qpos (7 = root offset: 3 xyz + 4 quat)
_L_ARM_QPOS = np.arange(7 + 15, 7 + 22)            # [22:29]
_R_ARM_QPOS = np.arange(7 + 22, 7 + 29)            # [29:36]

# Left arm joint DOF indices in qvel  (6 = root DOFs: 3 lin + 3 ang)
_L_ARM_QVEL = np.arange(6 + 15, 6 + 22)            # [21:28]
_R_ARM_QVEL = np.arange(6 + 22, 6 + 29)            # [28:35]

# Joint limits for arms from RESTRICTED_JOINT_RANGE (indices 15-28)
_L_ARM_LO = np.array([lo for lo, _ in consts.RESTRICTED_JOINT_RANGE[15:22]], dtype=np.float32)
_L_ARM_HI = np.array([hi for _, hi in consts.RESTRICTED_JOINT_RANGE[15:22]], dtype=np.float32)
_R_ARM_LO = np.array([lo for lo, _ in consts.RESTRICTED_JOINT_RANGE[22:29]], dtype=np.float32)
_R_ARM_HI = np.array([hi for _, hi in consts.RESTRICTED_JOINT_RANGE[22:29]], dtype=np.float32)

# PICO → MuJoCo world rotation (see module docstring)
R_PICO2ROBOT = np.array([
    [ 0.,  0., -1.],
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
], dtype=np.float32)

# Human head is ~0.25 m above the shoulders; chest ~0.15 m below shoulders.
# We estimate the "chest" reference point from which arms extend.
_HEAD_TO_CHEST_PICO = np.array([0.0, -0.40, 0.0], dtype=np.float32)   # -Y = down in PICO

# Scale: human arm reach from chest (~0.65 m) → robot arm reach from shoulder (~0.50 m)
_ARM_SCALE = 0.50 / 0.65

# Shoulder offset from chest in robot frame (approx):
#   left shoulder is ~+0.18 m in Y (left), right shoulder ~-0.18 m
_CHEST_TO_L_SHOULDER_ROBOT = np.array([0.0,  0.18, 0.10], dtype=np.float32)
_CHEST_TO_R_SHOULDER_ROBOT = np.array([0.0, -0.18, 0.10], dtype=np.float32)

# IK hyper-parameters
_IK_MAX_ITER  = 30
_IK_TOL       = 3e-3    # metres
_IK_ALPHA     = 0.4     # step size
_IK_DAMPING   = 5e-2    # DLS damping
_IK_NULL_GAIN = 0.1     # pull toward default in null-space


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _quat_to_yaw_pico(q_wxyz: np.ndarray) -> float:
    """Extract yaw (rotation around PICO +Y axis) from a wxyz quaternion."""
    w, x, y, z = q_wxyz
    # Yaw around Y: atan2(2(wx + yz) / ...) — standard Y-up yaw formula
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _yaw_to_quat_robot(yaw: float) -> np.ndarray:
    """Build wxyz quaternion for rotation around robot +Z axis."""
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _rot_error_so3(R_tgt: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Axis-angle rotation error (3,) for SO(3) targets."""
    R_err = R_tgt @ R_cur.T
    tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (theta / (2.0 * np.sin(theta))) * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=np.float32)


def _arm_ik(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    target_pos: np.ndarray,
    wrist_body_id: int,
    qpos_ids: np.ndarray,
    qvel_ids: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    q_default: np.ndarray,
) -> None:
    """Solve arm IK in-place on mj_data.qpos using damped Jacobian + null-space."""
    nv  = mj_model.nv
    n   = len(qvel_ids)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for _ in range(_IK_MAX_ITER):
        mujoco.mj_forward(mj_model, mj_data)
        pos_err = target_pos - mj_data.xpos[wrist_body_id]
        if np.linalg.norm(pos_err) < _IK_TOL:
            break

        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, wrist_body_id)
        J = jacp[:, qvel_ids]                          # (3, n)

        JJT = J @ J.T
        d_mat = (_IK_DAMPING ** 2) * np.eye(3)
        dq_task = J.T @ np.linalg.solve(JJT + d_mat, pos_err)

        # Null-space: gently pull joints toward default
        J_pinv = J.T @ np.linalg.inv(JJT + d_mat)    # (n, 3) pseudo-inverse
        null = np.eye(n) - J_pinv @ J                  # (n, n)
        q_cur = mj_data.qpos[qpos_ids]
        dq_null = null @ (_IK_NULL_GAIN * (q_default - q_cur))

        dq = _IK_ALPHA * (dq_task + dq_null)
        mj_data.qpos[qpos_ids] = np.clip(q_cur + dq, q_lo, q_hi)


# -----------------------------------------------------------------------
# PicoRetargeter
# -----------------------------------------------------------------------

class PicoRetargeter:
    """Convert a PicoFrame into a G1 qpos_full (36,) array using MuJoCo IK.

    Args:
        robot:              Robot type string (currently only "unitree_g1").
        actual_human_height: Used to scale arm reach (metres).
    """

    def __init__(self, robot: str = "unitree_g1", actual_human_height: float = 1.7) -> None:
        xml_path = str(consts.TRACK_XML)
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = 0.001
        self._mj_data = mujoco.MjData(self._mj_model)

        # Body IDs for wrists (last keypoint in each arm chain)
        self._l_wrist_id = self._mj_model.body("left_wrist_yaw_link").id
        self._r_wrist_id = self._mj_model.body("right_wrist_yaw_link").id

        # Default arm joint angles (7 per arm)
        self._l_arm_default = _DEFAULT_QPOS[_L_ARM_QPOS].copy()
        self._r_arm_default = _DEFAULT_QPOS[_R_ARM_QPOS].copy()

        # Scale arm reach by user height ratio
        self._arm_scale = _ARM_SCALE * (actual_human_height / 1.70)

        # Calibration: set once on first frame so PICO origin → robot frame
        self._calib_head_pos: np.ndarray | None = None  # first head_pos in PICO frame

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-calibrate on next frame (call when switching into tracking mode)."""
        self._calib_head_pos = None

    def retarget(self, frame: PicoFrame) -> np.ndarray:
        """Convert a PicoFrame to qpos_full (36,).

        The returned array has the same layout as DEFAULT_QPOS:
            [root_xyz(3), root_quat(4), joints(29)]
        """
        # --- 1. Calibrate: lock PICO origin on first call ---
        if self._calib_head_pos is None:
            self._calib_head_pos = frame.head_pos.copy()

        # --- 2. Build output qpos from default ---
        qpos = _DEFAULT_QPOS.copy()

        # --- 3. Root orientation from head yaw ---
        # PICO head yaw (rotation around PICO +Y) → robot root yaw (around +Z)
        head_yaw_pico = _quat_to_yaw_pico(frame.head_rot)
        qpos[3:7] = _yaw_to_quat_robot(head_yaw_pico)

        # Optional: propagate small head pitch to waist_pitch (joint index 14)
        # This keeps the robot's upper torso aligned with the user's lean.
        head_pitch_pico = _head_pitch_pico(frame.head_rot)
        waist_pitch_idx = 7 + 14   # qpos index for waist_pitch_joint
        waist_roll_idx  = 7 + 13
        lo_wp, hi_wp = consts.RESTRICTED_JOINT_RANGE[14]
        lo_wr, hi_wr = consts.RESTRICTED_JOINT_RANGE[13]
        qpos[waist_pitch_idx] = float(np.clip(head_pitch_pico * 0.5, lo_wp, hi_wp))
        # Yaw already embedded in root quat; waist_yaw stays 0.

        # --- 4. Target wrist positions in robot world frame ---
        # Estimate chest position in PICO space (below head)
        chest_pico = (frame.head_pos - self._calib_head_pos) + _HEAD_TO_CHEST_PICO
        # Convert chest to robot frame (XY kept at 0 for in-place tracking)
        chest_robot = np.array([0.0, 0.0, 0.78], dtype=np.float32)  # root height

        # Controller offset from head in PICO frame → robot frame
        l_offset_pico = frame.left_pos  - frame.head_pos
        r_offset_pico = frame.right_pos - frame.head_pos

        l_offset_robot = (R_PICO2ROBOT @ l_offset_pico.astype(np.float64)).astype(np.float32)
        r_offset_robot = (R_PICO2ROBOT @ r_offset_pico.astype(np.float64)).astype(np.float32)

        # Target = shoulder position + scaled arm offset
        l_target = chest_robot + _CHEST_TO_L_SHOULDER_ROBOT + l_offset_robot * self._arm_scale
        r_target = chest_robot + _CHEST_TO_R_SHOULDER_ROBOT + r_offset_robot * self._arm_scale

        # --- 5. Set MuJoCo state and solve IK ---
        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = 0.0

        _arm_ik(self._mj_model, self._mj_data,
                l_target, self._l_wrist_id,
                _L_ARM_QPOS, _L_ARM_QVEL,
                _L_ARM_LO, _L_ARM_HI, self._l_arm_default)

        _arm_ik(self._mj_model, self._mj_data,
                r_target, self._r_wrist_id,
                _R_ARM_QPOS, _R_ARM_QVEL,
                _R_ARM_LO, _R_ARM_HI, self._r_arm_default)

        # --- 6. Read back solved arm angles ---
        qpos[_L_ARM_QPOS] = self._mj_data.qpos[_L_ARM_QPOS]
        qpos[_R_ARM_QPOS] = self._mj_data.qpos[_R_ARM_QPOS]

        return qpos.astype(np.float32)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _head_pitch_pico(q_wxyz: np.ndarray) -> float:
    """Extract pitch (rotation around PICO +X axis) from wxyz quaternion."""
    w, x, y, z = q_wxyz
    sinp = 2.0 * (w * x - z * y)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    return np.arcsin(sinp)
