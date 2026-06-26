"""Convert PICO VR pose data to G1 robot qpos_full (36-D).

Supports two modes:
  • Arms-only (head + 2 controllers):
      - Root orientation from head yaw; XY fixed at 0; Z fixed at 0.78 m.
      - Waist pitch follows head pitch; waist yaw absorbed into root quat.
      - Arms: 7-DOF per arm via damped-LS Jacobian IK in MuJoCo.
      - Legs: held at default standing pose.

  • Full-body (head + 2 controllers + 2 foot trackers, optional waist tracker):
      - Root Z height tracks squat depth estimated from head/waist.
      - Legs: 6-DOF per leg via Jacobian IK; ankle targets derived from
              foot-tracker positions expressed relative to estimated root.
      - Arms: same Jacobian IK as arms-only mode.
      - Gracefully degrades to arms-only if foot data absent.

Foot-to-root IK formulation
----------------------------
The ankle target in robot world frame is:

    ankle_tgt = root_robot - R_PICO2ROBOT @ (root_pico - foot_pico) * leg_scale

where ``leg_scale = robot_vertical_leg_length / pico_vertical_leg_length``.
This correctly handles squats (root drops, foot stays on floor → knee bends)
and lateral/forward steps (foot moves in XZ → ankle moves in robot Y/X).

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

# Left leg joint indices inside qpos (7 = root offset)
_L_LEG_QPOS = np.arange(7,  13)                    # [7:13]
_R_LEG_QPOS = np.arange(13, 19)                    # [13:19]

# Left leg joint DOF indices in qvel (6 = root free DOFs)
_L_LEG_QVEL = np.arange(6,  12)                    # [6:12]
_R_LEG_QVEL = np.arange(12, 18)                    # [12:18]

# Joint limits for legs from RESTRICTED_JOINT_RANGE (indices 0-11)
_L_LEG_LO = np.array(
    [lo for lo, _ in consts.RESTRICTED_JOINT_RANGE[0:6]], dtype=np.float32)
_L_LEG_HI = np.array(
    [hi for _, hi in consts.RESTRICTED_JOINT_RANGE[0:6]], dtype=np.float32)
_R_LEG_LO = np.array(
    [lo for lo, _ in consts.RESTRICTED_JOINT_RANGE[6:12]], dtype=np.float32)
_R_LEG_HI = np.array(
    [hi for _, hi in consts.RESTRICTED_JOINT_RANGE[6:12]], dtype=np.float32)

# Left arm joint indices inside qpos (7 = root offset: 3 xyz + 4 quat)
_L_ARM_QPOS = np.arange(7 + 15, 7 + 22)            # [22:29]
_R_ARM_QPOS = np.arange(7 + 22, 7 + 29)            # [29:36]

# Left arm joint DOF indices in qvel (6 = root DOFs: 3 lin + 3 ang)
_L_ARM_QVEL = np.arange(6 + 15, 6 + 22)            # [21:28]
_R_ARM_QVEL = np.arange(6 + 22, 6 + 29)            # [28:35]

# Joint limits for arms from RESTRICTED_JOINT_RANGE (indices 15-28)
_L_ARM_LO = np.array(
    [lo for lo, _ in consts.RESTRICTED_JOINT_RANGE[15:22]], dtype=np.float32)
_L_ARM_HI = np.array(
    [hi for _, hi in consts.RESTRICTED_JOINT_RANGE[15:22]], dtype=np.float32)
_R_ARM_LO = np.array(
    [lo for lo, _ in consts.RESTRICTED_JOINT_RANGE[22:29]], dtype=np.float32)
_R_ARM_HI = np.array(
    [hi for _, hi in consts.RESTRICTED_JOINT_RANGE[22:29]], dtype=np.float32)

# Pelvis height as fraction of body height (standard anthropometry)
_PELVIS_HEIGHT_FRAC = 0.52

# PICO → MuJoCo world rotation (see module docstring)
R_PICO2ROBOT = np.array([
    [0., 0., -1.],
    [-1., 0., 0.],
    [0., 1., 0.],
], dtype=np.float32)

# Scale: human arm reach (~0.65 m) → robot arm reach from shoulder (~0.50 m)
_ARM_SCALE = 0.50 / 0.65

# Shoulder offset from chest in robot frame (+Y = left, -Y = right)
_CHEST_TO_L_SHOULDER = np.array([0.0, 0.18, 0.10], dtype=np.float32)
_CHEST_TO_R_SHOULDER = np.array([0.0, -0.18, 0.10], dtype=np.float32)

# IK hyper-parameters
_IK_MAX_ITER = 30
_IK_TOL = 3e-3     # metres
_IK_ALPHA = 0.4    # step size
_IK_DAMPING = 5e-2  # DLS damping
_IK_NULL_GAIN = 0.1  # pull toward default in null-space


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


def _limb_ik(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    target_pos: np.ndarray,
    end_body_id: int,
    qpos_ids: np.ndarray,
    qvel_ids: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    q_default: np.ndarray,
) -> None:
    """Damped-Jacobian IK in-place on mj_data.qpos. Works for arms and legs."""
    nv  = mj_model.nv
    n   = len(qvel_ids)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for _ in range(_IK_MAX_ITER):
        mujoco.mj_forward(mj_model, mj_data)
        pos_err = target_pos - mj_data.xpos[end_body_id]
        if np.linalg.norm(pos_err) < _IK_TOL:
            break

        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, end_body_id)
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
    """Convert a PicoFrame into a G1 qpos_full (36,) using MuJoCo IK.

    When foot trackers are present in the frame the class solves 6-DOF leg
    IK in addition to arm IK.  When foot data is absent (arms-only PICO
    setup) leg joints stay at the default standing pose.

    Args:
        robot: Robot type string (currently only "unitree_g1").
        actual_human_height: User height in metres; scales arm/leg reach.
    """

    def __init__(
        self,
        robot: str = "unitree_g1",  # reserved for future multi-robot support
        actual_human_height: float = 1.7,
    ) -> None:
        del robot  # XML path comes from consts.TRACK_XML (already robot-specific)
        xml_path = str(consts.TRACK_XML)
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = 0.001
        self._mj_data = mujoco.MjData(self._mj_model)

        # Body IDs for end-effectors
        self._l_wrist_id = self._mj_model.body("left_wrist_yaw_link").id
        self._r_wrist_id = self._mj_model.body("right_wrist_yaw_link").id
        self._l_ankle_id = self._mj_model.body("left_ankle_roll_link").id
        self._r_ankle_id = self._mj_model.body("right_ankle_roll_link").id

        # Default joint angles for null-space IK pulls
        self._l_arm_default = _DEFAULT_QPOS[_L_ARM_QPOS].copy()
        self._r_arm_default = _DEFAULT_QPOS[_R_ARM_QPOS].copy()
        self._l_leg_default = _DEFAULT_QPOS[_L_LEG_QPOS].copy()
        self._r_leg_default = _DEFAULT_QPOS[_R_LEG_QPOS].copy()

        # Default ankle positions in robot world frame (from forward kinematics)
        self._mj_data.qpos[:] = _DEFAULT_QPOS
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._default_l_ankle = self._mj_data.xpos[self._l_ankle_id].copy()
        self._default_r_ankle = self._mj_data.xpos[self._r_ankle_id].copy()

        # Arm reach scale
        self._arm_scale = _ARM_SCALE * (actual_human_height / 1.70)

        # Leg scale: robot vertical leg length / expected PICO vertical leg length.
        # Calibrated from actual foot data on first frame; pre-seeded from height.
        robot_vleg = float(_DEFAULT_QPOS[2] - self._default_l_ankle[2])
        pico_vleg_est = actual_human_height * _PELVIS_HEIGHT_FRAC
        self._leg_scale = robot_vleg / max(pico_vleg_est, 0.1)

        # Calibration state (set on first frame)
        self._calib_head_pos: np.ndarray | None = None
        self._calib_lf_pico: np.ndarray | None = None
        self._calib_rf_pico: np.ndarray | None = None
        # Estimated pelvis Y in PICO frame at calibration (Y-up, floor = 0)
        self._calib_root_y: float = actual_human_height * _PELVIS_HEIGHT_FRAC

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-calibrate on next frame."""
        self._calib_head_pos = None
        self._calib_lf_pico = None
        self._calib_rf_pico = None

    def retarget(self, frame: PicoFrame) -> np.ndarray:
        """Convert a PicoFrame to qpos_full (36,).

        Layout: [root_xyz(3), root_quat(4), joints(29)]
        """
        # --- 1. Calibrate on first frame ---
        if self._calib_head_pos is None:
            self._calib_head_pos = frame.head_pos.copy()
            if frame.left_foot_pos is not None:
                self._calib_lf_pico = frame.left_foot_pos.copy()
                self._calib_rf_pico = frame.right_foot_pos.copy()
                # Refine leg scale from actual foot positions
                feet_avg_y = float(
                    (frame.left_foot_pos[1] + frame.right_foot_pos[1]) / 2.0
                )
                pico_vleg = self._calib_root_y - feet_avg_y
                robot_vleg = float(_DEFAULT_QPOS[2] - self._default_l_ankle[2])
                self._leg_scale = robot_vleg / max(pico_vleg, 0.05)

        # --- 2. Base qpos ---
        qpos = _DEFAULT_QPOS.copy()

        # --- 3. Root orientation (head yaw → robot root yaw around Z) ---
        head_yaw = _quat_to_yaw_pico(frame.head_rot)
        qpos[3:7] = _yaw_to_quat_robot(head_yaw)

        # --- 4. Waist pitch tracks head pitch (gentle lean follow) ---
        head_pitch = _head_pitch_pico(frame.head_rot)
        waist_pitch_idx = 7 + 14
        lo_wp, hi_wp = consts.RESTRICTED_JOINT_RANGE[14]
        qpos[waist_pitch_idx] = float(
            np.clip(head_pitch * 0.5, lo_wp, hi_wp)
        )

        # --- 5. Root height: estimate from head (or waist tracker) ---
        if frame.waist_pos is not None:
            current_root_y = float(frame.waist_pos[1])
        else:
            # Root moves proportionally with head when squatting
            ratio = self._calib_root_y / max(
                float(self._calib_head_pos[1]), 0.1
            )
            current_root_y = float(frame.head_pos[1]) * ratio

        root_z = float(_DEFAULT_QPOS[2]) + (
            (current_root_y - self._calib_root_y) * self._leg_scale
        )
        qpos[2] = max(root_z, 0.50)  # safety floor

        # --- 6. Leg IK (only when foot trackers present) ---
        has_feet = (
            frame.left_foot_pos is not None
            and self._calib_lf_pico is not None
        )
        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = 0.0

        if has_feet:
            root_robot = qpos[:3].astype(np.float64)  # [x, y, root_z]

            # foot-to-root vectors in PICO frame → ankle targets in robot frame
            # ankle_tgt = root_robot - R @ (root_pico - foot_pico) * scale
            root_pico = np.array(
                [float(self._calib_lf_pico[0] + self._calib_rf_pico[0])
                 / 2.0,
                 current_root_y,
                 float(self._calib_lf_pico[2] + self._calib_rf_pico[2])
                 / 2.0],
                dtype=np.float64,
            )
            lf2root = root_pico - frame.left_foot_pos.astype(np.float64)
            rf2root = root_pico - frame.right_foot_pos.astype(np.float64)

            l_ankle_tgt = (
                root_robot
                - R_PICO2ROBOT.astype(np.float64) @ lf2root * self._leg_scale
            )
            r_ankle_tgt = (
                root_robot
                - R_PICO2ROBOT.astype(np.float64) @ rf2root * self._leg_scale
            )
            # Clamp: ankle must stay at or above near-floor level
            l_ankle_tgt[2] = max(l_ankle_tgt[2], 0.01)
            r_ankle_tgt[2] = max(r_ankle_tgt[2], 0.01)

            _limb_ik(
                self._mj_model, self._mj_data,
                l_ankle_tgt, self._l_ankle_id,
                _L_LEG_QPOS, _L_LEG_QVEL,
                _L_LEG_LO, _L_LEG_HI, self._l_leg_default,
            )
            _limb_ik(
                self._mj_model, self._mj_data,
                r_ankle_tgt, self._r_ankle_id,
                _R_LEG_QPOS, _R_LEG_QVEL,
                _R_LEG_LO, _R_LEG_HI, self._r_leg_default,
            )
            qpos[_L_LEG_QPOS] = self._mj_data.qpos[_L_LEG_QPOS]
            qpos[_R_LEG_QPOS] = self._mj_data.qpos[_R_LEG_QPOS]

        # --- 7. Arm IK ---
        chest_robot = np.array([0.0, 0.0, qpos[2]], dtype=np.float32)

        l_off = (R_PICO2ROBOT @ (frame.left_pos - frame.head_pos)
                 .astype(np.float64)).astype(np.float32)
        r_off = (R_PICO2ROBOT @ (frame.right_pos - frame.head_pos)
                 .astype(np.float64)).astype(np.float32)

        l_target = (chest_robot + _CHEST_TO_L_SHOULDER
                    + l_off * self._arm_scale)
        r_target = (chest_robot + _CHEST_TO_R_SHOULDER
                    + r_off * self._arm_scale)

        # Re-seed MuJoCo with solved leg state before arm IK
        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = 0.0

        _limb_ik(
            self._mj_model, self._mj_data,
            l_target, self._l_wrist_id,
            _L_ARM_QPOS, _L_ARM_QVEL,
            _L_ARM_LO, _L_ARM_HI, self._l_arm_default,
        )
        _limb_ik(
            self._mj_model, self._mj_data,
            r_target, self._r_wrist_id,
            _R_ARM_QPOS, _R_ARM_QVEL,
            _R_ARM_LO, _R_ARM_HI, self._r_arm_default,
        )
        qpos[_L_ARM_QPOS] = self._mj_data.qpos[_L_ARM_QPOS]
        qpos[_R_ARM_QPOS] = self._mj_data.qpos[_R_ARM_QPOS]

        return qpos.astype(np.float32)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _head_pitch_pico(q_wxyz: np.ndarray) -> float:
    """Extract pitch (rotation around PICO +X axis) from wxyz quaternion."""
    w, x, y, z = q_wxyz
    sinp = float(np.clip(2.0 * (w * x - z * y), -1.0, 1.0))
    return float(np.arcsin(sinp))
