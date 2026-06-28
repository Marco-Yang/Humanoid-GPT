"""Convert PICO VR pose data to G1 robot qpos_full (36-D).

Supports two arm-tracking modes (auto-selected each frame):

  • Body-joint arms (SONIC-style, preferred when body tracking active):
      - Uses SMPL body joints 22/23 (left/right wrist) from PICO body tracking.
      - Non-mirror, egocentric: user's left arm drives robot's left arm.
      - Calibrated by pressing BOTH grips (left_grip + right_grip > 0.5)
        while in T-pose; default pose held until then.
        Captures inv(neck_rot) and wrist offsets vs G1 FK defaults
        (follows SONIC ThreePointPose calibration logic).
      - No arm_scale in body-joint mode; controller fallback still uses scale.

  • Controller arms (fallback when body tracking absent):
      - Uses controller positions relative to HMD head.
      - Same egocentric convention as body-joint mode.

Leg tracking (requires foot trackers):
  • Full-body (head + foot trackers + waist tracker):
      - Root Z height from waist tracker (joint 0) with leg_scale.
      - Legs: 6-DOF per leg via Jacobian IK; ankle targets from foot-tracker
              positions (joints 10/11) relative to estimated root.
  • Gracefully degrades to default standing pose if foot data absent.

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
  User faces -Z; user's RIGHT side = PICO +X, user's LEFT side = PICO -X.
MuJoCo/Robot world:               +Z up,  +X forward,  +Y left

The rotation matrix from PICO→Robot is:
    R[0,:] = [0, 0, -1]   robot +X = -PICO Z  (forward)
    R[1,:] = [-1, 0,  0]  robot +Y = -PICO X  (user left → robot left)
    R[2,:] = [0, 1,  0]   robot +Z =  PICO Y  (up)
"""

from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as _SRot

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
    [-1., 0., 0.],   # PICO -X (user's left) → robot +Y (robot's left)
    [0., 1., 0.],
], dtype=np.float32)

# Scale: human arm reach (~0.65 m) → robot arm reach from shoulder (~0.50 m)
_ARM_SCALE = 0.50 / 0.65

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
    nv = mj_model.nv
    n = len(qvel_ids)
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
        del robot  # XML path from consts.TRACK_XML (robot-specific)
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

        # Default end-effector positions in robot world frame (FK)
        self._mj_data.qpos[:] = _DEFAULT_QPOS
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._default_l_ankle = self._mj_data.xpos[self._l_ankle_id].copy()
        self._default_r_ankle = self._mj_data.xpos[self._r_ankle_id].copy()
        self._default_l_wrist = self._mj_data.xpos[self._l_wrist_id].copy()
        self._default_r_wrist = self._mj_data.xpos[self._r_wrist_id].copy()

        # Arm reach scale
        self._arm_scale = _ARM_SCALE * (actual_human_height / 1.70)

        # Robot vertical leg length (pelvis Z above ankle Z in default pose).
        # Used to compute leg_scale once calibrated foot positions are known.
        self._robot_vleg = float(_DEFAULT_QPOS[2] - self._default_l_ankle[2])
        # Pre-seeded leg scale (refined on first frame from actual PICO data)
        pico_vleg_est = actual_human_height * _PELVIS_HEIGHT_FRAC
        self._leg_scale = self._robot_vleg / max(pico_vleg_est, 0.1)

        # Head calibration (first frame — only for yaw/head-height fallback)
        self._calib_head_pos: np.ndarray | None = None
        # Pelvis Y in PICO at calibration; from waist tracker or estimate
        self._calib_root_y: float = actual_human_height * _PELVIS_HEIGHT_FRAC

        # Leg calibration (button-triggered: both triggers for 5 frames).
        # Legs hold default pose until calibration fires.
        self._calib_lf_pico: np.ndarray | None = None
        self._calib_rf_pico: np.ndarray | None = None
        self._triggers_down_count: int = 0

        # Controller-delta arm calibration (captured at leg calibration time).
        # Stores (controller - waist) at the neutral pose so arm tracking
        # computes the DELTA from neutral → applied to default wrist position.
        self._calib_lc_rel: np.ndarray | None = None  # (3,) PICO frame
        self._calib_rc_rel: np.ndarray | None = None

        # SMPL body-joint arm calibration (both grips, requires SDK wrist data).
        # Dead path with TCP-only stream; kept for future SDK compatibility.
        self._calib_neck_inv = None         # _SRot | None
        self._calib_lw_offset: np.ndarray | None = None  # (3,) robot frame
        self._calib_rw_offset: np.ndarray | None = None
        self._grips_down_count: int = 0

        # Default wrist body-local (relative to robot root at DEFAULT_QPOS).
        # Used as the arm target baseline before any arm delta is applied.
        _def_root = _DEFAULT_QPOS[:3].copy()
        self._default_l_wrist_rel = self._default_l_wrist - _def_root
        self._default_r_wrist_rel = self._default_r_wrist - _def_root

        # IK warm start: carry previous frame's joint solution
        self._prev_qpos: np.ndarray | None = None

        print(
            "[PicoRetargeter] Both TRIGGERS = leg calibration (stand still)."
        )
        print(
            "[PicoRetargeter] Both GRIPS    = arm calibration (T-pose)."
        )

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all calibration; next button presses re-calibrate."""
        self._calib_head_pos = None
        self._calib_lf_pico = None
        self._calib_rf_pico = None
        self._calib_lc_rel = None
        self._calib_rc_rel = None
        self._calib_neck_inv = None
        self._calib_lw_offset = None
        self._calib_rw_offset = None
        self._grips_down_count = 0
        self._triggers_down_count = 0
        self._prev_qpos = None

    def retarget(self, frame: PicoFrame) -> np.ndarray:
        """Convert a PicoFrame to qpos_full (36,).

        Layout: [root_xyz(3), root_quat(4), joints(29)]
        """
        # --- 1. Head calibration (first frame only) ---
        if self._calib_head_pos is None:
            self._calib_head_pos = frame.head_pos.copy()

        # --- 1b. Leg calibration (both triggers held for 5 frames) ---
        triggers_down = (
            frame.left_trigger > 0.5 and frame.right_trigger > 0.5
        )
        if triggers_down:
            self._triggers_down_count += 1
            if (self._triggers_down_count == 5
                    and frame.left_foot_pos is not None
                    and frame.waist_pos is not None):
                self._capture_leg_calibration(frame)
                self._triggers_down_count = 100  # prevent re-trigger
        else:
            self._triggers_down_count = 0

        # --- 2. Base qpos (warm start from previous IK solution) ---
        if self._prev_qpos is not None:
            qpos = self._prev_qpos.copy()
        else:
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

        # --- 5. Root height ---
        # joint 0 (waist_pos) = pelvis tracker → use directly when available.
        # Fallback: track head displacement (pelvis follows head in squat).
        if frame.waist_pos is not None:
            current_root_y = float(frame.waist_pos[1])
        else:
            dh = float(frame.head_pos[1]) - float(self._calib_head_pos[1])
            current_root_y = self._calib_root_y + dh
        root_z = (
            float(_DEFAULT_QPOS[2])
            + (current_root_y - self._calib_root_y) * self._leg_scale
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
            # Use live waist X/Z so relative vector is correct even when user
            # has walked from the calibration position.
            if frame.waist_pos is not None:
                root_pico = np.array([
                    float(frame.waist_pos[0]),
                    current_root_y,
                    float(frame.waist_pos[2]),
                ], dtype=np.float64)
            else:
                root_pico = np.array([
                    float(
                        self._calib_lf_pico[0] + self._calib_rf_pico[0]
                    ) / 2.0,
                    current_root_y,
                    float(
                        self._calib_lf_pico[2] + self._calib_rf_pico[2]
                    ) / 2.0,
                ], dtype=np.float64)
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
        # SONIC-style: press both grips for 5 consecutive frames to calibrate.
        grips_down = frame.left_grip > 0.5 and frame.right_grip > 0.5
        if grips_down:
            self._grips_down_count += 1
            if (self._grips_down_count == 5
                    and frame.left_wrist_pos is not None
                    and frame.waist_pos is not None
                    and frame.neck_rot is not None):
                self._capture_arm_calibration(frame)
                self._grips_down_count = 100  # prevent re-trigger while held
        else:
            self._grips_down_count = 0

        # Prefer SONIC-style body joint tracking when calibrated.
        has_body_arms = (
            frame.left_wrist_pos is not None
            and frame.waist_pos is not None
            and self._calib_neck_inv is not None
        )

        if has_body_arms:
            # Wrist-to-waist in robot frame (non-mirror via R_PICO2ROBOT)
            lw_rel = (
                R_PICO2ROBOT.astype(np.float64)
                @ (frame.left_wrist_pos - frame.waist_pos).astype(np.float64)
            )
            rw_rel = (
                R_PICO2ROBOT.astype(np.float64)
                @ (frame.right_wrist_pos - frame.waist_pos).astype(np.float64)
            )
            # Apply neck-inverse (SONIC ThreePointPose._apply_calibration),
            # subtract stored offset → body-local wrist target in robot frame.
            # Result = g1_fk_default_rel + neck_inv.apply(delta_from_calib_pos)
            lw_cal = (
                self._calib_neck_inv.apply(lw_rel) - self._calib_lw_offset
            )
            rw_cal = (
                self._calib_neck_inv.apply(rw_rel) - self._calib_rw_offset
            )
            # Convert body-local → world by adding current root XYZ
            root_xyz = qpos[:3].astype(np.float64)
            l_target = (lw_cal + root_xyz).astype(np.float32)
            r_target = (rw_cal + root_xyz).astype(np.float32)
        elif (self._calib_lc_rel is not None and frame.waist_pos is not None):
            # Controller-delta arm tracking (waist-relative, calibrated neutral).
            # Mirrors SONIC ThreePointPose body-local delta approach but using
            # controller positions instead of body-joint wrist positions.
            R = R_PICO2ROBOT.astype(np.float64)
            wp = frame.waist_pos.astype(np.float64)
            lc_rel = frame.left_pos.astype(np.float64) - wp
            rc_rel = frame.right_pos.astype(np.float64) - wp
            delta_l = R @ (lc_rel - self._calib_lc_rel) * self._arm_scale
            delta_r = R @ (rc_rel - self._calib_rc_rel) * self._arm_scale
            root_xyz = qpos[:3].astype(np.float64)
            l_target = (self._default_l_wrist_rel + delta_l + root_xyz).astype(np.float32)
            r_target = (self._default_r_wrist_rel + delta_r + root_xyz).astype(np.float32)
        else:
            # No calibration yet — hold default wrist positions
            root_xyz = qpos[:3].astype(np.float64)
            l_target = (self._default_l_wrist_rel + root_xyz).astype(np.float32)
            r_target = (self._default_r_wrist_rel + root_xyz).astype(np.float32)

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

        self._prev_qpos = qpos.copy()
        return qpos.astype(np.float32)

    # ------------------------------------------------------------------

    def _capture_leg_calibration(self, frame: PicoFrame) -> None:
        """Capture neutral-stance reference for both leg IK and arm tracking.

        Call while standing still with arms in natural resting position.
        - Legs: captures foot & waist positions → sets leg_scale
        - Arms: captures controller positions relative to waist → used as
          the 'neutral' baseline for controller-delta arm tracking
        """
        assert frame.left_foot_pos is not None
        assert frame.waist_pos is not None

        # Leg calibration
        self._calib_lf_pico = frame.left_foot_pos.copy()
        self._calib_rf_pico = frame.right_foot_pos.copy()
        feet_avg_y = float(
            (frame.left_foot_pos[1] + frame.right_foot_pos[1]) / 2.0
        )
        self._calib_root_y = float(frame.waist_pos[1])
        pico_vleg = self._calib_root_y - feet_avg_y
        self._leg_scale = self._robot_vleg / max(pico_vleg, 0.05)

        # Arm neutral: controller position relative to waist in PICO frame
        wp = frame.waist_pos.astype(np.float64)
        self._calib_lc_rel = (frame.left_pos.astype(np.float64) - wp)
        self._calib_rc_rel = (frame.right_pos.astype(np.float64) - wp)
        lo = self._calib_lc_rel
        ro = self._calib_rc_rel
        print(
            f"[PicoRetargeter] Leg calibration done. "
            f"vleg={pico_vleg:.3f}  scale={self._leg_scale:.3f}  "
            f"L-ctrl [{lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}]  "
            f"R-ctrl [{ro[0]:.2f},{ro[1]:.2f},{ro[2]:.2f}]"
        )

    def _capture_arm_calibration(self, frame: PicoFrame) -> None:
        """SONIC ThreePointPose-style arm calibration (button-triggered).

        Call while in T-pose (arms extended sideways, body upright).
        Stores inv(neck_rot) and wrist offsets relative to G1 FK defaults so
        that _apply_calibration yields: target = g1_fk_rel + neck_inv(delta).
        """
        assert frame.neck_rot is not None
        assert frame.left_wrist_pos is not None
        assert frame.waist_pos is not None

        R = R_PICO2ROBOT.astype(np.float64)

        # Neck orientation in robot frame (conjugate similarity transform)
        neck_pico = _SRot.from_quat(frame.neck_rot, scalar_first=True)
        neck_robot = _SRot.from_matrix(
            R @ neck_pico.as_matrix() @ R.T
        )
        self._calib_neck_inv = neck_robot.inv()

        # Wrist-to-waist vectors in robot frame
        lw_rel = R @ (frame.left_wrist_pos - frame.waist_pos).astype(
            np.float64
        )
        rw_rel = R @ (frame.right_wrist_pos - frame.waist_pos).astype(
            np.float64
        )

        # Apply neck-inverse (same as SONIC _capture_calibration step 2)
        lw_corr = self._calib_neck_inv.apply(lw_rel)
        rw_corr = self._calib_neck_inv.apply(rw_rel)

        # offset = neck_inv(calib_rel) - g1_fk_rel
        # tracking: neck_inv(cur_rel) - offset = g1_fk_rel + neck_inv(delta)
        self._calib_lw_offset = lw_corr - self._default_l_wrist_rel
        self._calib_rw_offset = rw_corr - self._default_r_wrist_rel

        lo = self._calib_lw_offset
        ro = self._calib_rw_offset
        print(
            f"[PicoRetargeter] Arm calibration done. "
            f"L-off [{lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f}] "
            f"R-off [{ro[0]:.3f},{ro[1]:.3f},{ro[2]:.3f}]"
        )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _head_pitch_pico(q_wxyz: np.ndarray) -> float:
    """Extract pitch (rotation around PICO +X axis) from wxyz quaternion."""
    w, x, y, z = q_wxyz
    sinp = float(np.clip(2.0 * (w * x - z * y), -1.0, 1.0))
    return float(np.arcsin(sinp))
