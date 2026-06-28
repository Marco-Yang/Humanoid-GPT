"""TCP server that receives pose data from a PICO VR headset via Robotoolkit 1.1.1.

The PICO runs the Robotoolkit app which connects as a TCP CLIENT to this server
on port 63901 and streams JSON tracking data at ~100 Hz.

Wire protocol per message:
  [0x3F][type: u8][body_len: u32 LE][body: N bytes][ts: u32 LE][dev_id: u32 LE][0xA5]

Type 0x6D = Tracking. Body is a JSON object:
  {"functionName": "Tracking", "value": "<inner-json-string>"}

The inner JSON (value field, double-encoded):
  {
    "predictTime": float,
    "Head":       {"pose": "x,y,z,qw,qx,qy,qz", "status": int},
    "Controller": {
      "left":  {"pose": "x,y,z,qw,qx,qy,qz", "trigger": f, "grip": f,
                "axisX": f, "axisY": f, ...},
      "right": {...}
    },
    "BodyTracking": {             // present when PICO Motion Tracker active
      "LeftFoot":  {"pose": "x,y,z,qw,qx,qy,qz"},
      "RightFoot": {"pose": "x,y,z,qw,qx,qy,qz"},
      "Waist":     {"pose": "x,y,z,qw,qx,qy,qz"}   // optional
    },
    "timeStampNs": int,
    "Input": int
  }

Coordinate convention (OpenXR stage space):
  +Y up, -Z forward, +X right; positions in metres; quaternions [w, x, y, z].
  Origin: floor level below the user's head at initialisation.

The retarget layer (retarget_pico.py) converts to MuJoCo/robot frame (+Z up, +X forward).

Default port: 63901 (Robotoolkit PC Service port).
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

_MSG_TRACKING = 0x6D
_HEADER_OVERHEAD = 15  # 1(0x3F) + 1(type) + 4(len) + 4(ts) + 4(dev_id) + 1(0xA5)


@dataclass
class PicoFrame:
    timestamp: float
    # Head-mounted display
    head_pos: np.ndarray       # (3,) metres, PICO world frame
    head_rot: np.ndarray       # (4,) [w,x,y,z] quaternion
    # Left controller
    left_pos: np.ndarray       # (3,)
    left_rot: np.ndarray       # (4,) [w,x,y,z]
    left_trigger: float        # 0..1
    left_grip: float           # 0..1
    left_joystick: np.ndarray  # (2,) [x, y]
    # Right controller
    right_pos: np.ndarray
    right_rot: np.ndarray
    right_trigger: float
    right_grip: float
    right_joystick: np.ndarray
    # Optional full-body trackers (PICO Motion Tracker / BodyTracking)
    # Ankle joints (body-tracking joints 10/11 — confirmed empirically)
    left_foot_pos:  np.ndarray | None = None  # (3,) or None
    left_foot_rot:  np.ndarray | None = None  # (4,) [w,x,y,z] or None
    right_foot_pos: np.ndarray | None = None
    right_foot_rot: np.ndarray | None = None
    # Pelvis root (joint 0)
    waist_pos:      np.ndarray | None = None
    waist_rot:      np.ndarray | None = None
    # Neck (joint 12) — used for upper-body orientation reference (SONIC-style)
    neck_pos:        np.ndarray | None = None
    neck_rot:        np.ndarray | None = None
    # Wrist/hand joints (joint 22 = left, 23 = right) — SMPL-estimated from body tracking
    left_wrist_pos:  np.ndarray | None = None
    left_wrist_rot:  np.ndarray | None = None
    right_wrist_pos: np.ndarray | None = None
    right_wrist_rot: np.ndarray | None = None


class PicoClient:
    """Thread-safe TCP server that receives PICO pose data from Robotoolkit.

    Matches the interface expected by _retarget_worker in deploy/retarget.py:
      - start_thread()
      - get_frame_data(timeout) -> PicoFrame | None
      - stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 63901) -> None:
        self._host = host
        self._port = port
        self._srv_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: PicoFrame | None = None
        self._stop_evt = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_thread(self) -> None:
        self._srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv_sock.bind((self._host, self._port))
        self._srv_sock.listen(1)
        self._srv_sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"[PicoClient] TCP server on {self._host}:{self._port} — "
              f"connect PICO Robotoolkit → PC Service → {self._host}:{self._port}")

    def get_frame_data(self, timeout: float = 0.5) -> PicoFrame | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            time.sleep(0.005)
        return None

    def stop(self) -> None:
        self._stop_evt.set()
        if self._srv_sock is not None:
            try:
                self._srv_sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._srv_sock is not None
        while not self._stop_evt.is_set():
            try:
                conn, addr = self._srv_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            print(f"[PicoClient] PICO connected from {addr}")
            t = threading.Thread(target=self._recv_loop, args=(conn,), daemon=True)
            t.start()

    def _recv_loop(self, conn: socket.socket) -> None:
        buf = b""
        conn.settimeout(0.1)
        try:
            while not self._stop_evt.is_set():
                try:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    msgs, consumed = _parse_stream(buf)
                    buf = buf[consumed:]
                    for frame in msgs:
                        with self._lock:
                            self._latest = frame
                except socket.timeout:
                    pass
        except OSError:
            pass
        finally:
            conn.close()
            print("[PicoClient] PICO disconnected")


# ------------------------------------------------------------------
# Stream parser
# ------------------------------------------------------------------

def _parse_stream(data: bytes) -> tuple[list[PicoFrame], int]:
    frames: list[PicoFrame] = []
    i = 0
    last_good = 0
    while i < len(data):
        if data[i] != 0x3F:
            i += 1
            continue
        if i + 6 > len(data):
            break
        msg_type = data[i + 1]
        body_len = struct.unpack_from("<I", data, i + 2)[0]
        if body_len > 65536:
            i += 1
            continue
        end = i + _HEADER_OVERHEAD + body_len - 1  # index of 0xA5
        if end >= len(data):
            break
        if data[end] != 0xA5:
            i += 1
            continue
        body = data[i + 6: i + 6 + body_len]
        if msg_type == _MSG_TRACKING:
            frame = _parse_tracking(body)
            if frame is not None:
                frames.append(frame)
        last_good = end + 1
        i = end + 1
    return frames, last_good


def _pose_str(s: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse "x,y,z,qw,qx,qy,qz" → (pos(3,), rot(4,))."""
    v = [float(x) for x in s.split(",")]
    pos = np.array(v[:3], dtype=np.float32)
    rot = np.array(v[3:7], dtype=np.float32)
    norm = np.linalg.norm(rot)
    if norm > 1e-8:
        rot /= norm
    else:
        rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return pos, rot


def _parse_tracking(body: bytes) -> PicoFrame | None:
    try:
        outer = json.loads(body.decode("utf-8"))
        d = json.loads(outer["value"])

        head_pos, head_rot = _pose_str(d["Head"]["pose"])
        lc = d["Controller"]["left"]
        rc = d["Controller"]["right"]
        lc_pos, lc_rot = _pose_str(lc["pose"])
        rc_pos, rc_rot = _pose_str(rc["pose"])

        # Optional body tracking joints.
        lf_pos = lf_rot = rf_pos = rf_rot = wp_pos = wp_rot = None
        nk_pos = nk_rot = lw_pos = lw_rot = rw_pos = rw_rot = None

        # Primary: BodyTracking named fields (Robotoolkit 1.1.1 TCP stream).
        bt = d.get("BodyTracking", {})
        if bt:
            if "LeftFoot" in bt:
                lf_pos, lf_rot = _pose_str(bt["LeftFoot"]["pose"])
            if "RightFoot" in bt:
                rf_pos, rf_rot = _pose_str(bt["RightFoot"]["pose"])
            if "Waist" in bt:
                wp_pos, wp_rot = _pose_str(bt["Waist"]["pose"])

        # Secondary: Body.joints SMPL array (24 joints, SDK path).
        # 0=pelvis, 10=L-ankle, 11=R-ankle, 12=neck, 22=L-wrist, 23=R-wrist
        joints = d.get("Body", {}).get("joints", [])
        if len(joints) >= 12 and wp_pos is None:
            wp_pos, wp_rot = _pose_str(joints[0]["p"])
            lf_pos, lf_rot = _pose_str(joints[10]["p"])
            rf_pos, rf_rot = _pose_str(joints[11]["p"])
        if len(joints) >= 24:
            nk_pos, nk_rot = _pose_str(joints[12]["p"])
            lw_pos, lw_rot = _pose_str(joints[22]["p"])
            rw_pos, rw_rot = _pose_str(joints[23]["p"])

        ts = float(d.get("predictTime", time.time()))

        return PicoFrame(
            timestamp      = ts,
            head_pos       = head_pos,
            head_rot       = head_rot,
            left_pos       = lc_pos,
            left_rot       = lc_rot,
            left_trigger   = float(lc.get("trigger", 0.0)),
            left_grip      = float(lc.get("grip", 0.0)),
            left_joystick  = np.array([lc.get("axisX", 0.0),
                                       lc.get("axisY", 0.0)], dtype=np.float32),
            right_pos      = rc_pos,
            right_rot      = rc_rot,
            right_trigger  = float(rc.get("trigger", 0.0)),
            right_grip     = float(rc.get("grip", 0.0)),
            right_joystick = np.array([rc.get("axisX", 0.0),
                                       rc.get("axisY", 0.0)], dtype=np.float32),
            left_foot_pos   = lf_pos,
            left_foot_rot   = lf_rot,
            right_foot_pos  = rf_pos,
            right_foot_rot  = rf_rot,
            waist_pos       = wp_pos,
            waist_rot       = wp_rot,
            neck_pos        = nk_pos,
            neck_rot        = nk_rot,
            left_wrist_pos  = lw_pos,
            left_wrist_rot  = lw_rot,
            right_wrist_pos = rw_pos,
            right_wrist_rot = rw_rot,
        )
    except Exception as e:
        print(f"[PicoClient] tracking parse error: {e}")
        return None
