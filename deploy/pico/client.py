"""UDP client that receives 6-DOF pose data from a PICO VR headset.

The PICO side (Unity app) must send UDP packets to this host/port in JSON:

    {
      "t":    1234567890.123,        // UNIX timestamp (optional, float)
      "head": {"p": [x,y,z], "q": [w,x,y,z]},
      "lc":   {"p": [x,y,z], "q": [w,x,y,z], "trig": 0.0, "grip": 0.0, "joy": [jx, jy]},
      "rc":   {"p": [x,y,z], "q": [w,x,y,z], "trig": 0.0, "grip": 0.0, "joy": [jx, jy]}
    }

Coordinate convention expected from the PICO app (OpenXR stage space):
  - +Y up, -Z forward (direction the user faces at initialization), +X right
  - Positions in metres, quaternions as [w, x, y, z]
  - Origin: floor level below the user's initial head position

The retarget layer (retarget_pico.py) handles the frame conversion to the
MuJoCo/robot world frame (+Z up, +X forward).

Default port: 9864 (must match the PICO Unity app setting).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PicoFrame:
    timestamp: float
    # Head-mounted display
    head_pos: np.ndarray      # (3,) metres, PICO world frame
    head_rot: np.ndarray      # (4,) [w,x,y,z] quaternion
    # Left controller
    left_pos: np.ndarray      # (3,)
    left_rot: np.ndarray      # (4,) [w,x,y,z]
    left_trigger: float       # 0..1
    left_grip: float          # 0..1
    left_joystick: np.ndarray # (2,) [x, y]
    # Right controller
    right_pos: np.ndarray
    right_rot: np.ndarray
    right_trigger: float
    right_grip: float
    right_joystick: np.ndarray


class PicoClient:
    """Thread-safe UDP receiver for PICO pose packets.

    Matches the interface expected by _retarget_worker in deploy/retarget.py:
      - start_thread()
      - get_frame_data(timeout) -> PicoFrame | None
      - stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9864) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: PicoFrame | None = None
        self._stop_evt = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_thread(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.1)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[PicoClient] Listening on {self._host}:{self._port}")

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
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_evt.is_set():
            try:
                raw, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            frame = _parse_packet(raw)
            if frame is not None:
                with self._lock:
                    self._latest = frame


# ------------------------------------------------------------------
# Packet parsing
# ------------------------------------------------------------------

def _v3(d: dict, key: str) -> np.ndarray:
    return np.array(d[key], dtype=np.float32)


def _v4(d: dict, key: str) -> np.ndarray:
    q = np.array(d[key], dtype=np.float32)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _parse_packet(raw: bytes) -> PicoFrame | None:
    try:
        d = json.loads(raw.decode("utf-8"))
        head = d["head"]
        lc   = d["lc"]
        rc   = d["rc"]
        return PicoFrame(
            timestamp     = float(d.get("t", time.time())),
            head_pos      = _v3(head, "p"),
            head_rot      = _v4(head, "q"),
            left_pos      = _v3(lc, "p"),
            left_rot      = _v4(lc, "q"),
            left_trigger  = float(lc.get("trig", 0.0)),
            left_grip     = float(lc.get("grip", 0.0)),
            left_joystick = np.array(lc.get("joy", [0.0, 0.0]), dtype=np.float32),
            right_pos     = _v3(rc, "p"),
            right_rot     = _v4(rc, "q"),
            right_trigger = float(rc.get("trig", 0.0)),
            right_grip    = float(rc.get("grip", 0.0)),
            right_joystick= np.array(rc.get("joy", [0.0, 0.0]), dtype=np.float32),
        )
    except Exception as e:
        print(f"[PicoClient] packet parse error: {e}")
        return None
