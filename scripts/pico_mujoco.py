"""Live PICO → MuJoCo viewer: retarget VR pose onto G1 in real time.

Usage:
    python -m scripts.pico_mujoco [--port 63901] [--height 1.70]

Controls:
    R        — re-calibrate (reset T-pose reference on next frame)
    Ctrl-C   — quit

Requires PICO Robotoolkit 1.1.1:
    PC Service → enter this machine's IP → Connect
Foot and waist trackers are used automatically when Body tracking is enabled.
"""

from __future__ import annotations

import time
import argparse
import threading

import mujoco
import mujoco.viewer
import numpy as np

from tracking import constants as consts
from deploy.pico.client import PicoClient
from deploy.pico.retarget_pico import PicoRetargeter


def main() -> None:
    parser = argparse.ArgumentParser(description="PICO → MuJoCo live preview")
    parser.add_argument("--port",   type=int,   default=63901, help="TCP port for Robotoolkit PC Service")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--height", type=float, default=1.70,  help="Your height in metres")
    parser.add_argument("--freq",   type=int,   default=50,    help="Target display Hz")
    args = parser.parse_args()

    # --- MuJoCo model (viewer only — retargeter has its own internal copy) ---
    mj_model = mujoco.MjModel.from_xml_path(str(consts.TRACK_XML))
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # --- PICO client ---
    client = PicoClient(host=args.host, port=args.port)
    client.start_thread()
    print(f"[pico_mujoco] TCP server on {args.host}:{args.port}  — PICO Robotoolkit → PC Service → connect  (Ctrl-C to quit)")

    # --- Retargeter ---
    retargeter = PicoRetargeter(actual_human_height=args.height)

    # Shared state
    _lock  = threading.Lock()
    _qpos  = mj_data.qpos.copy()
    _alive = [True]

    def _recv_loop() -> None:
        """Background thread: receive PICO frames and retarget."""
        while _alive[0]:
            frame = client.get_frame_data(timeout=0.5)
            if frame is None:
                continue
            try:
                qpos = retargeter.retarget(frame)
                with _lock:
                    _qpos[:] = qpos
            except Exception as e:
                print(f"[retarget] {e}")

    recv_thread = threading.Thread(target=_recv_loop, daemon=True)
    recv_thread.start()

    dt = 1.0 / args.freq

    # --- MuJoCo passive viewer (blocks until window closed) ---
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0  # follow pelvis
        viewer.cam.distance  = 3.0
        viewer.cam.elevation = -20

        print("[pico_mujoco] Viewer open. Press R in this terminal to re-calibrate.")
        print("              Waiting for PICO packets …")

        t_next = time.monotonic()

        while viewer.is_running():
            t_now = time.monotonic()

            # Update displayed qpos
            with _lock:
                mj_data.qpos[:len(_qpos)] = _qpos

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Simple rate limiting
            t_next += dt
            sleep = t_next - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                t_next = time.monotonic()

    _alive[0] = False
    client.stop()
    print("[pico_mujoco] Done.")


# Allow 'R' key to re-calibrate from the terminal (non-blocking readline)
def _maybe_recalibrate(retargeter: PicoRetargeter) -> None:
    import sys, select
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch.lower() == "r":
            retargeter.reset()
            print("[pico_mujoco] Re-calibrated — hold T-pose.")


if __name__ == "__main__":
    main()
