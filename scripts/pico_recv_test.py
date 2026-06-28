"""Test PICO TCP connection (Robotoolkit) and display received data.

Usage:
    python -m scripts.pico_recv_test [--port 63901] [--host 0.0.0.0]

Start this first, then in PICO Robotoolkit → PC Service → enter this machine's
IP (e.g. 192.168.2.160).  Press Ctrl-C to stop.

Output columns:
    HEAD  : head position (m) and yaw (deg)
    LC/RC : left/right controller positions (m) + trigger/grip
    LF/RF : left/right foot tracker positions (m) — shown only when present
    WP    : waist tracker position — shown only when present
    Hz    : received packet rate
"""

from __future__ import annotations

import time
import math
import argparse

import numpy as np

from deploy.pico.client import PicoClient, PicoFrame


def _yaw_deg(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    siny = 2.0 * (w * y + z * x)
    cosy = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.atan2(siny, cosy))


def _fmt3(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]"


def _print_frame(frame: PicoFrame, hz: float) -> None:
    lines = []
    lines.append(
        f"HEAD  pos={_fmt3(frame.head_pos)}  yaw={_yaw_deg(frame.head_rot):+6.1f}°"
    )
    lines.append(
        f"LC    pos={_fmt3(frame.left_pos)}  trig={frame.left_trigger:.2f}  grip={frame.left_grip:.2f}"
    )
    lines.append(
        f"RC    pos={_fmt3(frame.right_pos)}  trig={frame.right_trigger:.2f}  grip={frame.right_grip:.2f}"
    )
    if frame.left_foot_pos is not None:
        lines.append(f"LF    pos={_fmt3(frame.left_foot_pos)}  (foot tracker)")
    else:
        lines.append("LF    -- no foot tracker --")

    if frame.right_foot_pos is not None:
        lines.append(f"RF    pos={_fmt3(frame.right_foot_pos)}  (foot tracker)")
    else:
        lines.append("RF    -- no foot tracker --")

    if frame.waist_pos is not None:
        lines.append(f"WP    pos={_fmt3(frame.waist_pos)}  (waist tracker)")

    lines.append(f"Hz    {hz:.1f}")

    # Move cursor up N lines and overwrite
    n = len(lines)
    print(f"\033[{n}A", end="")
    for line in lines:
        print(f"\033[2K{line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PICO receive test")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=63901)
    args = parser.parse_args()

    client = PicoClient(host=args.host, port=args.port)
    client.start_thread()

    print(f"[pico_recv_test] TCP server on {args.host}:{args.port}  (Ctrl-C to stop)")
    print("In PICO Robotoolkit → PC Service → enter this machine's IP …")

    # Reserve blank lines for in-place update
    for _ in range(9):
        print()

    frame_count = 0
    t0 = time.monotonic()
    last_ts = t0
    hz = 0.0

    try:
        while True:
            frame = client.get_frame_data(timeout=1.0)
            if frame is None:
                print("\033[9A", end="")
                print(f"\033[2K[pico_recv_test] No packet in 1 s — is the PICO app running and sending to {args.host}:{args.port}?")
                for _ in range(8):
                    print()
                continue

            now = time.monotonic()
            frame_count += 1
            dt = now - last_ts
            if dt >= 0.5:
                hz = frame_count / (now - t0)
                t0 = now
                frame_count = 0
            last_ts = now

            _print_frame(frame, hz)
            time.sleep(0.05)  # ~20 fps display refresh

    except KeyboardInterrupt:
        print("\n[pico_recv_test] Stopped.")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
