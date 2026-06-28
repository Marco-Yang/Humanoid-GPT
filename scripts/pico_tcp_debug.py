"""TCP debug server — print raw data received from PICO VR Toolkit app.

Usage:
    python -m scripts.pico_tcp_debug [--port 9864]

Run this, then connect from the PICO app. The script prints every byte
received so we can see the protocol / data format.
"""

from __future__ import annotations

import socket
import argparse
import threading
import time


def handle_client(conn: socket.socket, addr: tuple) -> None:
    print(f"\n[+] Connected from {addr}")
    try:
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                print(f"[-] {addr} disconnected")
                break
            buf += chunk
            # Try to print as UTF-8 text first, fallback to hex
            try:
                text = chunk.decode("utf-8")
                print(f"[data utf8 {len(chunk)}B] {text[:300]}")
            except UnicodeDecodeError:
                print(f"[data hex  {len(chunk)}B] {chunk[:64].hex()}")
    except Exception as e:
        print(f"[!] {addr} error: {e}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9864)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(5)
    print(f"[pico_tcp_debug] TCP listening on {args.host}:{args.port}")
    print("Connect from PICO VR Toolkit app now …  (Ctrl-C to stop)\n")

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[pico_tcp_debug] Stopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
