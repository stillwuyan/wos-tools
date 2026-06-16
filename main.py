#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
import os
from pathlib import Path
from typing import Optional
import re
import cv2
import av
import numpy as np
import importlib
import tkinter as tk
from tkinter import messagebox


from control import DeviceController
from tasks import handle_task
from log import configure_logging, get_logger

configure_logging()
logger = get_logger("main")


DEVICE_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"


class ScrcpyVideoUI:

    def __init__(
        self,
        server_jar: Path,
        adb: str = "adb",
        serial: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 27183,
        socket_name: str = "scrcpy",
        force_install: bool = False,
        selection_enabled: bool = False,
    ):
        self.server_jar = server_jar
        self.adb = adb
        self.serial = serial
        self.host = host
        self.port = port
        self.socket_name = socket_name
        self.force_install = force_install

        self._listener: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._task_cancel_event = threading.Event()

        self.display_width = 0
        self.display_height = 0
        self._control_conn: Optional[socket.socket] = None
        self.controller = DeviceController()

        self.selection_enabled = selection_enabled

        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.current_x = -1
        self.current_y = -1
        self.current_dev_x = -1
        self.current_dev_y = -1
        self.roi_image = None

        self._latest = None
        self._latest_lock = threading.Lock()
        self._latest_size = None

    def _adb_cmd(self, *args: str):
        cmd = [self.adb]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)
        return cmd

    def _get_frame_size(self, frame=None):
        if frame is None:
            return self._latest_size
        fh, fw = frame.shape[:2]
        return fw, fh

    def map_frame_to_device(self, x: int, y: int, frame=None):
        if not (self.display_width and self.display_height):
            return None
        size = self._get_frame_size(frame)
        if not size or size[0] <= 0 or size[1] <= 0:
            return None
        fw, fh = size
        dev_x = int(round(x * (self.display_width / float(fw))))
        dev_y = int(round(y * (self.display_height / float(fh))))
        return dev_x, dev_y

    def map_rect_frame_to_device(self, x1: int, y1: int, x2: int, y2: int, frame=None):
        p1 = self.map_frame_to_device(x1, y1, frame=frame)
        p2 = self.map_frame_to_device(x2, y2, frame=frame)
        if p1 is None or p2 is None:
            return None
        return (p1[0], p1[1], p2[0], p2[1])

    def _run_adb(self, *args: str, check: bool = True):
        proc = subprocess.run(self._adb_cmd(*args), text=True, capture_output=True)
        if check and proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"adb {' '.join(args)} failed")
        return proc

    def _open_listener(self):
        self._listener = socket.create_server((self.host, self.port), backlog=1)
        self._listener.settimeout(15)
        self.port = self._listener.getsockname()[1]
        logger.info("Listening for scrcpy connections on %s:%d", self.host, self.port)

    def _accept(self):
        assert self._listener is not None
        conn, addr = self._listener.accept()
        logger.info("Accepted connection from %s", addr)
        self._conn = conn
        return conn

    def install_server(self):
        if not self.server_jar.is_file():
            raise RuntimeError(f"scrcpy-server not found: {self.server_jar}")
        if not self.force_install and self._server_exists_on_device():
            logger.info("install: %s already exists on the device, skipping push", DEVICE_SERVER_PATH)
            return
        logger.info("install: pushing %s -> %s", self.server_jar.name, DEVICE_SERVER_PATH)
        self._run_adb("push", str(self.server_jar), DEVICE_SERVER_PATH)

    def _server_exists_on_device(self) -> bool:
        proc = subprocess.run(self._adb_cmd("shell", "test", "-f", DEVICE_SERVER_PATH), text=True, capture_output=True)
        return proc.returncode == 0

    def start_server_process(self):
        cmd = self._adb_cmd(
            "shell",
            f"CLASSPATH={DEVICE_SERVER_PATH}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            "3.3.4",
            "log_level=info",
            "video=true",
            "max-fps=15",
            "audio=false",
            "control=true",
            "power_on=false",
            "tunnel_forward=false",
            "send_device_meta=false",
            "send_dummy_byte=false",
            "cleanup=false",
        )
        self._server_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _resolve_display_size(self):
        proc = self._run_adb("shell", "wm", "size")
        matches = re.findall(r"(Override|Physical) size:\s*(\d+)x(\d+)", proc.stdout)
        if not matches:
            raise RuntimeError(f"Failed to parse display size: {proc.stdout.strip()}")

        for label, width, height in matches:
            if label == "Override":
                return (int(width), int(height))
        _, width, height = matches[0]
        return (int(width), int(height))

    def start(self):
        self._open_listener()
        reverse_target = f"localabstract:{self.socket_name}"
        subprocess.run(self._adb_cmd("reverse", "--remove", reverse_target), check=False)
        self._run_adb("reverse", reverse_target, f"tcp:{self.port}")
        logger.info("reverse tunnel set: %s -> tcp:%d", reverse_target, self.port)

        self.install_server()
        self.start_server_process()
        logger.info("Started scrcpy-server process (pid=%s)", getattr(self._server_proc, 'pid', '<unknown>'))

        video_conn = self._accept()
        video_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        control_conn = self._accept()
        control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._conn = video_conn
        self._control_conn = control_conn
        self.controller.set_connection(self._control_conn)

        return video_conn

    def _close_socket(self, sock):
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()
        self._close_socket(self._conn)
        self._close_socket(self._control_conn)
        try:
            if self._listener:
                self._listener.close()
            if self._server_proc:
                self._server_proc.kill()
        except Exception:
            pass
        if self._task_cancel_event:
            self._task_cancel_event.set()
        self._conn = None
        self._control_conn = None
        self._listener = None
        self._server_proc = None



    def _start_decode_thread(self, sock: socket.socket, codec_ctx):
        def decode_loop():
            while not self._stop_event.is_set():
                try:
                    hdr = self._read_exact(sock, 12)
                except Exception as exc:
                    logger.debug('decode_loop: read_exact failed: %s', exc)
                    break
                if not hdr:
                    break
                u64 = int.from_bytes(hdr[0:8], 'big')
                is_config = bool((u64 >> 63) & 0x1)
                packet_size = int.from_bytes(hdr[8:12], 'big')
                data = self._read_exact(sock, packet_size)
                if not data:
                    break
                if codec_ctx is None or is_config:
                    if is_config:
                        codec_ctx.extradata = data
                    continue
                pkt = av.packet.Packet(data)
                frames = codec_ctx.decode(pkt)
                for frame in frames:
                    img = frame.to_ndarray(format='rgba')
                    if isinstance(img, np.ndarray):
                        orig = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if img.shape[2] == 4 else cv2.COLOR_RGB2BGR)
                        with self._latest_lock:
                            self._latest = orig
                            fh, fw = orig.shape[:2]
                            self._latest_size = (fw, fh)

        t = threading.Thread(target=decode_loop, daemon=False)
        t.start()
        return t

    def _read_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except Exception:
                return b''
            if not chunk:
                return b''
            buf += chunk
        return buf

    def _get_window_size(self, win_name: str):
        try:
            rect = cv2.getWindowImageRect(win_name)
            return rect[2], rect[3]
        except Exception:
            return None

    def _map_window_to_frame(self, x: int, y: int, win_w: int, win_h: int):
        if not (win_w and win_h and self._latest_size):
            return None
        fw, fh = self._latest_size
        if not (fw and fh):
            return None
        fx = int(round((x / float(win_w)) * fw))
        fy = int(round((y / float(win_h)) * fh))
        return fx, fy

    def _map_window_to_device(self, x: int, y: int, win_w: int, win_h: int):
        if not (win_w and win_h and self.display_width and self.display_height):
            return None
        dx = int(round((x / float(win_w)) * self.display_width))
        dy = int(round((y / float(win_h)) * self.display_height))
        return dx, dy

    def _setup_window(self, win_name: str, display_size):
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if display_size is not None:
            cv2.resizeWindow(win_name, display_size[0], display_size[1])

        if self.selection_enabled:
            def mouse_callback(event, x, y, flags, param):
                self.current_x, self.current_y = x, y
                win_size = self._get_window_size(win_name)
                win_w, win_h = win_size if win_size else (display_size if display_size else (1, 1))

                frame_coord = self._map_window_to_frame(x, y, win_w, win_h)
                if frame_coord:
                    self.current_frame_x, self.current_frame_y = frame_coord
                else:
                    self.current_frame_x = self.current_frame_y = -1

                dev_coord = self._map_window_to_device(x, y, win_w, win_h)
                if dev_coord:
                    self.current_dev_x, self.current_dev_y = dev_coord
                else:
                    self.current_dev_x = self.current_dev_y = -1

                if event == cv2.EVENT_LBUTTONDOWN:
                    self.drawing = True
                    self.ix, self.iy = x, y
                elif event == cv2.EVENT_LBUTTONUP:
                    self.drawing = False
                    x1, y1 = min(self.ix, x), min(self.iy, y)
                    x2, y2 = max(self.ix, x), max(self.iy, y)
                    if x2 > x1 and y2 > y1:
                        f1 = self._map_window_to_frame(x1, y1, win_w, win_h)
                        f2 = self._map_window_to_frame(x2, y2, win_w, win_h)
                        with self._latest_lock:
                            if self._latest is not None and f1 and f2:
                                fx1, fy1 = f1
                                fx2, fy2 = f2
                                fw, fh = self._latest_size
                                fx1 = max(0, min(fw - 1, fx1))
                                fx2 = max(0, min(fw, fx2))
                                fy1 = max(0, min(fh - 1, fy1))
                                fy2 = max(0, min(fh, fy2))
                                self.roi_image = self._latest[fy1:fy2, fx1:fx2].copy() if fx2 > fx1 and fy2 > fy1 else None
                            else:
                                self.roi_image = None

                        d1 = self._map_window_to_device(x1, y1, win_w, win_h)
                        d2 = self._map_window_to_device(x2, y2, win_w, win_h)
                        if d1 and d2 and f1 and f2:
                            logger.info("Selected ROI: window=(%d,%d)->(%d,%d), frame=(%d,%d)->(%d,%d), device=(%d,%d)->(%d,%d)",
                                x1, y1, x2, y2, f1[0], f1[1], f2[0], f2[1], d1[0], d1[1], d2[0], d2[1])
                        elif f1 and f2:
                            logger.info("Selected ROI: window=(%d,%d)->(%d,%d), frame=(%d,%d)->(%d,%d)",
                                x1, y1, x2, y2, f1[0], f1[1], f2[0], f2[1])
                    else:
                        tol = 4
                        if abs(x - self.ix) <= tol and abs(y - self.iy) <= tol:
                            if self.current_dev_x != -1 and self.current_dev_y != -1:
                                logger.info("Mouse click at device=(%d, %d)", self.current_dev_x, self.current_dev_y)
                            else:
                                logger.info("Mouse click at window=(%d, %d)", x, y)

            cv2.setMouseCallback(win_name, mouse_callback, self)

    def _compute_display_size(self, width: Optional[int] = None, height: Optional[int] = None):
        """Return a (w,h) tuple scaled to one-third of provided or known display size, or None."""
        w = width if width is not None else (self.display_width if self.display_width else None)
        h = height if height is not None else (self.display_height if self.display_height else None)
        try:
            if not w or not h:
                return None
            return (max(1, int(w) // 3), max(1, int(h) // 3))
        except Exception:
            return None

    def match_template(self, template, threshold: float = 0.8, map_to_device: bool = False, debug: bool = False):
        start = time.perf_counter()
        if isinstance(template, (str, Path)):
            tpl = cv2.imread(str(template), cv2.IMREAD_COLOR)
            if tpl is None:
                raise RuntimeError(f"Failed to read template image: {template}")
        elif isinstance(template, np.ndarray):
            tpl = template.copy()
        else:
            raise TypeError("template must be a file path or numpy.ndarray")

        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl

        with self._latest_lock:
            if self._latest is None:
                return None
            frame = self._latest.copy()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        th, tw = tpl_gray.shape[:2]
        fh, fw = frame_gray.shape[:2]
        if th > fh or tw > fw:
            return None

        res = cv2.matchTemplate(frame_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        elapsed = time.perf_counter() - start
        logger.debug("Template match: score=%.4f, elapsed=%.6fs", max_val, elapsed)

        if max_val >= float(threshold):
            top_left = max_loc
            cx = top_left[0] + tw // 2
            cy = top_left[1] + th // 2

            if map_to_device and self.display_width and self.display_height:
                mapped = self.map_frame_to_device(cx, cy, frame=frame)
                if mapped is not None:
                    dev_x, dev_y = mapped
                    return (dev_x, dev_y, elapsed)
            return (cx, cy, elapsed)

        # On mismatch, save template + frame and a small info file for debugging.
        if debug:
            tpl_name = Path(template).stem if isinstance(template, (str, Path)) else "template_array"
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = Path(f"{tpl_name}_{ts}")
            out_dir.mkdir(parents=True, exist_ok=True)
            # write images (tpl and current frame)
            cv2.imwrite(str(out_dir / "template.png"), tpl)
            cv2.imwrite(str(out_dir / "frame.png"), frame)
            info = (
                f"threshold={threshold}\n"
                f"score={max_val}\n"
                f"max_loc={max_loc}\n"
                f"tpl_size={th}x{tw}\n"
                f"frame_size={fh}x{fw}\n"
                f"elapsed={elapsed:.6f}\n"
            )
            (out_dir / "info.txt").write_text(info, encoding="utf-8")
            logger.info("Saved mismatch artifacts to %s", out_dir)

        return None

    def run_gui(self, conn: socket.socket):
        prefix = self._read_exact(conn, 12)

        codec_id = width = height = None
        if len(prefix) >= 12:
            codec_id = int.from_bytes(prefix[0:4], 'big')
            width = int.from_bytes(prefix[4:8], 'big')
            height = int.from_bytes(prefix[8:12], 'big')

        codec_map = {0: 'h264', 1: 'hevc', 2: 'av1'}
        codec_name = codec_map.get(codec_id, 'h264')
        try:
            codec_ctx = av.codec.CodecContext.create(codec_name, 'r')
        except Exception:
            codec_ctx = None

        display_size = self._compute_display_size(width, height)
        if width and height:
            self.display_width = int(width)
            self.display_height = int(height)
            if self.controller:
                self.controller.set_display_size(int(width), int(height))

        dec_t = self._start_decode_thread(conn, codec_ctx)

        win_name = 'scrcpy-video'
        self._setup_window(win_name, display_size)

        try:
            self._ui_loop(win_name)
        finally:
            self._stop_event.set()
            self._close_socket(conn)
            if dec_t:
                dec_t.join(timeout=2.0)
            try:
                cv2.destroyWindow(win_name)
                cv2.waitKey(1)
            except Exception:
                pass

    def _ui_loop(self, win_name: str):
        while not self._stop_event.is_set():
            frame_to_show = None
            with self._latest_lock:
                if self._latest is not None:
                    frame_to_show = self._latest.copy()
            if frame_to_show is not None:
                win_size = self._get_window_size(win_name)
                if not win_size:
                    logger.debug('Window closed or invalid; recreating window')
                    display_size = self._compute_display_size()
                    self._setup_window(win_name, display_size)
                    win_w, win_h = display_size if display_size else frame_to_show.shape[1::-1]
                    time.sleep(0.05)
                else:
                    win_w, win_h = win_size

                disp = cv2.resize(frame_to_show, (win_w, win_h)) if win_w and win_h else frame_to_show

                if self.selection_enabled:
                    frame_x = getattr(self, 'current_frame_x', -1)
                    frame_y = getattr(self, 'current_frame_y', -1)
                    if frame_x != -1 and frame_y != -1:
                        dev = self.map_frame_to_device(frame_x, frame_y)
                        coord_text = f"Pos: ({dev[0]}, {dev[1]})" if dev is not None else f"Frame: ({frame_x}, {frame_y})"
                    else:
                        coord_text = f"Win: ({self.current_x}, {self.current_y})"
                    cv2.putText(disp, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if self.drawing:
                        cv2.rectangle(disp, (self.ix, self.iy), (self.current_x, self.current_y), (0, 255, 0), 2)
                cv2.imshow(win_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s') and self.selection_enabled:
                if self.roi_image is not None:
                    fname = f"selected_{int(time.time())}.png"
                    if cv2.imwrite(fname, self.roi_image):
                        logger.info("Saved ROI to %s", fname)
                        self.roi_image = None
                    else:
                        logger.error("Failed to save ROI to %s", fname)
            elif key == ord('t'):
                cfg = importlib.import_module('config')

                task_keys = []
                task_keys = list(cfg.task_list.keys())
                action_list = cfg.action_list

                if not task_keys:
                    messagebox.showinfo('Tasks', 'No tasks available in config.task_list')
                else:
                    def choose_task_dialog(options):
                        root = tk.Tk()
                        root.title('Choose task')
                        root.geometry('300x200')
                        root.attributes('-topmost', True)

                        sel = {'choice': None}

                        lb = tk.Listbox(root)
                        for it in options:
                            lb.insert(tk.END, it)
                        lb.pack(fill=tk.BOTH, expand=True)

                        def on_ok():
                            sel_idx = lb.curselection()
                            if sel_idx:
                                sel['choice'] = lb.get(sel_idx[0])
                            root.quit()

                        def on_cancel():
                            sel['choice'] = None
                            root.quit()

                        btn_frame = tk.Frame(root)
                        btn_frame.pack(fill=tk.X)
                        ok_btn = tk.Button(btn_frame, text='OK', command=on_ok)
                        ok_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
                        cancel_btn = tk.Button(btn_frame, text='Cancel', command=on_cancel)
                        cancel_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

                        root.protocol('WM_DELETE_WINDOW', on_cancel)
                        root.focus_force()
                        root.mainloop()
                        root.destroy()
                        return sel['choice']

                    choice = choose_task_dialog(task_keys)
                    if not choice:
                        logger.info('Task selection cancelled')
                    elif choice not in task_keys:
                        logger.error('Invalid task selected: %s', choice)
                    else:
                        task = cfg.task_list.get(choice)
                        logger.info('Selected task %s: %s', choice, task)
                        self._task_cancel_event.clear()
                        t = threading.Thread(target=handle_task, args=(self, choice, task, action_list), daemon=True)
                        t.start()
                        self._task_thread = t


def parse_args():
    parser = argparse.ArgumentParser(description='Play scrcpy server video via OpenCV + PyAV')
    parser.add_argument('--server-jar', help='Path to scrcpy-server artifact')
    parser.add_argument('--adb', default='adb')
    parser.add_argument('--serial')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=27183)
    parser.add_argument('--socket-name', default='scrcpy')
    parser.add_argument('--selection', action='store_true', help='Enable mouse coordinates and ROI selection/saving')
    return parser.parse_args()


def find_server_jar(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    env_path = os.environ.get("SCRCPY_SERVER_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent
    candidates = []
    for pattern in ("**/scrcpy-server.jar", "**/scrcpy-server"):
        for path in repo_root.glob(pattern):
            if path.is_file():
                candidates.append(path.resolve())

    if not candidates:
        raise RuntimeError(
            "Could not find a scrcpy-server build artifact. Specify the 3.3.4 scrcpy-server file via --server-jar or SCRCPY_SERVER_PATH."
        )

    candidates.sort(key=lambda item: len(str(item)))
    return candidates[0]


def main() -> int:
    args = parse_args()
    exit_code = 0
    ui = None
    try:
        server_jar = find_server_jar(args.server_jar)
        ui = ScrcpyVideoUI(server_jar=server_jar, adb=args.adb, serial=args.serial, host=args.host, port=args.port, socket_name=args.socket_name, selection_enabled=args.selection)
        conn = ui.start()
        ui.run_gui(conn)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        logger.exception("Fatal error")
        exit_code = 1
    finally:
        if ui is not None:
            try:
                ui.stop()
            except Exception:
                logger.exception("Error while stopping UI")
        return exit_code


if __name__ == '__main__':
    main()
