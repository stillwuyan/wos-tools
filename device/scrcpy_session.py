from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np

from device.controller import DeviceController
from device.coordinates import frame_to_device
from log import get_logger

logger = get_logger("scrcpy")

DEVICE_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"


@dataclass(frozen=True)
class FrameSnapshot:
    frame: np.ndarray
    frame_id: int
    decode_time: float
    frame_size: tuple[int, int]
    device_size: tuple[int, int]


class ScrcpySession:
    def __init__(
        self,
        server_jar: Path,
        adb: str = "adb",
        serial: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 27183,
        socket_name: str = "scrcpy",
        force_install: bool = False,
        tap_jitter_pixels: int = 0,
        screen_on: bool = True,
        stay_awake: bool = False,
    ):
        self.server_jar = server_jar
        self.adb = adb
        self.serial = serial
        self.host = host
        self.port = port
        self.socket_name = socket_name
        self.force_install = force_install
        self.tap_jitter_pixels = tap_jitter_pixels
        self.screen_on = screen_on
        self.stay_awake = stay_awake

        self._listener: Optional[socket.socket] = None
        self._video_conn: Optional[socket.socket] = None
        self._control_conn: Optional[socket.socket] = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._decode_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream_ended = threading.Event()
        self._stream_ended.set()
        self._stream_error: Optional[Exception] = None
        self._reverse_active = False
        self._display_power_off_requested = False

        self.controller = DeviceController(tap_jitter_pixels=tap_jitter_pixels)
        self.display_width = 0
        self.display_height = 0
        self.codec_name = "h264"

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_size: Optional[tuple[int, int]] = None
        self._latest_device_size: Optional[tuple[int, int]] = None
        self._frame_id = 0
        self._latest_decode_time: Optional[float] = None
        self._frame_lock = threading.Lock()

    def _adb_cmd(self, *args: str) -> list[str]:
        cmd = [self.adb]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)
        return cmd

    def _run_adb(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(self._adb_cmd(*args), text=True, capture_output=True)
        if check and proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"adb {' '.join(args)} failed")
        return proc

    def _server_exists_on_device(self) -> bool:
        proc = subprocess.run(self._adb_cmd("shell", "test", "-f", DEVICE_SERVER_PATH), text=True, capture_output=True)
        return proc.returncode == 0

    def install_server(self) -> None:
        if not self.server_jar.is_file():
            raise RuntimeError(f"scrcpy-server not found: {self.server_jar}")
        if not self.force_install and self._server_exists_on_device():
            logger.info("scrcpy-server exists; skip copy")
            return
        logger.info("Copying scrcpy-server to device: %s -> %s", self.server_jar, DEVICE_SERVER_PATH)
        self._run_adb("push", str(self.server_jar), DEVICE_SERVER_PATH)

    def start_server_process(self) -> None:
        cmd = self._adb_cmd(
            "shell",
            f"CLASSPATH={DEVICE_SERVER_PATH}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            "3.3.4",
            "log_level=info",
            "video=true",
            "max_fps=15",
            "audio=false",
            "control=true",
            f"stay_awake={str(self.stay_awake).lower()}",
            "power_on=false",
            "tunnel_forward=false",
            "send_device_meta=false",
            "send_dummy_byte=false",
            "cleanup=false",
        )
        self._server_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _open_listener(self) -> None:
        self._listener = socket.create_server((self.host, self.port), backlog=1)
        self._listener.settimeout(15)
        self.port = self._listener.getsockname()[1]

    def _accept(self) -> socket.socket:
        assert self._listener is not None
        conn, _ = self._listener.accept()
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except BaseException:
            conn.close()
            raise
        return conn

    def start(self) -> None:
        if any(resource is not None for resource in (
            self._listener, self._video_conn, self._control_conn,
            self._server_proc, self._decode_thread,
        )) or self._reverse_active:
            raise RuntimeError("scrcpy session is already started or requires cleanup")
        self._stop_event.clear()
        self._stream_ended.clear()
        self._stream_error = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_size = None
            self._latest_device_size = None
            self._latest_decode_time = None
        try:
            self._open_listener()
            reverse_target = f"localabstract:{self.socket_name}"
            subprocess.run(
                self._adb_cmd("reverse", "--remove", reverse_target),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # ADB may create the mapping even if the command is interrupted.
            self._reverse_active = True
            self._run_adb("reverse", reverse_target, f"tcp:{self.port}")

            self.install_server()
            self.start_server_process()

            self._video_conn = self._accept()
            self._control_conn = self._accept()
            self.controller.set_connection(self._control_conn)

            self._read_stream_metadata()
            self._start_decode_thread()
            if self.stay_awake:
                logger.warning("stay_awake is ineffective with cleanup=false; device may sleep")
            if not self.screen_on:
                self._display_power_off_requested = True
                self.controller.set_display_power(False)
                logger.info("Requested device display power off (no device acknowledgement)")
        except BaseException:
            try:
                self.stop()
            except Exception:
                logger.exception("Failed to clean up scrcpy startup")
            raise

    def stop(self) -> None:
        self._stop_event.set()
        self._stream_ended.set()
        socket_error = None
        try:
            if self._display_power_off_requested:
                try:
                    self.controller.set_display_power(True)
                    self._display_power_off_requested = False
                    logger.info("Requested device display power on (no device acknowledgement)")
                except OSError as exc:
                    logger.exception("Failed to request device display power on")
                    socket_error = exc
                try:
                    if self._control_conn is not None:
                        self._control_conn.shutdown(socket.SHUT_WR)
                except OSError as exc:
                    logger.exception("Failed to shut down scrcpy control socket")
                    if socket_error is None:
                        socket_error = exc
                if self._server_proc is not None:
                    try:
                        self._server_proc.wait(timeout=2)
                        self._server_proc = None
                    except subprocess.TimeoutExpired:
                        logger.warning("scrcpy server did not exit after display power restore request")
        finally:
            self._close_resources()
        if socket_error is not None:
            raise socket_error

    def _close_resources(self) -> None:
        socket_error = None
        for sock in (self._video_conn, self._control_conn, self._listener):
            try:
                self._close_socket(sock)
            except OSError as exc:
                logger.exception("Failed to close scrcpy socket")
                if socket_error is None:
                    socket_error = exc
        self._video_conn = None
        self._control_conn = None
        self._listener = None
        self.controller.set_connection(None)
        with self._frame_lock:
            self._latest_frame = None
            self._latest_size = None
            self._latest_device_size = None
            self._latest_decode_time = None
        try:
            if self._server_proc is not None:
                if self._server_proc.poll() is None:
                    self._server_proc.kill()
                self._server_proc.wait(timeout=2)
                self._server_proc = None
        finally:
            try:
                if self._decode_thread is not None:
                    self._decode_thread.join(timeout=2.0)
                    if self._decode_thread.is_alive():
                        raise RuntimeError("scrcpy decode thread did not stop")
                    self._decode_thread = None
            finally:
                if self._reverse_active:
                    self._run_adb("reverse", "--remove", f"localabstract:{self.socket_name}")
                    self._reverse_active = False
        if socket_error is not None:
            raise socket_error

    def _close_socket(self, sock: Optional[socket.socket]) -> None:
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def _read_stream_metadata(self) -> None:
        assert self._video_conn is not None
        prefix = self._read_exact(self._video_conn, 12)
        if len(prefix) < 12:
            raise RuntimeError("Failed to read stream metadata")
        codec_id = int.from_bytes(prefix[0:4], "big")
        width = int.from_bytes(prefix[4:8], "big")
        height = int.from_bytes(prefix[8:12], "big")

        codec_map = {0: "h264", 1: "hevc", 2: "av1"}
        self.codec_name = codec_map.get(codec_id, "h264")
        with self._frame_lock:
            self.display_width = int(width)
            self.display_height = int(height)
            self.controller.set_display_size(self.display_width, self.display_height)
        logger.info("Stream: codec=%s size=%dx%d", self.codec_name, self.display_width, self.display_height)

    def _start_decode_thread(self) -> None:
        assert self._video_conn is not None
        video_conn = self._video_conn
        codec_ctx = av.codec.CodecContext.create(self.codec_name, "r")

        def decode_loop():
            try:
                while not self._stop_event.is_set():
                    hdr = self._read_exact(video_conn, 12)
                    if not hdr:
                        raise ConnectionError("Video stream ended")
                    u64 = int.from_bytes(hdr[0:8], "big")
                    is_config = bool((u64 >> 63) & 0x1)
                    packet_size = int.from_bytes(hdr[8:12], "big")
                    data = self._read_exact(video_conn, packet_size)
                    if not data:
                        raise ConnectionError("Video stream ended during a packet")
                    if is_config:
                        codec_ctx.extradata = data
                        continue

                    pkt = av.packet.Packet(data)
                    frames = codec_ctx.decode(pkt)
                    for frame in frames:
                        img = frame.to_ndarray(format="rgba")
                        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        fh, fw = bgr.shape[:2]
                        with self._frame_lock:
                            if self._stop_event.is_set():
                                break
                            self._latest_frame = bgr
                            self._latest_size = (fw, fh)
                            self._latest_device_size = (self.display_width, self.display_height)
                            self._frame_id += 1
                            self._latest_decode_time = time.monotonic()
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._stream_error = exc
                    logger.exception("scrcpy video stream failed")
            finally:
                with self._frame_lock:
                    self._latest_frame = None
                    self._latest_size = None
                    self._latest_device_size = None
                    self._latest_decode_time = None
                self._stream_ended.set()

        self._decode_thread = threading.Thread(target=decode_loop, daemon=True)
        self._decode_thread.start()

    @property
    def stream_ended(self) -> bool:
        """Whether the stream is unavailable, including before start and after stop."""
        return self._stream_ended.is_set()

    @property
    def stream_error(self) -> Optional[Exception]:
        """The last unexpected video-stream failure, reset by the next start."""
        return self._stream_error

    def wait_for_stream_end(self, timeout: Optional[float] = None) -> bool:
        """Wait until the stream ends or is stopped; return False on timeout."""
        return self._stream_ended.wait(timeout)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        self.check_stream_health()
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            self.check_stream_health()
            return self._latest_frame.copy()

    def get_frame_snapshot(self) -> Optional[FrameSnapshot]:
        """Return a copied BGR frame and metadata from the same decoded frame."""
        self.check_stream_health()
        with self._frame_lock:
            self.check_stream_health()
            if self._latest_frame is None:
                return None
            return FrameSnapshot(
                frame=self._latest_frame.copy(),
                frame_id=self._frame_id,
                decode_time=self._latest_decode_time,
                frame_size=self._latest_size,
                device_size=self._latest_device_size,
            )

    def get_latest_size(self) -> Optional[tuple[int, int]]:
        self.check_stream_health()
        with self._frame_lock:
            self.check_stream_health()
            return self._latest_size

    def check_stream_health(self) -> None:
        if self._stream_error is not None:
            raise RuntimeError("scrcpy video stream failed") from self._stream_error
        if self._stop_event.is_set() or self._stream_ended.is_set():
            raise RuntimeError("scrcpy video stream is stopped")

    def map_frame_to_device(self, x: int, y: int) -> Optional[tuple[int, int]]:
        size = self.get_latest_size()
        if size is None:
            return None
        return frame_to_device(x, y, size, (self.display_width, self.display_height))

    def _read_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf
