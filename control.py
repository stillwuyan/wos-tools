import struct
import time
from typing import Optional

from log import get_logger

logger = get_logger(__name__)

TOUCH_EVENT_FORMAT = struct.Struct(
    ">BBQiiHHHII"
)

from utils import _u16fp


class DeviceController:
    def __init__(self, control_conn=None, display_width: int = 0, display_height: int = 0):
        self._control_conn = control_conn
        self.display_width = display_width
        self.display_height = display_height

    def set_connection(self, conn):
        self._control_conn = conn

    def set_display_size(self, w: int, h: int):
        self.display_width = int(w)
        self.display_height = int(h)

    def send_touch(self, action: int, x: int, y: int, pressure: float) -> None:
        if self._control_conn is None:
            raise RuntimeError("Control connection is not established")
        if not (0 <= x < self.display_width and 0 <= y < self.display_height):
            raise ValueError(f"Coordinates out of bounds: ({x}, {y}) not in {self.display_width}x{self.display_height}")

        payload = TOUCH_EVENT_FORMAT.pack(
            2,
            action,
            0xFFFFFFFFFFFFFFFE,
            x,
            y,
            self.display_width,
            self.display_height,
            _u16fp(pressure),
            0,
            0,
        )
        self._control_conn.sendall(payload)

    def tap(self, x: int, y: int, hold_ms: int = 30) -> None:
        self.send_touch(0, x, y, 1.0)
        time.sleep(max(hold_ms, 0) / 1000.0)
        self.send_touch(1, x, y, 0.0)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, steps: Optional[int] = None) -> None:
        duration_ms = max(duration_ms, 1)
        if steps is None:
            steps = max(2, duration_ms // 16)

        self.send_touch(0, x1, y1, 1.0)
        start = time.monotonic()
        for index in range(1, steps):
            ratio = index / steps
            x = round(x1 + (x2 - x1) * ratio)
            y = round(y1 + (y2 - y1) * ratio)
            target = start + (duration_ms / 1000.0) * ratio
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            self.send_touch(2, x, y, 1.0)
        self.send_touch(1, x2, y2, 0.0)
