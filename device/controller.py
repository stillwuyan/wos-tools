import random
import struct
import time
from operator import index as integer_index
from typing import Optional


TOUCH_EVENT_FORMAT = struct.Struct(">BBQiiHHHII")
KEYCODE_EVENT_FORMAT = struct.Struct(">BBIII")


def _u16fp(value: float) -> int:
    value = max(0.0, min(1.0, value))
    return int(round(value * 0xFFFF))


class DeviceController:
    def __init__(self, control_conn=None, display_width: int = 0, display_height: int = 0,
                 tap_jitter_pixels: int = 0):
        if type(tap_jitter_pixels) is not int or tap_jitter_pixels < 0:
            raise ValueError("tap_jitter_pixels must be a non-negative integer")
        self._control_conn = control_conn
        self.display_width = display_width
        self.display_height = display_height
        self.tap_jitter_pixels = tap_jitter_pixels

    def set_connection(self, conn):
        self._control_conn = conn

    def set_display_size(self, width: int, height: int):
        self.display_width = int(width)
        self.display_height = int(height)

    def set_display_power(self, on: bool) -> None:
        if self._control_conn is None:
            raise RuntimeError("Control connection is not established")
        self._control_conn.sendall(struct.pack(">BB", 10, int(on)))

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

    def _apply_tap_jitter(self, x: int, y: int) -> tuple[int, int]:
        """Randomize a tap within +/- tap_jitter_pixels, clamped to the display.

        Zero jitter (the default) returns the coordinates unchanged.
        """
        jitter = self.tap_jitter_pixels
        if jitter == 0:
            return x, y
        x = x + random.randrange(-jitter, jitter + 1)
        y = y + random.randrange(-jitter, jitter + 1)
        if self.display_width > 0:
            x = max(0, min(x, self.display_width - 1))
        if self.display_height > 0:
            y = max(0, min(y, self.display_height - 1))
        return x, y

    def tap(self, x: int, y: int, hold_ms: int = 30) -> None:
        x, y = self._apply_tap_jitter(x, y)
        self.send_touch(0, x, y, 1.0)
        try:
            time.sleep(max(hold_ms, 0) / 1000.0)
            self.send_touch(1, x, y, 0.0)
        except BaseException:
            try:
                self.send_touch(1, x, y, 0.0)
            except BaseException:
                pass
            raise

    def back(self) -> None:
        if self._control_conn is None:
            raise RuntimeError("Control connection is not established")
        down = KEYCODE_EVENT_FORMAT.pack(0, 0, 4, 0, 0)
        up = KEYCODE_EVENT_FORMAT.pack(0, 1, 4, 0, 0)
        self._control_conn.sendall(down + up)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, steps: Optional[int] = None) -> None:
        duration_ms = max(duration_ms, 1)
        if steps is None:
            steps = max(2, duration_ms // 16)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("steps must be an integer >= 2")
        for x, y in ((x1, y1), (x2, y2)):
            integer_index(x)
            integer_index(y)
            if not (0 <= x < self.display_width and 0 <= y < self.display_height):
                raise ValueError(f"Coordinates out of bounds: ({x}, {y}) not in {self.display_width}x{self.display_height}")

        self.send_touch(0, x1, y1, 1.0)
        last_x, last_y = x1, y1
        try:
            start = time.monotonic()
            for index in range(1, steps + 1):
                ratio = index / steps
                target = start + (duration_ms / 1000.0) * ratio
                remaining = target - time.monotonic()
                while remaining > 0:
                    time.sleep(remaining)
                    remaining = target - time.monotonic()
                if index == steps:
                    self.send_touch(1, x2, y2, 0.0)
                else:
                    x = round(x1 + (x2 - x1) * ratio)
                    y = round(y1 + (y2 - y1) * ratio)
                    self.send_touch(2, x, y, 1.0)
                    last_x, last_y = x, y
        except BaseException:
            try:
                self.send_touch(1, last_x, last_y, 0.0)
            except BaseException:
                pass
            raise
