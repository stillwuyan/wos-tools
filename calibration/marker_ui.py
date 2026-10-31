from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from calibration.records import MarkRecordStore
from device.coordinates import frame_to_device, frame_to_window, window_to_frame
from device.scrcpy_session import ScrcpySession
from log import get_logger

logger = get_logger("marker")


class MarkerUI:
    def __init__(self, session: ScrcpySession, output_dir: Path, window_scale: float = 0.33,
                 home: Optional[Path] = None):
        self.session = session
        self.window_scale = max(0.1, min(1.0, window_scale))
        self.window_name = "wos-marker"
        self.store = MarkRecordStore(output_dir, home=home)

        self.show_help = True
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.current_x = -1
        self.current_y = -1
        self.current_frame_x = -1
        self.current_frame_y = -1
        self.current_dev_x = -1
        self.current_dev_y = -1
        self._drag_start_frame = None
        self.pending_record = None
        self.pending_roi = None
        self.pending_frame_rect = None

        self._display_w = 0
        self._display_h = 0
        self._displayed_frame = None
        self._displayed_device_size = None
        self._selection_frame = None
        self._selection_device_size = None
        self._callback_error = None

    def run(self) -> None:
        try:
            self._setup_window()
            while True:
                self._check_stream_health()
                snapshot = self.session.get_frame_snapshot()
                frame = self._selection_frame if self._selection_frame is not None else (snapshot.frame if snapshot else None)
                if frame is None:
                    key = cv2.waitKey(30) & 0xFF
                    self._check_stream_health()
                    if self._handle_key(key):
                        break
                    continue

                fh, fw = frame.shape[:2]
                self._ensure_display_size(fw, fh)
                self._displayed_frame = frame
                self._displayed_device_size = (
                    self._selection_device_size if self._selection_frame is not None
                    else snapshot.device_size
                )
                disp = cv2.resize(frame, (self._display_w, self._display_h))

                self._draw_overlay(disp)
                if self.drawing and self._drag_start_frame is not None and self.current_frame_x >= 0:
                    start = frame_to_window(*self._drag_start_frame, (self._display_w, self._display_h), (fw, fh))
                    end = frame_to_window(self.current_frame_x, self.current_frame_y, (self._display_w, self._display_h), (fw, fh))
                    cv2.rectangle(disp, start, end, (0, 255, 0), 2)
                elif self.pending_frame_rect is not None:
                    x1, y1, x2, y2 = self.pending_frame_rect
                    start = frame_to_window(x1, y1, (self._display_w, self._display_h), (fw, fh))
                    end = frame_to_window(x2, y2, (self._display_w, self._display_h), (fw, fh))
                    cv2.rectangle(disp, start, end, (0, 255, 255), 2)
                cv2.imshow(self.window_name, disp)

                key = cv2.waitKey(1) & 0xFF
                self._check_stream_health()
                if self._handle_key(key):
                    break
        finally:
            self._clear_selection()
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                logger.warning("Could not destroy marker window", exc_info=True)

    def _handle_key(self, key: int) -> bool:
        if key in (27, ord("q")):
            return True
        if key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("s"):
            self._confirm_pending()
        elif key == 26:
            try:
                step_name = self.store.undo()
                if step_name is not None:
                    logger.info("Undid %s.", step_name)
            except Exception:
                logger.exception("Failed to undo marker record")
        return False

    def _setup_window(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)

    def _ensure_display_size(self, frame_w: int, frame_h: int) -> None:
        if self._display_w > 0 and self._display_h > 0:
            return
        self._display_w = max(1, int(frame_w * self.window_scale))
        self._display_h = max(1, int(frame_h * self.window_scale))
        cv2.resizeWindow(self.window_name, self._display_w, self._display_h)

    def _map_window_to_frame(self, x: int, y: int) -> Optional[tuple[int, int]]:
        frame = self._selection_frame if self._selection_frame is not None else self._displayed_frame
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        # HighGUI reports mouse coordinates in the image supplied to imshow,
        # even when its window is resized.
        return window_to_frame(x, y, (self._display_w, self._display_h), (fw, fh))

    def _clear_selection(self) -> None:
        self.drawing = False
        self._drag_start_frame = None
        self.pending_record = None
        self.pending_roi = None
        self.pending_frame_rect = None
        self._selection_frame = None
        self._selection_device_size = None

    def _check_stream_health(self) -> None:
        try:
            if self._callback_error is not None:
                raise self._callback_error
            self.session.check_stream_health()
        except Exception:
            self._clear_selection()
            raise

    def _on_mouse_event(self, event, x, y, flags, param):
        try:
            self._check_stream_health()
            self._process_mouse_event(event, x, y)
        except Exception as exc:
            # Exceptions must be surfaced by the event loop, not swallowed by HighGUI.
            self._clear_selection()
            if self._callback_error is None:
                self._callback_error = exc

    def _process_mouse_event(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clear_selection()
            self._selection_frame = self._displayed_frame
            self._selection_device_size = self._displayed_device_size
        self.current_x, self.current_y = x, y
        frame_xy = self._map_window_to_frame(x, y)
        self.current_frame_x = self.current_frame_y = -1
        self.current_dev_x = self.current_dev_y = -1
        if frame_xy is not None:
            self.current_frame_x, self.current_frame_y = frame_xy
            frame = self._selection_frame if self._selection_frame is not None else self._displayed_frame
            device_size = self._selection_device_size if self._selection_frame is not None else self._displayed_device_size
            fh, fw = frame.shape[:2]
            dev_xy = frame_to_device(*frame_xy, (fw, fh), device_size) if device_size else None
            if dev_xy is not None:
                self.current_dev_x, self.current_dev_y = dev_xy

        if event == cv2.EVENT_LBUTTONDOWN:
            if frame_xy is not None:
                self.drawing = True
                self._drag_start_frame = frame_xy
                self.ix, self.iy = x, y
            else:
                self._clear_selection()
        elif event == cv2.EVENT_LBUTTONUP:
            if not self.drawing or self._drag_start_frame is None:
                return
            self.drawing = False
            start = self._drag_start_frame
            self._drag_start_frame = None
            if frame_xy is None:
                logger.warning("Selection ended outside the image.")
                self._clear_selection()
                return
            if abs(x - self.ix) <= 4 and abs(y - self.iy) <= 4:
                self._prepare_point_click()
            else:
                self._prepare_roi(start, frame_xy)
            if self.pending_record is None:
                self._clear_selection()
        elif event == cv2.EVENT_RBUTTONUP:
            self._confirm_pending()

    def _prepare_point_click(self) -> None:
        if self.current_frame_x < 0 or self.current_frame_y < 0:
            logger.warning("No frame coordinate available for point.")
            return
        if self.current_dev_x < 0 or self.current_dev_y < 0:
            logger.warning("No device coordinate available for point.")
            return
        self.pending_record = {
            "action": "point",
            "time": int(time.time()),
            "frame": {"x": self.current_frame_x, "y": self.current_frame_y},
            "device": {"x": self.current_dev_x, "y": self.current_dev_y},
            "frame_size": {"width": self._selection_frame.shape[1], "height": self._selection_frame.shape[0]},
            "device_size": {"width": self._selection_device_size[0], "height": self._selection_device_size[1]},
        }
        self.pending_roi = None
        self.pending_frame_rect = None
        logger.info("Point: (%d,%d) [pending]", self.current_dev_x, self.current_dev_y)

    def _prepare_roi(self, f1: tuple[int, int], f2: tuple[int, int]) -> None:
        frame = self._selection_frame
        if frame is None:
            logger.warning("No video frame available for ROI.")
            return

        fx1, fy1 = f1
        fx2, fy2 = f2
        lx, rx = min(fx1, fx2), max(fx1, fx2)
        ty, by = min(fy1, fy2), max(fy1, fy2)
        fh, fw = frame.shape[:2]
        if rx <= lx or by <= ty or rx >= fw or by >= fh:
            logger.warning("Invalid ROI for the current video frame.")
            return
        device_size = self._selection_device_size
        d1 = frame_to_device(lx, ty, (fw, fh), device_size) if device_size else None
        d2 = frame_to_device(rx, by, (fw, fh), device_size) if device_size else None
        if d1 is None or d2 is None:
            logger.warning("No device coordinates available for ROI.")
            return
        self.pending_roi = frame[ty:by, lx:rx].copy()
        self.pending_frame_rect = (lx, ty, rx, by)
        center = ((lx + rx - 1) // 2, (ty + by - 1) // 2)
        device_center = frame_to_device(*center, (fw, fh), device_size)
        self.pending_record = {
            "action": "template",
            "time": int(time.time()),
            "frame_rect": {"x1": lx, "y1": ty, "x2": rx, "y2": by},
            "device_rect": {"x1": d1[0], "y1": d1[1], "x2": d2[0], "y2": d2[1]},
            "frame_size": {"width": fw, "height": fh},
            "device_size": {"width": device_size[0], "height": device_size[1]},
            "frame_center": {"x": center[0], "y": center[1]},
            "device_center": {"x": device_center[0], "y": device_center[1]},
        }
        logger.info("ROI: (%d,%d)-(%d,%d) [pending]", lx, ty, rx, by)

    def _confirm_pending(self) -> None:
        self._check_stream_health()
        if self.pending_record is None:
            logger.info("No pending point or ROI to save.")
            return
        record = {key: value for key, value in self.pending_record.items() if key != "action"}
        try:
            if self.pending_record["action"] == "point":
                step_name = self.store.save_point(record)
            else:
                step_name = self.store.save_template(record, self.pending_roi)
        except Exception:
            logger.exception("Failed to save marker record")
            return
        logger.info("Saved %s", step_name)
        self._clear_selection()

    def _draw_overlay(self, disp):
        line1 = f"Frame: ({self.current_frame_x},{self.current_frame_y})"
        line2 = f"Device: ({self.current_dev_x},{self.current_dev_y})"
        cv2.putText(disp, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)
        cv2.putText(disp, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)

        if self.show_help:
            help_y = disp.shape[0] - 34
            cv2.putText(disp, "Click: point | Drag: ROI | Right/S: save", (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 220, 60), 1)
            cv2.putText(disp, "Ctrl+Z: undo | Q: quit | H: help", (10, help_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 220, 60), 1)
