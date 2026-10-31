import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import cv2
import yaml

from calibration.marker_ui import MarkerUI
from calibration.records import MarkRecordStore
from device.coordinates import frame_to_device, window_to_frame
from device.scrcpy_session import ScrcpySession


class CoordinateTests(unittest.TestCase):
    def test_ui_default_home_is_fixed_when_cwd_changes(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            home = Path(directory).resolve()
            original_cwd = Path.cwd()
            try:
                os.chdir(other)
                with patch("calibration.records.PROJECT_HOME", home):
                    ui = MarkerUI(MagicMock(), Path("other-records"))
                os.chdir(home.parent)
                sizes = {"frame_size": {"width": 10, "height": 10},
                         "device_size": {"width": 10, "height": 10}}
                step = ui.store.save_template(sizes, np.zeros((2, 2, 3), dtype=np.uint8))
                data = yaml.safe_load(ui.store.records_file.read_text(encoding="utf-8"))
                image_path = Path(data["steps"][step]["image"])
                self.assertEqual(ui.store.output_dir, home / "other-records")
                self.assertEqual(ui.store.records_file.parent, home / "other-records")
                self.assertEqual(image_path.parent, Path("captures/templates"))
                self.assertEqual(data["steps"][step]["image"], f"captures/templates/{image_path.name}")
                self.assertFalse((ui.store.output_dir / "templates").exists())
                self.assertTrue((home / image_path).is_file())
                self.assertEqual(ui.store.undo(), step)
                self.assertFalse((home / image_path).exists())
            finally:
                os.chdir(original_cwd)

    def test_resized_window_coordinates_and_bounds(self):
        self.assertEqual(window_to_frame(330, 660, (660, 1320), (1000, 2000)), (500, 1000))
        self.assertEqual(window_to_frame(659, 1319, (660, 1320), (1000, 2000)), (998, 1998))
        self.assertIsNone(window_to_frame(660, 1320, (660, 1320), (1000, 2000)))
        self.assertEqual(frame_to_device(999, 1999, (1000, 2000), (500, 1000)), (499, 999))

    def test_ui_uses_highgui_image_coordinates_after_window_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            session = MagicMock()
            ui = MarkerUI(session, Path("captures"), home=Path(directory))
            ui._displayed_frame = np.zeros((2000, 1000, 3), dtype=np.uint8)
            ui._display_w, ui._display_h = 330, 660
            with patch("calibration.marker_ui.cv2.getWindowImageRect", return_value=(10, 20, 660, 1320)):
                self.assertEqual(ui._map_window_to_frame(165, 330), (500, 1000))
                self.assertIsNone(ui._map_window_to_frame(330, 660))
            session.get_latest_size.assert_not_called()


class MarkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.session = MagicMock()
        self.session.display_width = 100
        self.session.display_height = 200
        self.home = Path(directory.name).resolve()
        self.ui = MarkerUI(self.session, Path("captures"), home=self.home)
        self.frame = np.arange(200 * 100 * 3, dtype=np.uint8).reshape(200, 100, 3)
        self.ui._displayed_frame = self.frame
        self.ui._display_w, self.ui._display_h = 50, 100
        self.ui._displayed_device_size = (100, 200)

    def mouse(self, event, x, y):
        self.ui._on_mouse_event(event, x, y, 0, None)

    def point(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, 25, 50)
        self.mouse(cv2.EVENT_LBUTTONUP, 25, 50)

    def test_point_maps_against_displayed_not_latest_frame(self):
        self.session.get_latest_size.return_value = (200, 400)
        self.point()
        self.assertEqual(self.ui.pending_record["frame"], {"x": 50, "y": 100})
        self.assertEqual(self.ui.pending_record["device"], {"x": 50, "y": 100})
        self.assertEqual(self.ui.pending_record["frame_size"], {"width": 100, "height": 200})
        self.assertEqual(self.ui.pending_record["device_size"], {"width": 100, "height": 200})
        self.session.get_latest_size.assert_not_called()
        self.session.map_frame_to_device.assert_not_called()

    def test_rendered_snapshot_supplies_point_sizes(self):
        self.session.get_frame_snapshot.return_value = SimpleNamespace(frame=self.frame, device_size=(200, 400))
        with patch("calibration.marker_ui.cv2.imshow"), \
                patch("calibration.marker_ui.cv2.waitKey", return_value=ord("q")), \
                patch.object(self.ui, "_setup_window"), \
                patch("calibration.marker_ui.cv2.destroyWindow"):
            self.ui.run()
        self.point()
        self.assertEqual(self.ui.pending_record["device"], {"x": 100, "y": 200})
        self.ui._confirm_pending()
        data = yaml.safe_load(self.ui.store.records_file.read_text(encoding="utf-8"))
        record = data["steps"]["step_001"]
        self.assertEqual(data["meta"], {"frame_size": {"width": 100, "height": 200},
                                        "device_size": {"width": 200, "height": 400}})
        self.assertTrue(data["meta"].keys().isdisjoint(record))

    def test_roi_crops_displayed_frame_not_newer_frame(self):
        self.session.get_latest_frame.return_value = np.zeros_like(self.frame)
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        self.mouse(cv2.EVENT_LBUTTONUP, 20, 40)
        np.testing.assert_array_equal(self.ui.pending_roi, self.frame[20:80, 10:40])
        self.assertEqual(self.ui.pending_frame_rect, (10, 20, 40, 80))
        self.session.get_latest_frame.assert_not_called()

    def test_pending_selection_freezes_rendered_frame(self):
        self.point()
        self.session.get_frame_snapshot.return_value = SimpleNamespace(frame=np.zeros_like(self.frame), device_size=(100, 200))
        with patch("calibration.marker_ui.cv2.imshow") as imshow, \
                patch("calibration.marker_ui.cv2.waitKey", return_value=ord("q")), \
                patch.object(self.ui, "_setup_window"), \
                patch("calibration.marker_ui.cv2.destroyWindow"), \
                patch.object(self.ui, "_draw_overlay"):
            self.ui.run()
        np.testing.assert_array_equal(
            imshow.call_args.args[1], cv2.resize(self.frame, (50, 100))
        )

    def test_drag_uses_initial_frame_and_device_size(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        self.ui._displayed_frame = np.zeros((400, 200, 3), dtype=np.uint8)
        self.ui._displayed_device_size = (200, 400)
        self.mouse(cv2.EVENT_LBUTTONUP, 20, 40)
        np.testing.assert_array_equal(self.ui.pending_roi, self.frame[20:80, 10:40])
        self.assertEqual(self.ui.pending_record["device_rect"],
                         {"x1": 10, "y1": 20, "x2": 40, "y2": 80})
        self.assertEqual(self.ui.pending_record["frame_size"], {"width": 100, "height": 200})
        self.assertEqual(self.ui.pending_record["device_size"], {"width": 100, "height": 200})
        self.assertEqual(self.ui.pending_record["frame_center"], {"x": 24, "y": 49})
        self.assertEqual(self.ui.pending_record["device_center"], {"x": 24, "y": 49})

    def test_roi_center_uses_half_open_crop_and_frozen_device_size(self):
        self.ui._displayed_device_size = (200, 400)
        self.mouse(cv2.EVENT_LBUTTONDOWN, 20, 40)
        self.ui._displayed_device_size = (1000, 1000)
        self.mouse(cv2.EVENT_LBUTTONUP, 5, 10)
        record = self.ui.pending_record
        self.assertEqual(record["frame_center"], {"x": 24, "y": 49})
        self.assertEqual(record["device_center"], {"x": 48, "y": 98})
        self.assertEqual(record["device_size"], {"width": 200, "height": 400})
        self.ui._confirm_pending()
        data = yaml.safe_load(self.ui.store.records_file.read_text(encoding="utf-8"))
        self.assertEqual(data["steps"]["step_001"]["frame_center"], record["frame_center"])
        self.assertEqual(data["meta"], {key: record[key] for key in ("frame_size", "device_size")})
        self.assertTrue(data["meta"].keys().isdisjoint(data["steps"]["step_001"]))

    def test_resolution_change_preserves_pending_roi_and_existing_data(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, 25, 50)
        self.ui._displayed_frame = np.zeros((400, 200, 3), dtype=np.uint8)
        self.ui._displayed_device_size = (400, 800)
        self.mouse(cv2.EVENT_LBUTTONUP, 25, 50)
        self.ui._confirm_pending()
        path = self.ui.store.records_file
        original = path.read_bytes()
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        self.mouse(cv2.EVENT_LBUTTONUP, 20, 40)
        pending = self.ui.pending_record
        roi = self.ui.pending_roi
        with self.assertLogs("marker", level="ERROR"):
            self.ui._confirm_pending()
        self.assertIs(self.ui.pending_record, pending)
        self.assertIs(self.ui.pending_roi, roi)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.ui.store.templates_dir.iterdir()), [])
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        first_sizes = {"frame_size": {"width": 100, "height": 200},
                       "device_size": {"width": 100, "height": 200}}
        next_sizes = {"frame_size": {"width": 200, "height": 400},
                      "device_size": {"width": 400, "height": 800}}
        self.assertEqual(data["meta"], first_sizes)
        self.assertTrue(first_sizes.keys().isdisjoint(data["steps"]["step_001"]))
        self.assertEqual({key: pending[key] for key in next_sizes}, next_sizes)
        self.assertEqual(self.ui.store.undo(), "step_001")
        self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8")),
                         {"meta": first_sizes, "steps": {}})
        with self.assertLogs("marker", level="ERROR"):
            self.ui._confirm_pending()
        self.assertIs(self.ui.pending_record, pending)
        self.assertIs(self.ui.pending_roi, roi)
        self.assertEqual(list(self.ui.store.templates_dir.iterdir()), [])

        self.ui._displayed_frame = self.frame
        self.ui._displayed_device_size = (100, 200)
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        self.mouse(cv2.EVENT_LBUTTONUP, 20, 40)
        self.ui._confirm_pending()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(data["meta"], first_sizes)
        self.assertTrue(first_sizes.keys().isdisjoint(data["steps"]["step_001"]))
        image = self.home / data["steps"]["step_001"]["image"]
        np.testing.assert_array_equal(cv2.imread(str(image)), self.frame[20:80, 10:40])

    def test_template_save_failure_cleans_image_and_retries_pending(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        self.mouse(cv2.EVENT_LBUTTONUP, 20, 40)
        pending = self.ui.pending_record
        roi = self.ui.pending_roi
        original = self.ui.store.records_file.read_bytes()
        with patch("calibration.records.os.replace", side_effect=OSError("disk full")), \
                self.assertLogs("marker", level="ERROR"):
            self.ui._confirm_pending()
        self.assertIs(self.ui.pending_record, pending)
        self.assertIs(self.ui.pending_roi, roi)
        self.assertIs(self.ui._selection_frame, self.frame)
        self.assertEqual(self.ui.store.records_file.read_bytes(), original)
        self.assertEqual(list(self.ui.store.templates_dir.iterdir()), [])
        self.ui._confirm_pending()
        self.assertIsNone(self.ui.pending_record)
        data = yaml.safe_load(self.ui.store.records_file.read_text(encoding="utf-8"))
        self.assertEqual(data["meta"], {key: pending[key] for key in ("frame_size", "device_size")})
        image = self.home / data["steps"]["step_001"]["image"]
        np.testing.assert_array_equal(cv2.imread(str(image)), roi)

    def test_odd_and_one_pixel_roi_centers(self):
        self.ui._selection_frame = self.frame
        self.ui._selection_device_size = (100, 200)
        for start, end, center in [((10, 20), (13, 25), (11, 22)), ((10, 20), (11, 21), (10, 20))]:
            with self.subTest(start=start, end=end):
                self.ui._prepare_roi(start, end)
                self.assertEqual(self.ui.pending_record["frame_center"], {"x": center[0], "y": center[1]})

    def test_disconnect_while_waiting_for_frame_is_reported(self):
        self.session.get_frame_snapshot.return_value = None

        def disconnected(_):
            self.session.check_stream_health.side_effect = RuntimeError("video stream failed")
            return ord("q")

        with patch.object(self.ui, "_setup_window"), \
                patch("calibration.marker_ui.cv2.destroyWindow"), \
                patch("calibration.marker_ui.cv2.waitKey", side_effect=disconnected):
            with self.assertRaisesRegex(RuntimeError, "video stream failed"):
                self.ui.run()

    def test_frame_read_failure_clears_pending_selection(self):
        self.point()
        self.session.get_frame_snapshot.side_effect = RuntimeError("video stream failed")
        with patch.object(self.ui, "_setup_window"), \
                patch("calibration.marker_ui.cv2.destroyWindow"):
            with self.assertRaisesRegex(RuntimeError, "video stream failed"):
                self.ui.run()
        self.assertIsNone(self.ui.pending_record)
        self.assertIsNone(self.ui._selection_frame)

    def test_save_failure_keeps_pending_selection_for_retry(self):
        self.point()
        record = self.ui.pending_record
        with patch.object(self.ui.store, "save_point", side_effect=OSError("disk full")), \
                self.assertLogs("marker", level="ERROR"):
            self.ui._confirm_pending()
        self.assertIs(self.ui.pending_record, record)
        self.assertIs(self.ui._selection_frame, self.frame)
        self.ui._confirm_pending()
        self.assertIsNone(self.ui.pending_record)
        self.assertIsNone(self.ui._selection_frame)

    def test_image_failure_in_mouse_callback_is_recoverable(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        self.mouse(cv2.EVENT_LBUTTONUP, 20, 40)
        with patch.object(self.ui.store, "save_template", side_effect=cv2.error("image failed")), \
                self.assertLogs("marker", level="ERROR"):
            self.mouse(cv2.EVENT_RBUTTONUP, 20, 40)
        self.assertIsNotNone(self.ui.pending_record)
        self.assertIsNone(self.ui._callback_error)

    def test_undo_failure_keeps_ui_running(self):
        with patch.object(self.ui.store, "undo", side_effect=OSError("disk full")), \
                self.assertLogs("marker", level="ERROR"):
            self.assertFalse(self.ui._handle_key(26))

    def test_stream_failure_blocks_save_and_clears_pending_selection(self):
        self.point()
        self.session.check_stream_health.side_effect = RuntimeError("video stream failed")
        with patch.object(self.ui.store, "save_point") as save:
            self.mouse(cv2.EVENT_RBUTTONUP, 25, 50)
            with self.assertRaisesRegex(RuntimeError, "video stream failed"):
                self.ui._check_stream_health()
        save.assert_not_called()
        self.assertIsNone(self.ui.pending_record)
        self.assertIsNone(self.ui._selection_frame)
        self.assertFalse(self.ui.drawing)

    def test_run_surfaces_callback_failure_even_if_quit_is_pressed(self):
        self.point()
        self.session.get_frame_snapshot.return_value = SimpleNamespace(frame=self.frame, device_size=(100, 200))

        def disconnected(_):
            self.session.check_stream_health.side_effect = RuntimeError("video stream failed")
            self.mouse(cv2.EVENT_RBUTTONUP, 25, 50)
            return ord("q")

        with patch.object(self.ui, "_setup_window"), \
                patch("calibration.marker_ui.cv2.imshow"), \
                patch("calibration.marker_ui.cv2.destroyWindow") as destroy, \
                patch("calibration.marker_ui.cv2.waitKey", side_effect=disconnected):
            with self.assertRaisesRegex(RuntimeError, "video stream failed"):
                self.ui.run()
        destroy.assert_called_once_with(self.ui.window_name)

    def test_invalid_new_selection_clears_old_pending_record(self):
        self.point()
        self.mouse(cv2.EVENT_LBUTTONDOWN, 5, 10)
        with self.assertLogs("marker", level="WARNING"):
            self.mouse(cv2.EVENT_LBUTTONUP, 50, 100)
        self.assertIsNone(self.ui.pending_record)
        self.assertIsNone(self.ui._selection_frame)


class RecordTests(unittest.TestCase):
    def test_save_and_undo_point_and_template(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MarkRecordStore(Path("captures"), home=Path(directory))
            sizes = {"frame_size": {"width": 100, "height": 200},
                     "device_size": {"width": 100, "height": 200}}
            point = {**sizes, "frame": {"x": 10, "y": 20}, "device": {"x": 10, "y": 20}}
            template = {**sizes, "frame_rect": {"x1": 1, "y1": 1, "x2": 11, "y2": 11},
                        "device_rect": {"x1": 1, "y1": 1, "x2": 11, "y2": 11},
                        "frame_center": {"x": 5, "y": 5}, "device_center": {"x": 5, "y": 5}}
            self.assertEqual(store.save_point(point), "step_001")
            roi = np.zeros((10, 10, 3), dtype=np.uint8)
            self.assertEqual(store.save_template(template, roi), "step_002")
            data = yaml.safe_load(store.records_file.read_text(encoding="utf-8"))
            self.assertEqual(list(data["steps"]), ["step_001", "step_002"])
            image = Path(directory) / data["steps"]["step_002"]["image"]
            self.assertTrue(image.is_file())
            self.assertEqual(store.undo(), "step_002")
            self.assertFalse(image.exists())
            self.assertEqual(store.undo(), "step_001")
            self.assertEqual(store.save_point(point), "step_001")

    def test_invalid_records_are_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MarkRecordStore(Path("captures"), home=Path(directory))
            original = "steps: invalid\n"
            store.records_file.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "steps"):
                store.save_point({"frame_size": {"width": 100, "height": 200},
                                  "device_size": {"width": 100, "height": 200},
                                  "frame": {"x": 1, "y": 1}, "device": {"x": 1, "y": 1}})
            self.assertEqual(store.records_file.read_text(encoding="utf-8"), original)


class SessionTests(unittest.TestCase):
    def test_stream_failure_prevents_stale_frame_access(self):
        session = ScrcpySession(Path("unused"))
        session._stream_error = ConnectionError("video closed")
        with self.assertRaisesRegex(RuntimeError, "video stream failed"):
            session.get_latest_frame()

    def test_stop_removes_reverse_mapping(self):
        session = ScrcpySession(Path("unused"))
        session._reverse_active = True
        with patch.object(session, "_run_adb") as run_adb:
            session.stop()
        run_adb.assert_called_once_with("reverse", "--remove", "localabstract:scrcpy")
        self.assertFalse(session._reverse_active)

    def test_stopped_session_prevents_stale_frame_access(self):
        session = ScrcpySession(Path("unused"))
        session._latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        session.stop()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            session.get_latest_frame()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            session.get_latest_size()
        self.assertIsNone(session._latest_frame)

    def test_decode_eof_is_exposed_and_clears_stale_frame(self):
        session = ScrcpySession(Path("unused"))
        session._video_conn = MagicMock()
        session._latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch("device.scrcpy_session.av.codec.CodecContext.create"), \
                patch.object(session, "_read_exact", return_value=b""), \
                self.assertLogs("scrcpy", level="ERROR"):
            session._start_decode_thread()
            session._decode_thread.join(timeout=2)
        self.assertTrue(session._stream_ended.is_set())
        self.assertIsNone(session._latest_frame)
        with self.assertRaisesRegex(RuntimeError, "video stream failed"):
            session.get_latest_frame()

    def test_start_failure_cleans_listener_and_reverse(self):
        for failure in ("install_server", "_accept", "_read_stream_metadata", "_start_decode_thread"):
            with self.subTest(failure=failure):
                session = ScrcpySession(Path("unused"))
                listener = MagicMock()
                video, control = MagicMock(), MagicMock()

                def open_listener():
                    session._listener = listener

                with patch.object(session, "_open_listener", side_effect=open_listener), \
                        patch.object(session, "_run_adb") as adb, \
                        patch("device.scrcpy_session.subprocess.run"), \
                        patch.object(session, "install_server"), \
                        patch.object(session, "start_server_process"), \
                        patch.object(session, "_accept", side_effect=[video, control]), \
                        patch.object(session, "_read_stream_metadata"), \
                        patch.object(session, "_start_decode_thread"), \
                        patch.object(session, failure, side_effect=RuntimeError("startup failed")):
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        session.start()
                    session.stop()
                listener.close.assert_called_once()
                self.assertIsNone(session._listener)
                self.assertIsNone(session._video_conn)
                self.assertIsNone(session._control_conn)
                self.assertFalse(session._reverse_active)
                self.assertEqual(adb.call_count, 2)
                adb.assert_called_with("reverse", "--remove", "localabstract:scrcpy")

    def test_reverse_is_removed_even_if_server_shutdown_fails(self):
        session = ScrcpySession(Path("unused"))
        session._reverse_active = True
        session._server_proc = MagicMock()
        session._server_proc.wait.side_effect = OSError("wait failed")
        with patch.object(session, "_run_adb") as adb:
            with self.assertRaisesRegex(OSError, "wait failed"):
                session.stop()
        adb.assert_called_once_with("reverse", "--remove", "localabstract:scrcpy")

    def test_failed_reverse_removal_can_be_retried(self):
        session = ScrcpySession(Path("unused"))
        session._reverse_active = True
        with patch.object(session, "_run_adb", side_effect=[RuntimeError("adb failed"), None]) as adb:
            with self.assertRaisesRegex(RuntimeError, "adb failed"):
                session.stop()
            self.assertTrue(session._reverse_active)
            session.stop()
        self.assertEqual(adb.call_count, 2)
        self.assertFalse(session._reverse_active)


if __name__ == "__main__":
    unittest.main()
