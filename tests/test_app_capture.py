import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app


def parse(*args):
    with patch("sys.argv", ["app.py", "capture", *args]):
        return app.parse_args()


class CaptureParserTests(unittest.TestCase):
    def test_defaults_and_dispatch(self):
        with patch("sys.argv", ["app.py", "capture"]), \
                patch.object(app, "run_capture_mode", return_value=7) as capture:
            self.assertEqual(app.main(), 7)
        args = capture.call_args.args[0]
        self.assertEqual(args.home, app.PROJECT_HOME)
        self.assertEqual((args.mode, args.output, args.config, args.server_jar),
                         ("capture", None, "config.yaml", None))
        self.assertEqual((args.adb, args.serial, args.host, args.port, args.socket_name),
                         ("adb", None, "127.0.0.1", 27183, "scrcpy"))

    def test_custom_connection_output_and_screen_flags(self):
        args = parse("--config", "other.yaml", "--server-jar", "other.jar", "--adb", "adb2",
                     "--serial", "serial", "--host", "host", "--port", "42",
                     "--socket-name", "socket", "--output", "shots/one.png", "--screen-off")
        self.assertEqual((args.config, args.server_jar, args.adb, args.serial, args.host,
                          args.port, args.socket_name, args.output, args.screen_on),
                         ("other.yaml", "other.jar", "adb2", "serial", "host", 42,
                          "socket", "shots/one.png", False))

    def test_help_includes_output_and_connection_options(self):
        output = io.StringIO()
        with patch("sys.argv", ["app.py", "capture", "--help"]), \
                contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exc:
            app.parse_args()
        self.assertEqual(exc.exception.code, 0)
        for text in ("--output", "--serial", "--screen-off", "frame_<timestamp>.png"):
            self.assertIn(text, output.getvalue())


class CaptureExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.args = SimpleNamespace(
            home=self.home, config="config.yaml", server_jar="server.jar", adb="adb",
            serial="device-1", host="127.0.0.1", port=27183, socket_name="scrcpy",
            output="shots/frame.png", screen_on=None, stay_awake=None,
        )
        self.snapshot = SimpleNamespace(
            frame=np.zeros((3, 4, 3), dtype=np.uint8), frame_size=(4, 3),
            decode_time=time.monotonic(),
        )

    def test_saves_one_fresh_frame_and_cleans_up(self):
        with patch.object(app, "load_server_jar", return_value=Path("server.jar")) as load, \
                patch.object(app, "resolve_screen_options", return_value=(True, False)) as screen, \
                patch.object(app, "ScrcpySession") as make_session, \
                patch.object(app.cv2, "imwrite", return_value=True) as imwrite:
            session = make_session.return_value
            session.get_frame_snapshot.return_value = self.snapshot
            with patch.object(app.logger, "info") as log:
                self.assertEqual(app.run_capture_mode(self.args), 0)

        load.assert_called_once_with(self.home / "config.yaml", "server.jar", home=self.home)
        screen.assert_called_once_with(self.args, self.home)
        make_session.assert_called_once_with(
            server_jar=Path("server.jar"), adb="adb", serial="device-1",
            host="127.0.0.1", port=27183, socket_name="scrcpy",
            screen_on=True, stay_awake=False,
        )
        session.start.assert_called_once_with()
        session.get_frame_snapshot.assert_called_once_with()
        imwrite.assert_called_once_with(str(self.home / "shots/frame.png"), self.snapshot.frame)
        session.stop.assert_called_once_with()
        log.assert_called_once_with("Screenshot saved: %s (%dx%d)",
                                    self.home / "shots/frame.png", 4, 3)
        session.controller.tap.assert_not_called()
        session.controller.swipe.assert_not_called()
        session.controller.back.assert_not_called()

    def test_default_path_is_timestamped_under_project_captures(self):
        self.args.output = None
        with patch.object(app, "load_server_jar", return_value=Path("server.jar")), \
                patch.object(app, "resolve_screen_options", return_value=(True, False)), \
                patch.object(app, "ScrcpySession") as make_session, \
                patch.object(app.cv2, "imwrite", return_value=True) as imwrite, \
                patch.object(app.time, "strftime", return_value="20261108_151403"), \
                patch.object(app.time, "time_ns", return_value=1234):
            make_session.return_value.get_frame_snapshot.return_value = self.snapshot
            self.assertEqual(app.run_capture_mode(self.args), 0)
        self.assertEqual(imwrite.call_args.args[0], str(
            self.home / "captures/automatic/frame_20261108_151403_000001234.png"))

    def test_waits_until_frame_arrives(self):
        with patch.object(app, "load_server_jar"), patch.object(app, "resolve_screen_options", return_value=(True, False)), \
                patch.object(app, "ScrcpySession") as make_session, \
                patch.object(app.cv2, "imwrite", return_value=True) as imwrite, \
                patch.object(app.time, "sleep") as sleep:
            session = make_session.return_value
            session.get_frame_snapshot.side_effect = [None, self.snapshot]
            self.assertEqual(app.run_capture_mode(self.args), 0)
        self.assertEqual(session.get_frame_snapshot.call_count, 2)
        sleep.assert_called_once_with(0.1)
        imwrite.assert_called_once()
        session.stop.assert_called_once_with()

    def test_timeout_and_write_failure_clean_up(self):
        with patch.object(app, "load_server_jar"), patch.object(app, "resolve_screen_options", return_value=(True, False)), \
                patch.object(app, "ScrcpySession") as make_session, \
                patch.object(app.time, "monotonic", side_effect=[0, 15]), \
                patch.object(app.logger, "exception") as log:
            session = make_session.return_value
            self.assertEqual(app.run_capture_mode(self.args), 1)
        session.get_frame_snapshot.assert_not_called()
        session.stop.assert_called_once_with()
        log.assert_called_once_with("capture mode failed")

        with patch.object(app, "load_server_jar"), patch.object(app, "resolve_screen_options", return_value=(True, False)), \
                patch.object(app, "ScrcpySession") as make_session, \
                patch.object(app.cv2, "imwrite", return_value=False), \
                patch.object(app.logger, "exception") as log:
            make_session.return_value.get_frame_snapshot.return_value = self.snapshot
            self.assertEqual(app.run_capture_mode(self.args), 1)
        make_session.return_value.stop.assert_called_once_with()
        log.assert_called_once_with("capture mode failed")

    def test_interrupt_still_stops_session(self):
        with patch.object(app, "load_server_jar"), patch.object(app, "resolve_screen_options", return_value=(True, False)), \
                patch.object(app, "ScrcpySession") as make_session:
            session = make_session.return_value
            session.start.side_effect = KeyboardInterrupt()
            self.assertEqual(app.run_capture_mode(self.args), 130)
        session.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
