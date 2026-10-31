import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import app
from device.controller import DeviceController, KEYCODE_EVENT_FORMAT, TOUCH_EVENT_FORMAT


def parse(*args):
    with patch("sys.argv", ["app.py", "cmd", *args]):
        return app.parse_args()


class CommandParserTests(unittest.TestCase):
    def test_defaults_and_dispatch(self):
        with patch("sys.argv", ["app.py", "cmd", "back"]), \
             patch.object(app, "run_cmd_mode", return_value=7) as run:
            self.assertEqual(app.main(), 7)
        args = run.call_args.args[0]
        self.assertEqual(args.home, app.PROJECT_HOME)
        self.assertEqual((args.mode, args.command, args.config, args.server_jar),
                         ("cmd", "back", "config.yaml", None))
        self.assertEqual((args.adb, args.serial, args.host, args.port, args.socket_name),
                         ("adb", None, "127.0.0.1", 27183, "scrcpy"))
        self.assertEqual(parse("swipe", "0", "1", "2", "3").duration, 300)

    def test_custom_connection_and_command_parameters(self):
        args = parse("--config", "other.yaml", "--server-jar", "other.jar", "--adb", "adb2",
                     "--serial", "serial", "--host", "host", "--port", "42",
                     "--socket-name", "socket", "swipe", "1", "2", "3", "4", "--duration", "45")
        self.assertEqual((args.config, args.server_jar, args.adb, args.serial, args.host, args.port,
                          args.socket_name), ("other.yaml", "other.jar", "adb2", "serial", "host", 42, "socket"))
        self.assertEqual((args.x1, args.y1, args.x2, args.y2, args.duration), (1, 2, 3, 4, 45))

    def test_invalid_args_fail_before_connecting(self):
        invalid = [(), ("unknown",), ("back", "1"), ("click",), ("click", "1"),
                   ("click", "-1", "0"), ("click", "0", "-1"), ("click", "1.5", "0"),
                   ("click", "x", "0"), ("click", "1", "2", "3"),
                   ("swipe", "1", "2", "3"), ("swipe", "-1", "0", "0", "0"),
                   ("swipe", "0", "-1", "0", "0"), ("swipe", "0", "0", "-1", "0"),
                   ("swipe", "0", "0", "0", "-1"),
                   ("swipe", "0", "0", "0", "0", "--duration", "0"),
                   ("swipe", "0", "0", "0", "0", "--duration", "-1"),
                   ("swipe", "0", "0", "0", "0", "--duration", "1.5"),
                   ("back", "--duration", "5"), ("--port", "bad", "back"),
                   ("back", "--adb", "adb2")]
        with patch.object(app, "ScrcpySession") as session, patch.object(app, "load_server_jar") as load:
            for argv in invalid:
                with self.subTest(argv=argv), patch("sys.argv", ["app.py", "cmd", *argv]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    app.main()
                self.assertEqual(exc.exception.code, 2)
            session.assert_not_called()
            load.assert_not_called()

    def test_help_shows_hierarchy_and_duration(self):
        for argv, expected in [(("--help",), ("--config", "--server-jar", "--adb", "--serial", "--host",
                                                  "--port", "--socket-name", "{back,click,swipe}")),
                               (("click", "--help"), ("X Y",)),
                               (("swipe", "--help"), ("X1 Y1 X2 Y2", "--duration MILLISECONDS", "300"))]:
            output = io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exc:
                parse(*argv)
            self.assertEqual(exc.exception.code, 0)
            for text in expected:
                self.assertIn(text, output.getvalue())


class CommandExecutionTests(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        for name in ("load_server_jar", "load_tap_jitter", "ScrcpySession"):
            patcher = patch.object(app, name)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            setattr(self, name, mocked)
            self.calls.attach_mock(mocked, name)
        sleep_patcher = patch.object(app.time, "sleep")
        self.sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)
        self.calls.attach_mock(self.sleep, "sleep")
        self.load_server_jar.return_value = Path("server.jar")
        self.load_tap_jitter.return_value = 0
        self.session = self.ScrcpySession.return_value
        self.controller = self.session.controller
        self.controller.display_width = 100
        self.controller.display_height = 200

    def test_order_and_parameters_for_each_command(self):
        for command, expected in [(("back",), call.ScrcpySession().controller.back()),
                                  (("click", "99", "199"), call.ScrcpySession().controller.tap(99, 199)),
                                  (("swipe", "0", "0", "99", "199"),
                                   call.ScrcpySession().controller.swipe(0, 0, 99, 199, duration_ms=300)),
                                  (("swipe", "1", "2", "3", "4", "--duration", "1"),
                                   call.ScrcpySession().controller.swipe(1, 2, 3, 4, duration_ms=1))]:
            with self.subTest(command=command):
                self.calls.reset_mock()
                args = parse("--config", "other.yaml", "--server-jar", "other.jar", "--adb", "adb2",
                             "--serial", "s", "--host", "h", "--port", "42", "--socket-name", "sock",
                             *command)
                with patch.object(app.logger, "warning") as warning:
                    self.assertEqual(app.run_cmd_mode(args), 0)
                warning.assert_called_once_with("Command sent; no device ACK")
                self.assertEqual(self.calls.mock_calls, [
                    call.load_server_jar(app.PROJECT_HOME / "other.yaml", "other.jar", home=app.PROJECT_HOME),
                    call.load_tap_jitter(app.PROJECT_HOME / "other.yaml", home=app.PROJECT_HOME),
                    call.ScrcpySession(server_jar=Path("server.jar"), adb="adb2", serial="s",
                                       host="h", port=42, socket_name="sock", tap_jitter_pixels=0,
                                       screen_on=True, stay_awake=False),
                    call.ScrcpySession().start(), call.ScrcpySession().check_stream_health(),
                    expected, call.sleep(0.2), call.ScrcpySession().check_stream_health(),
                    call.ScrcpySession().stop(),
                ])

    def test_out_of_bounds_never_calls_controller(self):
        for command in [("click", "100", "0"), ("click", "0", "200"),
                        ("swipe", "100", "0", "1", "1"), ("swipe", "0", "200", "1", "1"),
                        ("swipe", "0", "0", "100", "1"), ("swipe", "0", "0", "1", "200")]:
            with self.subTest(command=command):
                self.calls.reset_mock()
                with patch.object(app.logger, "exception") as log:
                    self.assertEqual(app.run_cmd_mode(parse(*command)), 1)
                log.assert_called_once_with("cmd mode failed")
                self.session.start.assert_called_once_with()
                self.session.stop.assert_called_once_with()
                self.controller.tap.assert_not_called()
                self.controller.swipe.assert_not_called()
                self.sleep.assert_not_called()

    def test_errors_and_interrupts_always_cleanup(self):
        for source in (self.session.start, self.controller.back, self.sleep, self.session.stop):
            for error, status in ((OSError("failed"), 1), (KeyboardInterrupt(), 130)):
                with self.subTest(source=source, error=error):
                    self.calls.reset_mock()
                    source.side_effect = error
                    with patch.object(app.logger, "exception") as log:
                        self.assertEqual(app.run_cmd_mode(parse("back")), status)
                    self.session.stop.assert_called_once_with()
                    if isinstance(error, OSError):
                        log.assert_called_once_with("failed to stop session" if source is self.session.stop
                                                   else "cmd mode failed")
                    else:
                        log.assert_not_called()
                    if source in (self.session.start, self.controller.back):
                        self.sleep.assert_not_called()
                    source.side_effect = None

    def test_interrupt_status_survives_stop_error(self):
        self.controller.back.side_effect = KeyboardInterrupt()
        self.session.stop.side_effect = OSError("stop")
        with patch.object(app.logger, "exception") as log:
            self.assertEqual(app.run_cmd_mode(parse("back")), 130)
        log.assert_called_once_with("failed to stop session")

    def test_video_failure_before_or_after_command_is_reported(self):
        for side_effect, sent in (([RuntimeError("stream failed")], False),
                                  ([None, RuntimeError("stream failed")], True)):
            with self.subTest(sent=sent):
                self.calls.reset_mock()
                self.session.check_stream_health.side_effect = side_effect
                with patch.object(app.logger, "exception"):
                    self.assertEqual(app.run_cmd_mode(parse("back")), 1)
                self.assertEqual(self.controller.back.call_count, int(sent))
                self.session.stop.assert_called_once_with()
                self.session.check_stream_health.side_effect = None

    def test_failure_before_session_does_not_stop(self):
        for error, status in ((OSError("config"), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(error=error):
                self.calls.reset_mock()
                self.load_server_jar.side_effect = error
                with patch.object(app.logger, "exception"):
                    self.assertEqual(app.run_cmd_mode(parse("back")), status)
                self.ScrcpySession.assert_not_called()
                self.session.stop.assert_not_called()

    def test_session_constructor_failure_does_not_stop(self):
        self.ScrcpySession.side_effect = OSError("constructor")
        with patch.object(app.logger, "exception") as log:
            self.assertEqual(app.run_cmd_mode(parse("back")), 1)
        log.assert_called_once_with("cmd mode failed")
        self.session.start.assert_not_called()
        self.session.stop.assert_not_called()

    def test_real_controller_prevalidates_swipe_and_sends_valid_packets(self):
        conn = Mock()
        self.session.controller = DeviceController(conn, 100, 200)
        with patch.object(app.logger, "exception"), patch.object(app.logger, "warning"):
            self.assertEqual(app.run_cmd_mode(parse("swipe", "0", "0", "100", "0")), 1)
            conn.sendall.assert_not_called()
            self.assertEqual(app.run_cmd_mode(parse("click", "99", "199")), 0)
        packets = [TOUCH_EVENT_FORMAT.unpack(args.args[0]) for args in conn.sendall.call_args_list]
        self.assertEqual([(p[1], p[3], p[4]) for p in packets], [(0, 99, 199), (1, 99, 199)])
        conn.reset_mock()
        with patch.object(app.logger, "warning"), patch("device.controller.time.monotonic", side_effect=[0, 1, 1]):
            self.assertEqual(app.run_cmd_mode(parse("swipe", "0", "0", "99", "199", "--duration", "1")), 0)
        packets = [TOUCH_EVENT_FORMAT.unpack(args.args[0]) for args in conn.sendall.call_args_list]
        self.assertEqual([(p[1], p[3], p[4]) for p in packets], [(0, 0, 0), (2, 50, 100), (1, 99, 199)])
        conn.reset_mock()
        with patch.object(app.logger, "warning"):
            self.assertEqual(app.run_cmd_mode(parse("back")), 0)
        data = conn.sendall.call_args.args[0]
        self.assertEqual([KEYCODE_EVENT_FORMAT.unpack_from(data, offset)[1:3]
                          for offset in (0, KEYCODE_EVENT_FORMAT.size)], [(0, 4), (1, 4)])


if __name__ == "__main__":
    unittest.main()
