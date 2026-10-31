import contextlib
import io
import tempfile
import threading
import unittest
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

import cv2
import numpy as np

import app
from automation.runtime import Runtime

RealPreviewWindow = app.PreviewWindow


class AutomationAppTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(task_list="tasks.yaml", config="config.yaml", server_jar="server.jar",
                                    adb="custom-adb", serial="device", host="host", port=1234,
                                    socket_name="socket", screen_on=None, stay_awake=None)
        self.plan = SimpleNamespace(resources={"resource": 1}, scenes={"scene": 2}, tasks=[], loop=None)
        self.calls = Mock()
        for name in ("load_plan", "Detector", "load_server_jar", "load_tap_jitter",
                     "ScrcpySession", "PreviewWindow", "Runtime", "run_list", "run_loop"):
            patcher = patch.object(app, name)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            setattr(self, name, mocked)
            self.calls.attach_mock(mocked, name)
        screen_patcher = patch.object(app, "load_scrcpy_bool",
                                      side_effect=lambda key, default, path, home=None: default)
        self.load_scrcpy_bool = screen_patcher.start()
        self.addCleanup(screen_patcher.stop)
        self.load_plan.return_value = self.plan
        self.load_server_jar.return_value = Path("server.jar")
        self.load_tap_jitter.return_value = 0
        self.session = self.ScrcpySession.return_value
        self.runtime = self.Runtime.return_value
        self.run_list.return_value = True
        self.PreviewWindow.return_value.start.return_value = False

    def test_success_assembles_and_validates_first_frame_in_order(self):
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.assertEqual(self.calls.mock_calls, [
            call.load_plan("tasks.yaml", home=app.PROJECT_HOME),
            call.Detector(self.plan.resources, self.plan.scenes),
            call.load_server_jar(app.PROJECT_HOME / "config.yaml", "server.jar", home=app.PROJECT_HOME),
            call.load_tap_jitter(app.PROJECT_HOME / "config.yaml", home=app.PROJECT_HOME),
            call.ScrcpySession(server_jar=Path("server.jar"), adb="custom-adb", serial="device",
                               host="host", port=1234, socket_name="socket", tap_jitter_pixels=0,
                               screen_on=True, stay_awake=False),
            call.ScrcpySession().start(),
            call.Runtime(self.plan, self.session, self.Detector.return_value, cancel_event=ANY),
            call.Runtime().snapshot(),
            call.run_list(self.runtime, self.plan, cancel_event=ANY),
            call.ScrcpySession().stop(),
        ])
        self.assertIs(self.Runtime.call_args.kwargs["cancel_event"],
                      self.run_list.call_args.kwargs["cancel_event"])
        self.run_loop.assert_not_called()
        self.PreviewWindow.assert_not_called()

    def test_preview_stops_even_when_run_fails_before_finish(self):
        self.plan.scrcpy = {"screen_on": False}
        self.session.start.side_effect = RuntimeError("startup failed")
        self.assertEqual(app.run_automation_mode(self.args), 1)
        self.PreviewWindow.assert_not_called()
        self.session.start.reset_mock()
        self.session.start.side_effect = None
        self.runtime.snapshot.side_effect = RuntimeError("snapshot failed")
        self.assertEqual(app.run_automation_mode(self.args), 1)
        self.PreviewWindow().stop.assert_called_once_with()
        self.runtime.snapshot.side_effect = None

    def test_loop_plan_runs_run_loop_instead_of_run_list(self):
        self.plan.loop = {"task": {"task": "t", "start_scenes": ["m"]}, "count": 2}
        self.run_loop.return_value = True
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.run_loop.assert_called_once_with(self.runtime, self.plan.loop, cancel_event=ANY)
        self.run_list.assert_not_called()

    def test_run_uses_task_list_screen_options(self):
        self.plan.scrcpy = {"screen_on": False, "stay_awake": True}
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.assertFalse(self.ScrcpySession.call_args.kwargs["screen_on"])
        self.assertTrue(self.ScrcpySession.call_args.kwargs["stay_awake"])
        self.PreviewWindow.assert_called_once_with(self.session)
        self.PreviewWindow.reset_mock()
        self.plan.scrcpy = {"screen_on": True, "stay_awake": False}
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.assertTrue(self.ScrcpySession.call_args.kwargs["screen_on"])
        self.assertFalse(self.ScrcpySession.call_args.kwargs["stay_awake"])
        self.PreviewWindow.assert_not_called()

    def test_preview_follows_resolved_options_not_just_task_list(self):
        self.load_scrcpy_bool.side_effect = lambda key, default, path, home=None: False
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.PreviewWindow.assert_called_once_with(self.session)
        self.PreviewWindow.reset_mock()
        self.plan.scrcpy = {"screen_on": True}
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.PreviewWindow.assert_not_called()

    def test_cli_can_enable_screen_off_and_stay_awake_over_task_settings(self):
        self.plan.scrcpy = {"screen_on": True, "stay_awake": False}
        self.args.screen_on = False
        self.args.stay_awake = True
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.assertFalse(self.ScrcpySession.call_args.kwargs["screen_on"])
        self.assertTrue(self.ScrcpySession.call_args.kwargs["stay_awake"])
        self.PreviewWindow.assert_called_once_with(self.session)

    def test_invalid_plan_or_detector_never_connects(self):
        for source in (self.load_plan, self.Detector):
            with self.subTest(source=source):
                self.calls.reset_mock()
                source.side_effect = ValueError("invalid configuration")
                self.assertEqual(app.run_automation_mode(self.args), 1)
                self.ScrcpySession.assert_not_called()
                self.load_server_jar.assert_not_called()
                source.side_effect = None

    def test_failure_after_session_construction_always_stops(self):
        for source in (self.session.start, self.Runtime, self.runtime.snapshot, self.run_list):
            with self.subTest(source=source):
                self.calls.reset_mock()
                source.side_effect = RuntimeError("failure")
                self.assertEqual(app.run_automation_mode(self.args), 1)
                self.session.stop.assert_called_once_with()
                if source is not self.run_list:
                    self.run_list.assert_not_called()
                self.runtime.finish.assert_not_called()
                source.side_effect = None

    def test_failed_list_and_stop_failure_return_error(self):
        self.run_list.return_value = False
        self.assertEqual(app.run_automation_mode(self.args), 1)
        self.session.stop.assert_called_once_with()
        self.session.stop.reset_mock()
        self.run_list.return_value = True
        self.session.stop.side_effect = OSError("stop failed")
        self.assertEqual(app.run_automation_mode(self.args), 1)
        self.session.stop.assert_called_once_with()

    def test_interrupt_stops_session_without_recovery_and_returns_130(self):
        for source in (self.session.start, self.runtime.snapshot, self.run_list, self.session.stop):
            with self.subTest(source=source):
                self.calls.reset_mock()
                source.side_effect = KeyboardInterrupt
                self.assertEqual(app.run_automation_mode(self.args), 130)
                self.session.stop.assert_called_once_with()
                self.runtime.finish.assert_not_called()
                source.side_effect = None

    def test_interrupt_before_session_does_not_connect(self):
        self.load_plan.side_effect = KeyboardInterrupt
        self.assertEqual(app.run_automation_mode(self.args), 130)
        self.ScrcpySession.assert_not_called()

    def test_preview_main_thread_and_automation_worker_run_concurrently(self):
        self.plan.scrcpy = {"screen_on": False}
        main_id = threading.get_ident()
        worker_ready = threading.Event()
        release_worker = threading.Event()
        events = []

        def run(runtime, plan, *, cancel_event):
            events.append(("worker", threading.get_ident()))
            worker_ready.set()
            self.assertTrue(release_worker.wait(3))
            return True

        def update():
            self.assertTrue(worker_ready.wait(3))
            events.append(("preview", threading.get_ident()))
            release_worker.set()
            return False

        self.run_list.side_effect = run
        self.PreviewWindow.return_value.start.return_value = True
        self.PreviewWindow.return_value.update.side_effect = update
        self.assertEqual(app.run_automation_mode(self.args), 0)
        self.assertEqual(events[1], ("preview", main_id))
        self.assertNotEqual(events[0][1], main_id)
        self.session.stop.assert_called_once()

    def test_main_thread_interrupt_cancels_worker_before_stopping_session(self):
        self.plan.scrcpy = {"screen_on": False}
        worker_ready = threading.Event()
        worker_exited = threading.Event()

        def run(runtime, plan, *, cancel_event):
            worker_ready.set()
            if not cancel_event.wait(3):
                raise AssertionError("cancellation not delivered")
            worker_exited.set()
            raise KeyboardInterrupt

        def update():
            self.assertTrue(worker_ready.wait(3))
            raise KeyboardInterrupt

        def stop():
            self.assertTrue(worker_exited.is_set())

        self.run_list.side_effect = run
        self.PreviewWindow.return_value.start.return_value = True
        self.PreviewWindow.return_value.update.side_effect = update
        self.session.stop.side_effect = stop
        self.assertEqual(app.run_automation_mode(self.args), 130)
        self.session.stop.assert_called_once()

    def test_real_preview_runs_all_highgui_calls_on_main_thread(self):
        self.plan.scrcpy = {"screen_on": False}
        main_id = threading.get_ident()
        worker_ready, release_worker = threading.Event(), threading.Event()
        calls = []
        frame = np.zeros((40, 30, 3), dtype=np.uint8)
        self.session.get_frame_snapshot.return_value = SimpleNamespace(frame=frame)

        def run(runtime, plan, *, cancel_event):
            worker_ready.set()
            if not release_worker.wait(3):
                raise RuntimeError("preview did not pump events")
            return True

        def gui(name, result=None):
            def invoke(*args):
                calls.append((name, threading.get_ident()))
                if name == "waitKey":
                    self.assertTrue(worker_ready.wait(3))
                    release_worker.set()
                return result
            return invoke

        self.run_list.side_effect = run
        with patch.object(app, "PreviewWindow", RealPreviewWindow), \
                patch("app.cv2.namedWindow", side_effect=gui("namedWindow")), \
                patch("app.cv2.resizeWindow", side_effect=gui("resizeWindow")), \
                patch("app.cv2.imshow", side_effect=gui("imshow")), \
                patch("app.cv2.waitKey", side_effect=gui("waitKey", -1)), \
                patch("app.cv2.getWindowProperty", side_effect=gui("getWindowProperty", 1)), \
                patch("app.cv2.destroyWindow", side_effect=gui("destroyWindow")):
            self.assertEqual(app.run_automation_mode(self.args), 0)
        self.assertEqual({name for name, _ in calls},
                         {"namedWindow", "resizeWindow", "imshow", "waitKey", "getWindowProperty", "destroyWindow"})
        self.assertTrue(all(thread_id == main_id for _, thread_id in calls))

    def test_real_snapshot_rejects_mismatched_sizes_before_running(self):
        self.Runtime.side_effect = Runtime
        self.plan.resources = {"point": {"frame_size": {"width": 20, "height": 10},
                                         "device_size": {"width": 20, "height": 10}}}
        self.session.get_frame_snapshot.return_value = SimpleNamespace(
            frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_size=(10, 10),
            device_size=(10, 10), frame_id=1, decode_time=time.monotonic())
        self.assertEqual(app.run_automation_mode(self.args), 1)
        self.run_list.assert_not_called()
        self.session.stop.assert_called_once_with()

    def test_run_parser_and_dispatch(self):
        argv = ["app.py", "run", "tasks.yaml", "--config", "config2.yaml", "--server-jar", "x.jar",
                "--adb", "adb2", "--serial", "s", "--host", "h", "--port", "42", "--socket-name", "sock"]
        with patch("sys.argv", argv), patch.object(app, "run_automation_mode", return_value=7) as run:
            self.assertEqual(app.main(), 7)
        args = run.call_args.args[0]
        self.assertEqual(args.home, app.PROJECT_HOME)
        self.assertEqual((args.mode, args.task_list, args.config, args.server_jar),
                         ("run", "tasks.yaml", "config2.yaml", "x.jar"))
        self.assertEqual((args.adb, args.serial, args.host, args.port, args.socket_name),
                         ("adb2", "s", "h", 42, "sock"))

    def test_mark_parser_and_dispatch_unchanged(self):
        with patch("sys.argv", ["app.py", "mark"]), patch.object(app, "run_mark_mode", return_value=9) as mark:
            self.assertEqual(app.main(), 9)
        args = mark.call_args.args[0]
        self.assertEqual(args.home, app.PROJECT_HOME)
        self.assertEqual((args.config, args.adb, args.host, args.port, args.socket_name),
                         ("config.yaml", "adb", "127.0.0.1", 27183, "scrcpy"))
        self.assertEqual((args.output_dir, args.window_scale), ("captures", 0.33))

    def test_mark_window_is_shown_regardless_of_screen_option(self):
        self.args.output_dir = "captures"
        self.args.window_scale = 0.33
        for screen_on in (False, True):
            with self.subTest(screen_on=screen_on), \
                    patch.object(app, "resolve_screen_options", return_value=(screen_on, False)), \
                    patch.object(app, "MarkerUI") as marker:
                self.assertEqual(app.run_mark_mode(self.args), 0)
                marker.return_value.run.assert_called_once()
                self.assertEqual(self.ScrcpySession.call_args.kwargs["screen_on"], screen_on)

    def test_help_shows_compact_usage_and_descriptions(self):
        cases = [
            (["--help"], "usage: app.py [options] {mark,run,cmd} ...", ("commands:", "mark", "run", "cmd")),
            (["run", "--help"], "usage: app.py run [options] TASK_LIST",
             ("positional arguments:", "Path to the task list YAML file", "options:", "--serial")),
            (["mark", "--help"], "usage: app.py mark [options]", ("options:", "--output-dir", "--window-scale")),
            (["cmd", "--help"], "usage: app.py cmd [options] COMMAND [command options]",
             ("options:", "commands:", "back", "click", "swipe")),
            (["cmd", "swipe", "--help"], "usage: app.py cmd swipe [options] X1 Y1 X2 Y2",
             ("positional arguments:", "--duration MILLISECONDS")),
        ]
        for argv, usage, descriptions in cases:
            with self.subTest(argv=argv), patch("sys.argv", ["app.py", *argv]), \
                    contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as exc:
                app.parse_args()
            self.assertEqual(exc.exception.code, 0)
            self.assertEqual(output.getvalue().splitlines()[0], usage)
            for description in descriptions:
                self.assertIn(description, output.getvalue())


class ScreenOptionsTests(unittest.TestCase):
    def args(self, **overrides):
        base = {"config": "config.yaml", "screen_on": None, "stay_awake": None}
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_missing_config_keeps_defaults(self):
        self.assertEqual(app.resolve_screen_options(self.args(), app.PROJECT_HOME / "absent"), (True, False))

    def test_config_values_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.yaml").write_text("scrcpy:\n  screen_on: false\n  stay_awake: false\n",
                                              encoding="utf-8")
            self.assertEqual(app.resolve_screen_options(self.args(), home), (False, False))
            (home / "config.yaml").write_text("scrcpy:\n  stay_awake: false\n", encoding="utf-8")
            self.assertEqual(app.resolve_screen_options(self.args(), home), (True, False))

    def test_task_list_overrides_global_and_cli_overrides_task_list(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.yaml").write_text("scrcpy:\n  screen_on: true\n  stay_awake: false\n",
                                              encoding="utf-8")
            options = {"screen_on": False, "stay_awake": True}
            self.assertEqual(app.resolve_screen_options(self.args(), home, options), (False, True))
            self.assertEqual(app.resolve_screen_options(self.args(), home, {"stay_awake": True}), (True, True))
            self.assertEqual(app.resolve_screen_options(self.args(screen_on=False, stay_awake=True),
                                                        home, {"screen_on": True, "stay_awake": False}),
                             (False, True))

    def test_cli_flags_override_config(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.yaml").write_text("scrcpy:\n  screen_on: true\n  stay_awake: false\n",
                                              encoding="utf-8")
            self.assertEqual(app.resolve_screen_options(self.args(screen_on=False), home), (False, False))
            self.assertEqual(app.resolve_screen_options(self.args(stay_awake=True), home), (True, True))
            self.assertEqual(app.resolve_screen_options(
                self.args(screen_on=False, stay_awake=True), home), (False, True))

    def test_non_boolean_config_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            for key in ("screen_on", "stay_awake"):
                with self.subTest(key=key):
                    (home / "config.yaml").write_text(f"scrcpy:\n  {key}: 1\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"scrcpy.{key}' must be a boolean"):
                        app.resolve_screen_options(self.args(), home)

    def test_obsolete_config_is_rejected_with_migration_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.yaml").write_text("scrcpy:\n  turn_screen_off: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "turn_screen_off.*obsolete.*screen_on"):
                app.resolve_screen_options(self.args(), home)

    def test_cli_flags_are_parsed_in_all_modes(self):
        for mode, extra in (("mark", []), ("run", ["tasks.yaml"]), ("cmd", ["back"])):
            for flags, expected in (([], (None, None)),
                                    (["--screen-off"], (False, None)),
                                    (["--stay-awake"], (None, True)),
                                    (["--screen-off", "--stay-awake"], (False, True))):
                with self.subTest(mode=mode, flags=flags):
                    with patch("sys.argv", ["app.py", mode, *flags, *extra]):
                        args = app.parse_args()
                    self.assertEqual((args.screen_on, args.stay_awake), expected)

    def test_removed_screen_flags_are_rejected(self):
        for mode, extra in (("mark", []), ("run", ["tasks.yaml"]), ("cmd", ["back"])):
            for flag in ("--screen-on", "--no-screen-on", "--no-stay-awake", "--keep-screen"):
                with self.subTest(mode=mode, flag=flag), \
                        patch("sys.argv", ["app.py", mode, flag, *extra]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    app.parse_args()
                self.assertEqual(exc.exception.code, 2)


class PreviewWindowTests(unittest.TestCase):
    def test_renders_and_pumps_events_without_a_frame(self):
        frame = np.zeros((40, 30, 3), dtype=np.uint8)
        snapshot = SimpleNamespace(frame=frame, frame_id=1, decode_time=0.0)
        session = Mock()
        session.get_frame_snapshot.side_effect = [None, snapshot, RuntimeError("stream failed")]
        with patch("app.cv2.namedWindow"), \
                patch("app.cv2.resizeWindow") as resize, \
                patch("app.cv2.resize", return_value=frame) as do_resize, \
                patch("app.cv2.imshow") as imshow, \
                patch("app.cv2.waitKey") as wait, \
                patch("app.cv2.getWindowProperty", return_value=1), \
                patch("app.cv2.destroyWindow") as destroy:
            preview = app.PreviewWindow(session)
            with patch("sys.argv", ["app.py", "mark"]):
                self.assertEqual(preview.window_scale, app.parse_args().window_scale)
            preview.start()
            try:
                self.assertTrue(preview.update())
                self.assertTrue(preview.update())
                self.assertFalse(preview.update())
            finally:
                preview.stop()
        resize.assert_called_once_with("wos-run-preview", 9, 13)
        self.assertEqual(imshow.call_count, 1)
        self.assertEqual(do_resize.call_args.args[1], (9, 13))
        self.assertEqual(wait.call_count, 2)
        destroy.assert_called_once_with("wos-run-preview")

    def test_preview_swallows_stream_and_teardown_error(self):
        session = Mock()
        session.get_frame_snapshot.side_effect = cv2.error("gone")
        with patch("app.cv2.namedWindow"), patch("app.cv2.imshow"), \
                patch("app.cv2.destroyWindow") as destroy:
            destroy.side_effect = cv2.error("gone")
            preview = app.PreviewWindow(session)
            preview.start()
            self.assertFalse(preview.update())
            preview.stop()
        self.assertFalse(preview._open)

    def test_closed_window_stops_only_preview(self):
        session = Mock()
        session.get_frame_snapshot.return_value = None
        with patch("app.cv2.namedWindow"), patch("app.cv2.waitKey"), \
                patch("app.cv2.getWindowProperty", return_value=0), patch("app.cv2.destroyWindow"):
            preview = app.PreviewWindow(session)
            self.assertTrue(preview.start())
            self.assertFalse(preview.update())
            preview.stop()


if __name__ == "__main__":
    unittest.main()
