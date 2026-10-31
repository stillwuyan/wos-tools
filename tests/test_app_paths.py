import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from home_paths import PROJECT_HOME, resolve_home_path


class HomePathTests(unittest.TestCase):
    def test_project_home_is_root_beside_app(self):
        self.assertEqual(PROJECT_HOME, Path(app.__file__).resolve().parent)

    def test_tilde_is_a_literal_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            for value in ("~", "~/file", "~other/file"):
                with self.subTest(value=value):
                    self.assertEqual(resolve_home_path(value, home), (home / value).resolve())

    @unittest.skipUnless(sys.platform == "win32", "Windows paths")
    def test_windows_path_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            for value, kind in (("D:foo", "drive"), ("D:", "drive"),
                                (r"\foo", "root"), ("/foo", "root"),
                                ("\\", "root"), ("/", "root")):
                for path in (value, Path(value)):
                    with self.subTest(path=path), self.assertRaises(ValueError) as raised:
                        resolve_home_path(path, home)
                    self.assertEqual(str(raised.exception),
                                     f"Windows {kind}-relative path cannot be resolved against home: {path}")
            for value in ("foo", r"dir\foo", "dir/foo", str(home / "absolute")):
                for path in (value, Path(value)):
                    with self.subTest(path=path):
                        self.assertEqual(resolve_home_path(path, home), (home / path).resolve())
            with patch.object(Path, "resolve", lambda self: self):
                unc = r"\\server\share\foo"
                self.assertEqual(resolve_home_path(unc, home), Path(unc))


class ServerJarPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.config = self.home / "sub" / "config.yaml"
        self.config.parent.mkdir()
        self.config.write_text("scrcpy:\n  server_jar: server.jar\n", encoding="utf-8")
        self.jar = self.home / "server.jar"
        self.jar.touch()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_config_and_jar_are_relative_to_home_not_cwd_or_config_directory(self):
        self.assertNotEqual(Path.cwd().resolve(), self.home)
        (self.config.parent / "server.jar").touch()
        with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": "fallback.jar"}):
            self.assertEqual(app.load_server_jar(Path("sub/config.yaml"), None, self.home), self.jar)
        self.assertEqual(app.load_server_jar(self.config, None, self.home / "."), self.jar)

    def test_absolute_paths_and_explicit_priority(self):
        self.assertEqual(app.load_server_jar(self.config, None, self.home), self.jar)
        self.config.write_text(f"scrcpy:\n  server_jar: '{self.jar.as_posix()}'\n", encoding="utf-8")
        self.assertEqual(app.load_server_jar(self.config, None, self.home), self.jar)
        with self.assertRaisesRegex(RuntimeError, "scrcpy-server not found"):
            app.load_server_jar(Path("missing.yaml"), "missing.jar", self.home)
        self.assertEqual(app.load_server_jar(self.config, str(self.jar), self.home), self.jar)

    def test_configured_missing_path_does_not_fall_back(self):
        self.jar.unlink()
        with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": "fallback.jar"}):
            with self.assertRaisesRegex(RuntimeError, "scrcpy server configured .* does not exist"):
                app.load_server_jar(self.config, None, self.home)
        with self.assertRaisesRegex(RuntimeError, "Configuration file not found or unreadable"):
            app.load_server_jar(Path("missing.yaml"), None, self.home)

    def test_environment_and_search_use_home(self):
        self.config.write_text("scrcpy: {}\n", encoding="utf-8")
        (self.home / "env.jar").touch()
        (self.home / "explicit.jar").touch()
        with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": "env.jar"}):
            self.assertEqual(app.load_server_jar(self.config, None, self.home), self.home / "env.jar")
            self.assertEqual(app.load_server_jar(self.config, "explicit.jar", self.home),
                             self.home / "explicit.jar")
        with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": str(self.jar)}):
            self.assertEqual(app.load_server_jar(self.config, None, self.home), self.jar)
        nested = self.home / "nested"
        nested.mkdir()
        candidate = nested / "scrcpy-server.jar"
        candidate.touch()
        self.assertEqual(app.load_server_jar(self.config, None, self.home), candidate)
        candidate.unlink()
        with self.assertRaisesRegex(RuntimeError, "Could not find scrcpy-server"):
            app.load_server_jar(self.config, None, self.home)

    def test_tilde_config_and_server_paths_are_relative_to_home(self):
        tilde = self.home / "~"
        tilde.mkdir()
        config = tilde / "config.yaml"
        jar = tilde / "server.jar"
        jar.touch()
        config.write_text("scrcpy:\n  server_jar: ~/server.jar\n", encoding="utf-8")
        self.assertEqual(app.load_server_jar("~/config.yaml", None, self.home), jar)
        self.assertEqual(app.load_server_jar("missing.yaml", "~/server.jar", self.home), jar)
        config.write_text("scrcpy: {}\n", encoding="utf-8")
        with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": "~/server.jar"}):
            self.assertEqual(app.load_server_jar("~/config.yaml", None, self.home), jar)

    def test_home_defaults_to_project_home(self):
        with patch.object(app, "PROJECT_HOME", self.home), patch.object(app.Path, "cwd", side_effect=AssertionError):
            self.assertEqual(app.load_server_jar(Path("sub/config.yaml"), None), self.jar)

        original = Path.cwd()
        try:
            os.chdir(self.home)
            with patch.object(Path, "is_file", return_value=True):
                self.assertEqual(app.load_server_jar(Path("unused.yaml"), "server.jar"),
                                 PROJECT_HOME / "server.jar")
        finally:
            os.chdir(original)

    def test_missing_explicit_and_environment_keep_priority_and_errors(self):
        with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": "missing-env.jar"}):
            with self.assertRaisesRegex(RuntimeError, "scrcpy-server not found: .*missing-explicit.jar"):
                app.load_server_jar(Path("missing.yaml"), "missing-explicit.jar", self.home)
            self.config.write_text("scrcpy: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "scrcpy-server not found: .*missing-env.jar"):
                app.load_server_jar(self.config, None, self.home)

    @unittest.skipUnless(sys.platform == "win32", "Windows relative paths")
    def test_windows_ambiguous_server_paths_are_rejected(self):
        for value, error in (("D:server.jar", "drive-relative"),
                             (r"\server.jar", "root-relative"),
                             ("/server.jar", "root-relative")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, error):
                    app.load_server_jar(value, None, self.home)
                with self.assertRaisesRegex(ValueError, error):
                    app.load_server_jar(self.config, value, self.home)
                self.config.write_text(f"scrcpy:\n  server_jar: '{value}'\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    app.load_server_jar(self.config, None, self.home)
                self.config.write_text("scrcpy: {}\n", encoding="utf-8")
                with patch.dict(os.environ, {"SCRCPY_SERVER_PATH": value}):
                    with self.assertRaisesRegex(ValueError, error):
                        app.load_server_jar(self.config, None, self.home)

    @unittest.skipIf(sys.platform == "win32", "Unix filenames")
    def test_unix_drive_like_filename_is_not_rejected(self):
        jar = self.home / "D:server.jar"
        jar.touch()
        self.assertEqual(app.load_server_jar(self.config, "D:server.jar", self.home), jar)


class ModePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.args = SimpleNamespace(home=self.home / ".", config="sub/config.yaml", server_jar="server.jar",
                                    adb="adb", serial=None, host="host", port=1, socket_name="socket",
                                    output_dir="captures", window_scale=0.5, task_list="tasks.yaml",
                                    command="back", screen_on=None, stay_awake=None)

    def test_mark_passes_home_and_resolves_output(self):
        with patch.object(app, "load_server_jar") as load, patch.object(app, "ScrcpySession") as session, \
                patch.object(app, "MarkerUI") as marker:
            self.assertEqual(app.run_mark_mode(self.args), 0)
        load.assert_called_once_with(self.home / "sub/config.yaml", "server.jar", home=self.home)
        marker.assert_called_once_with(session=session.return_value, output_dir=self.home / "captures",
                                       window_scale=0.5, home=self.home)
        marker.return_value.run.assert_called_once_with()

    def test_mark_absolute_output_is_preserved(self):
        self.args.output_dir = str(self.home.parent / "absolute-captures")
        with patch.object(app, "load_server_jar"), patch.object(app, "ScrcpySession"), \
                patch.object(app, "MarkerUI") as marker:
            self.assertEqual(app.run_mark_mode(self.args), 0)
        self.assertEqual(marker.call_args.kwargs["output_dir"], Path(self.args.output_dir).resolve())

    def test_mark_tilde_output_is_relative_to_home(self):
        self.args.output_dir = "~/captures"
        with patch.object(app, "load_server_jar"), patch.object(app, "ScrcpySession"), \
                patch.object(app, "MarkerUI") as marker:
            self.assertEqual(app.run_mark_mode(self.args), 0)
        self.assertEqual(marker.call_args.kwargs["output_dir"], self.home / "~" / "captures")

    def test_automation_and_cmd_pass_home(self):
        with patch.object(app, "load_server_jar") as load, patch.object(app, "ScrcpySession"), \
                patch.object(app, "PreviewWindow"), \
                patch.object(app, "load_plan") as plan, patch.object(app, "Detector"), \
                patch.object(app, "Runtime"), patch.object(app, "run_list", return_value=True):
            plan.return_value.loop = None
            self.assertEqual(app.run_automation_mode(self.args), 0)
        plan.assert_called_once_with("tasks.yaml", home=self.home)
        load.assert_called_once_with(self.home / "sub/config.yaml", "server.jar", home=self.home)
        with patch.object(app, "load_server_jar") as load, patch.object(app, "ScrcpySession"), \
                patch.object(app.time, "sleep"):
            self.assertEqual(app.run_cmd_mode(self.args), 0)
        load.assert_called_once_with(self.home / "sub/config.yaml", "server.jar", home=self.home)

    def test_modes_default_home_to_project_home_from_another_directory(self):
        del self.args.home
        original = Path.cwd()
        try:
            os.chdir(self.home)
            for runner in (app.run_mark_mode, app.run_automation_mode, app.run_cmd_mode):
                with self.subTest(runner=runner.__name__), \
                        patch.object(app, "load_server_jar") as load, patch.object(app, "ScrcpySession"), \
                        patch.object(app, "MarkerUI") as marker, patch.object(app, "load_plan") as plan, \
                        patch.object(app, "Detector"), patch.object(app, "Runtime"), \
                        patch.object(app, "PreviewWindow"), \
                        patch.object(app, "run_list", return_value=True), patch.object(app.time, "sleep"):
                    plan.return_value.loop = None
                    self.assertEqual(runner(self.args), 0)
                load.assert_called_once_with(PROJECT_HOME / "sub/config.yaml", "server.jar", home=PROJECT_HOME)
                if runner is app.run_mark_mode:
                    self.assertEqual(marker.call_args.kwargs["home"], PROJECT_HOME)
                    self.assertEqual(marker.call_args.kwargs["output_dir"], PROJECT_HOME / "captures")
                if runner is app.run_automation_mode:
                    plan.assert_called_once_with("tasks.yaml", home=PROJECT_HOME)
        finally:
            os.chdir(original)

    def test_main_captures_project_home_before_parsing(self):
        args = SimpleNamespace(mode="mark")
        with patch.object(app.Path, "cwd", side_effect=AssertionError), \
                patch.object(app, "parse_args", return_value=args), \
                patch.object(app, "run_mark_mode", return_value=0) as run:
            self.assertEqual(app.main(), 0)
        run.assert_called_once_with(args)
        self.assertEqual(args.home, PROJECT_HOME)

        with patch.object(app, "parse_args", return_value=args), \
                patch.object(app, "run_mark_mode", return_value=0):
            self.assertEqual(app.main(home=self.home), 0)
        self.assertEqual(args.home, self.home)

    def test_adb_paths_are_resolved_against_captured_home_in_all_modes(self):
        for mode, runner in (("mark", app.run_mark_mode), ("run", app.run_automation_mode),
                             ("cmd", app.run_cmd_mode)):
            for adb in ("adb", "adb2", "./adb", "tools/adb", r"tools\adb", str(self.home / "adb")):
                with self.subTest(mode=mode, adb=adb):
                    self.args.adb = adb
                    with patch.object(app, "load_server_jar"), patch.object(app, "ScrcpySession") as session, \
                            patch.object(app, "MarkerUI"), patch.object(app, "load_plan") as plan, \
                            patch.object(app, "Detector"), patch.object(app, "Runtime"), \
                            patch.object(app, "PreviewWindow"), \
                            patch.object(app, "run_list", return_value=True), patch.object(app.time, "sleep"):
                        plan.return_value.loop = None
                        self.assertEqual(runner(self.args), 0)
                    expected = adb if adb in ("adb", "adb2") else str((self.home / adb).resolve())
                    self.assertEqual(session.call_args.kwargs["adb"], expected)

    def test_main_home_is_used_after_cwd_changes_during_parsing(self):
        self.args.mode = "mark"
        self.args.adb = "tools/adb"
        original = Path.cwd()

        def parse_and_change_cwd():
            os.chdir(self.home.parent)
            return self.args

        try:
            os.chdir(self.home)
            with patch.object(app, "parse_args", side_effect=parse_and_change_cwd), \
                    patch.object(app, "load_server_jar") as load, patch.object(app, "ScrcpySession") as session, \
                    patch.object(app, "MarkerUI") as marker:
                self.assertEqual(app.main(), 0)
            self.assertEqual(session.call_args.kwargs["adb"], str(PROJECT_HOME / "tools" / "adb"))
            load.assert_called_once_with(PROJECT_HOME / "sub" / "config.yaml", "server.jar", home=PROJECT_HOME)
            self.assertEqual(marker.call_args.kwargs["output_dir"], PROJECT_HOME / "captures")
        finally:
            os.chdir(original)

    def test_missing_server_never_constructs_session_in_any_mode(self):
        self.args.config = "config.yaml"
        config = self.home / self.args.config
        for mode, runner in (("mark", app.run_mark_mode), ("run", app.run_automation_mode),
                             ("cmd", app.run_cmd_mode)):
            for source in ("explicit", "config", "env"):
                with self.subTest(mode=mode, source=source):
                    config.write_text("scrcpy:\n  server_jar: missing.jar\n" if source == "config"
                                      else "scrcpy: {}\n", encoding="utf-8")
                    self.args.server_jar = "missing.jar" if source == "explicit" else None
                    env = {"SCRCPY_SERVER_PATH": "missing.jar"} if source == "env" else {}
                    with patch.dict(os.environ, env, clear=True), patch.object(app, "ScrcpySession") as session, \
                            patch.object(app, "load_plan"), patch.object(app, "Detector"), \
                            patch.object(app.logger, "exception"):
                        self.assertEqual(runner(self.args), 1)
                    session.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows relative paths")
    def test_windows_ambiguous_options_fail_before_session(self):
        (self.home / "server.jar").touch()
        for runner, field in ((app.run_mark_mode, "output_dir"), (app.run_mark_mode, "config"),
                              (app.run_mark_mode, "server_jar"), (app.run_mark_mode, "adb"),
                              (app.run_automation_mode, "config"), (app.run_automation_mode, "server_jar"),
                              (app.run_automation_mode, "adb"), (app.run_cmd_mode, "config"),
                              (app.run_cmd_mode, "server_jar"), (app.run_cmd_mode, "adb")):
            for value, error in (("D:relative", "drive-relative"),
                                 (r"\relative", "root-relative"), ("/relative", "root-relative")):
                with self.subTest(runner=runner.__name__, field=field, value=value):
                    original = getattr(self.args, field)
                    setattr(self.args, field, value)
                    errors = []
                    try:
                        with patch.object(app, "ScrcpySession") as session, patch.object(app, "load_plan"), \
                                patch.object(app, "Detector"), \
                                patch.object(app.logger, "exception", side_effect=lambda *a: errors.append(sys.exc_info()[1])):
                            self.assertEqual(runner(self.args), 1)
                        session.assert_not_called()
                        self.assertEqual(len(errors), 1)
                        self.assertIsInstance(errors[0], ValueError)
                        self.assertIn(error, str(errors[0]))
                    finally:
                        setattr(self.args, field, original)

    @unittest.skipUnless(sys.platform == "win32", "Windows relative paths")
    def test_windows_ambiguous_task_list_fails_before_session(self):
        for value in ("D:tasks.yaml", r"\tasks.yaml", "/tasks.yaml"):
            with self.subTest(value=value):
                self.args.task_list = value
                errors = []
                with patch.object(app, "ScrcpySession") as session, \
                        patch.object(app.logger, "exception", side_effect=lambda *a: errors.append(sys.exc_info()[1])):
                    self.assertEqual(app.run_automation_mode(self.args), 1)
                session.assert_not_called()
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ValueError)
                self.assertIn("Invalid list path", str(errors[0]))
                self.assertIn("relative", str(errors[0].__cause__))

    @unittest.skipUnless(sys.platform == "win32", "Windows paths")
    def test_windows_adb_path_forms(self):
        for value in (r"tools\adb.exe", "tools/adb.exe", str(self.home / "adb.exe")):
            with self.subTest(value=value):
                self.assertEqual(app.resolve_adb(value, self.home), str((self.home / value).resolve()))
        for value in ("D:adb.exe", "D:", r"D:tools\adb.exe"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "drive-relative"):
                app.resolve_adb(value, self.home)
        for value in (r"\adb.exe", "/adb.exe", "\\", "/"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "root-relative"):
                app.resolve_adb(value, self.home)


if __name__ == "__main__":
    unittest.main()
