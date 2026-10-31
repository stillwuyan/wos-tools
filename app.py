from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path, PureWindowsPath
from typing import Optional

import cv2
import yaml

from calibration import MarkerUI
from device import ScrcpySession
from log import configure_logging, get_logger
from home_paths import PROJECT_HOME, resolve_home_path
from automation.config import load_plan
from automation.runner import run_list, run_loop
from automation.runtime import Runtime
from automation.vision import Detector

configure_logging()
logger = get_logger("app")


def non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def positive_int(value: str) -> int:
    number = non_negative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_screen_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--screen-off", dest="screen_on", action="store_false", default=None,
                        help="Request that the device display be turned off")
    parser.add_argument("--stay-awake", action="store_true", default=None,
                        help="Set the server stay_awake option to true (ineffective with cleanup=false)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WOS tools", usage="%(prog)s [options] {mark,run,cmd} ...")
    sub = parser.add_subparsers(dest="mode", required=True, title="commands", metavar="COMMAND", prog=parser.prog)

    mark = sub.add_parser("mark", help="Run marker mode", description="Capture points and templates from the device.",
                          usage="%(prog)s [options]")
    mark.add_argument("--config", default="config.yaml", help="Application configuration file (default: config.yaml)")
    mark.add_argument("--server-jar", help="Override scrcpy-server path from config.yaml")
    mark.add_argument("--adb", default="adb", help="ADB executable (default: adb)")
    mark.add_argument("--serial", help="ADB device serial")
    mark.add_argument("--host", default="127.0.0.1", help="Listener host (default: 127.0.0.1)")
    mark.add_argument("--port", type=int, default=27183, help="Listener port (default: 27183)")
    mark.add_argument("--socket-name", default="scrcpy", help="ADB reverse socket name (default: scrcpy)")
    mark.add_argument("--output-dir", default="captures", help="Capture output directory (default: captures)")
    mark.add_argument("--window-scale", type=float, default=0.33, help="Preview scale (default: 0.33)")
    add_screen_options(mark)

    run = sub.add_parser("run", help="Run an automation task list", description="Run tasks from an ordered YAML task list.",
                         usage="%(prog)s [options] TASK_LIST")
    run.add_argument("task_list", metavar="TASK_LIST", help="Path to the task list YAML file")
    run.add_argument("--trace-scenes", action="store_true",
                     help="Log per-scene template scores on every detect and save screenshots on scene "
                          "changes or conflicts for covers validation (output under captures/scene-trace/)")
    run.add_argument("--config", default="config.yaml", help="Application configuration file (default: config.yaml)")
    run.add_argument("--server-jar", help="Override scrcpy-server path from config.yaml")
    run.add_argument("--adb", default="adb", help="ADB executable (default: adb)")
    run.add_argument("--serial", help="ADB device serial")
    run.add_argument("--host", default="127.0.0.1", help="Listener host (default: 127.0.0.1)")
    run.add_argument("--port", type=int, default=27183, help="Listener port (default: 27183)")
    run.add_argument("--socket-name", default="scrcpy", help="ADB reverse socket name (default: scrcpy)")
    add_screen_options(run)

    cmd = sub.add_parser("cmd", help="Send a single device command",
                         description="Send one command to the device. Place connection options before COMMAND.",
                         usage="%(prog)s [options] COMMAND [command options]")
    cmd.add_argument("--config", default="config.yaml", help="Application configuration file (default: config.yaml)")
    cmd.add_argument("--server-jar", help="Override scrcpy-server path from config.yaml")
    cmd.add_argument("--adb", default="adb", help="ADB executable (default: adb)")
    cmd.add_argument("--serial", help="ADB device serial")
    cmd.add_argument("--host", default="127.0.0.1", help="Listener host (default: 127.0.0.1)")
    cmd.add_argument("--port", type=int, default=27183, help="Listener port (default: 27183)")
    cmd.add_argument("--socket-name", default="scrcpy", help="ADB reverse socket name (default: scrcpy)")
    add_screen_options(cmd)
    commands = cmd.add_subparsers(dest="command", required=True, title="commands", metavar="{back,click,swipe}",
                                  prog=cmd.prog)
    commands.add_parser("back", help="Press the back key", description="Press the Android back key.",
                        usage="%(prog)s")
    click = commands.add_parser("click", help="Tap a device coordinate", description="Tap a device coordinate.",
                                usage="%(prog)s X Y")
    click.add_argument("x", type=non_negative_int, metavar="X", help="Horizontal device coordinate")
    click.add_argument("y", type=non_negative_int, metavar="Y", help="Vertical device coordinate")
    swipe = commands.add_parser("swipe", help="Swipe between device coordinates",
                                description="Swipe from (X1, Y1) to (X2, Y2) in device coordinates.",
                                usage="%(prog)s [options] X1 Y1 X2 Y2")
    for coordinate, description in (("x1", "Start horizontal coordinate"), ("y1", "Start vertical coordinate"),
                                    ("x2", "End horizontal coordinate"), ("y2", "End vertical coordinate")):
        swipe.add_argument(coordinate, type=non_negative_int, metavar=coordinate.upper(), help=description)
    swipe.add_argument("--duration", type=positive_int, default=300, metavar="MILLISECONDS",
                       help="Swipe duration in milliseconds (default: 300)")
    return parser.parse_args()


def load_server_jar(config_path: Path, explicit: Optional[str], home: Optional[Path] = None) -> Path:
    home = Path(home if home is not None else PROJECT_HOME).expanduser().resolve()
    if explicit:
        path = resolve_home_path(explicit, home)
        if not path.is_file():
            raise RuntimeError(f"scrcpy-server not found: {path}")
        return path

    config_path = resolve_home_path(config_path, home)
    if not config_path.is_file():
        raise RuntimeError(f"Configuration file not found or unreadable: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a YAML mapping: {config_path}")

    scrcpy_config = config.get("scrcpy", {})
    if not isinstance(scrcpy_config, dict):
        raise ValueError(f"'scrcpy' configuration must be a YAML mapping: {config_path}")

    configured_path = scrcpy_config.get("server_jar", "")
    if not isinstance(configured_path, str):
        raise ValueError(f"'scrcpy.server_jar' must be a string: {config_path}")
    configured_path = configured_path.strip()
    if configured_path:
        path = resolve_home_path(configured_path, home)
        if not path.is_file():
            raise RuntimeError(f"scrcpy server configured in {config_path} does not exist: {path}")
        return path

    env_path = os.environ.get("SCRCPY_SERVER_PATH")
    if env_path:
        path = resolve_home_path(env_path, home)
        if not path.is_file():
            raise RuntimeError(f"scrcpy-server not found: {path}")
        return path

    candidates: list[Path] = []
    for pattern in ("**/scrcpy-server.jar", "**/scrcpy-server"):
        for path in home.glob(pattern):
            if path.is_file():
                candidates.append(path.resolve())
    if not candidates:
        raise RuntimeError(
            f"Could not find scrcpy-server. Set scrcpy.server_jar in {config_path}, "
            "or use --server-jar / SCRCPY_SERVER_PATH."
        )
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def load_tap_jitter(config_path: Path, home: Optional[Path] = None) -> int:
    """Read the optional tap-jitter radius (device pixels); 0 disables it.

    A missing file or missing key means no jitter. A present but non-integer or
    negative value is a configuration error.
    """
    home = Path(home if home is not None else PROJECT_HOME).expanduser().resolve()
    config_path = resolve_home_path(config_path, home)
    if not config_path.is_file():
        return 0
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        return 0
    scrcpy_config = config.get("scrcpy")
    if not isinstance(scrcpy_config, dict):
        return 0
    value = scrcpy_config.get("tap_jitter_pixels")
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(f"'scrcpy.tap_jitter_pixels' must be a non-negative integer: {config_path}")
    return value


def load_scrcpy_bool(key: str, default: bool, config_path: Path, home: Optional[Path] = None) -> bool:
    """Read an optional boolean scrcpy.<key> setting; a missing file or key keeps the default."""
    home = Path(home if home is not None else PROJECT_HOME).expanduser().resolve()
    config_path = resolve_home_path(config_path, home)
    if not config_path.is_file():
        return default
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        return default
    scrcpy_config = config.get("scrcpy")
    if not isinstance(scrcpy_config, dict):
        return default
    if "turn_screen_off" in scrcpy_config:
        raise ValueError(f"'scrcpy.turn_screen_off' is obsolete; replace it with the inverse "
                         f"'scrcpy.screen_on': {config_path}")
    value = scrcpy_config.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"'scrcpy.{key}' must be a boolean: {config_path}")
    return value


def resolve_screen_options(args: argparse.Namespace, home: Path, task_scrcpy=None):
    """Resolve screen options: defaults, global config, task list, then CLI."""
    config_path = resolve_home_path(args.config, home)
    screen_on = load_scrcpy_bool("screen_on", True, config_path, home=home)
    stay_awake = load_scrcpy_bool("stay_awake", False, config_path, home=home)
    if task_scrcpy is not None:
        screen_on = task_scrcpy.get("screen_on", screen_on)
        stay_awake = task_scrcpy.get("stay_awake", stay_awake)
    if getattr(args, "screen_on", None) is not None:
        screen_on = args.screen_on
    if getattr(args, "stay_awake", None) is not None:
        stay_awake = args.stay_awake
    return screen_on, stay_awake


def resolve_adb(adb: str, home: Path) -> str:
    if ("/" in adb or "\\" in adb or Path(adb).is_absolute()
            or (sys.platform == "win32" and PureWindowsPath(adb).drive)):
        return str(resolve_home_path(adb, home))
    return adb


def run_mark_mode(args: argparse.Namespace) -> int:
    session: Optional[ScrcpySession] = None
    try:
        home = Path(getattr(args, "home", None) or PROJECT_HOME).expanduser().resolve()
        config_path = resolve_home_path(args.config, home)
        server_jar = load_server_jar(config_path, args.server_jar, home=home)
        adb = resolve_adb(args.adb, home)
        output_dir = resolve_home_path(args.output_dir, home)
        screen_on, stay_awake = resolve_screen_options(args, home)
        session = ScrcpySession(
            server_jar=server_jar,
            adb=adb,
            serial=args.serial,
            host=args.host,
            port=args.port,
            socket_name=args.socket_name,
            screen_on=screen_on,
            stay_awake=stay_awake,
        )
        session.start()
        ui = MarkerUI(session=session, output_dir=output_dir,
                      window_scale=args.window_scale, home=home)
        ui.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.exception("mark mode failed")
        return 1
    finally:
        if session is not None:
            try:
                session.stop()
            except Exception:
                logger.exception("failed to stop session")


class PreviewWindow:
    """Read-only live view; all HighGUI calls run on the calling thread."""

    def __init__(self, session: "ScrcpySession", window_scale: float = 0.33):
        self.session = session
        self.window_scale = max(0.1, min(1.0, window_scale))
        self.window_name = "wos-run-preview"
        self._open = False
        self._display_w = 0
        self._display_h = 0

    def start(self) -> bool:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            logger.warning("preview unavailable: could not create window")
            return False
        self._open = True
        logger.info("preview window enabled: showing device stream")
        return True

    def update(self) -> bool:
        """Render at most one frame and pump events, even without a frame."""
        try:
            snapshot = self.session.get_frame_snapshot()
            if snapshot is not None:
                frame = snapshot.frame
                fh, fw = frame.shape[:2]
                if self._display_w == 0:
                    self._display_w = max(1, int(fw * self.window_scale))
                    self._display_h = max(1, int(fh * self.window_scale))
                    cv2.resizeWindow(self.window_name, self._display_w, self._display_h)
                disp = cv2.resize(frame, (self._display_w, self._display_h))
                cv2.imshow(self.window_name, disp)
            cv2.waitKey(20)
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("preview window closed; automation continues")
                return False
            return True
        except Exception:
            logger.warning("preview unavailable (window or stream error)", exc_info=True)
            return False

    def stop(self) -> None:
        if not self._open:
            return
        self._open = False
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass


def run_automation_mode(args: argparse.Namespace) -> int:
    session: Optional[ScrcpySession] = None
    status = 1
    trace = None
    preview: Optional[PreviewWindow] = None
    worker = None
    cancel_event = threading.Event()
    done = threading.Event()
    result = {"status": 1}
    try:
        home = Path(getattr(args, "home", None) or PROJECT_HOME).expanduser().resolve()
        plan = load_plan(args.task_list, home=home)
        detector = Detector(plan.resources, plan.scenes)
        if getattr(args, "trace_scenes", False):
            from automation.scene_trace import TraceDetector
            trace = TraceDetector(detector, plan.scenes, plan.resources,
                                   home / "captures" / "scene-trace")
            logger.info("scene tracing enabled: per-frame scores and screenshots under captures/scene-trace")
            detector = trace
        config_path = resolve_home_path(args.config, home)
        server_jar = load_server_jar(config_path, args.server_jar, home=home)
        tap_jitter = load_tap_jitter(config_path, home=home)
        screen_on, stay_awake = resolve_screen_options(args, home, getattr(plan, "scrcpy", None))
        session = ScrcpySession(
            server_jar=server_jar,
            adb=resolve_adb(args.adb, home),
            serial=args.serial,
            host=args.host,
            port=args.port,
            socket_name=args.socket_name,
            tap_jitter_pixels=tap_jitter,
            screen_on=screen_on,
            stay_awake=stay_awake,
        )
        session.start()
        preview_active = False
        if not screen_on:
            preview = PreviewWindow(session)
            preview_active = preview.start()
        runtime = Runtime(plan, session, detector, cancel_event=cancel_event)

        def execute_automation():
            try:
                runtime.snapshot()
                if getattr(plan, "loop", None) is not None:
                    result["status"] = 0 if run_loop(runtime, plan.loop, cancel_event=cancel_event) else 1
                else:
                    result["status"] = 0 if run_list(runtime, plan, cancel_event=cancel_event) else 1
            except KeyboardInterrupt:
                result["status"] = 130
            except BaseException:
                logger.exception("run mode failed")
            finally:
                done.set()

        worker = threading.Thread(target=execute_automation, name="automation")
        worker.start()
        while not done.is_set():
            if preview_active:
                preview_active = preview.update()
            else:
                done.wait(0.05)
        worker.join()
        status = result["status"]
    except KeyboardInterrupt:
        cancel_event.set()
        status = 130
    except Exception:
        cancel_event.set()
        logger.exception("run mode failed")
    finally:
        if worker is not None and worker.is_alive():
            cancel_event.set()
            worker.join()
        if preview is not None:
            preview.stop()
        if trace is not None:
            for line in trace.write_summary().splitlines():
                logger.info("scene trace: %s", line)
        if session is not None:
            try:
                session.stop()
            except KeyboardInterrupt:
                status = 130
            except Exception:
                logger.exception("failed to stop session")
                if status != 130:
                    status = 1
    return status


def run_cmd_mode(args: argparse.Namespace) -> int:
    session: Optional[ScrcpySession] = None
    status = 1
    try:
        home = Path(getattr(args, "home", None) or PROJECT_HOME).expanduser().resolve()
        config_path = resolve_home_path(args.config, home)
        server_jar = load_server_jar(config_path, args.server_jar, home=home)
        tap_jitter = load_tap_jitter(config_path, home=home)
        screen_on, stay_awake = resolve_screen_options(args, home)
        session = ScrcpySession(
            server_jar=server_jar, adb=resolve_adb(args.adb, home), serial=args.serial,
            host=args.host, port=args.port, socket_name=args.socket_name,
            tap_jitter_pixels=tap_jitter,
            screen_on=screen_on,
            stay_awake=stay_awake,
        )
        session.start()
        session.check_stream_health()
        controller = session.controller
        points = []
        if args.command == "click":
            points = [(args.x, args.y)]
        elif args.command == "swipe":
            points = [(args.x1, args.y1), (args.x2, args.y2)]
        for x, y in points:
            if not (0 <= x < controller.display_width and 0 <= y < controller.display_height):
                raise ValueError(f"Coordinates out of bounds: ({x}, {y}) not in "
                                 f"{controller.display_width}x{controller.display_height}")
        if args.command == "back":
            controller.back()
        elif args.command == "click":
            controller.tap(args.x, args.y)
        else:
            controller.swipe(args.x1, args.y1, args.x2, args.y2, duration_ms=args.duration)
        logger.warning("Command sent; no device ACK")
        time.sleep(0.2)
        session.check_stream_health()
        status = 0
    except KeyboardInterrupt:
        status = 130
    except Exception:
        logger.exception("cmd mode failed")
    finally:
        if session is not None:
            try:
                session.stop()
            except KeyboardInterrupt:
                status = 130
            except Exception:
                logger.exception("failed to stop session")
                if status != 130:
                    status = 1
    return status


def main(home: Optional[Path] = None) -> int:
    home = Path(home if home is not None else PROJECT_HOME).expanduser().resolve()
    args = parse_args()
    args.home = home
    if args.mode == "mark":
        return run_mark_mode(args)
    if args.mode == "run":
        return run_automation_mode(args)
    if args.mode == "cmd":
        return run_cmd_mode(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
