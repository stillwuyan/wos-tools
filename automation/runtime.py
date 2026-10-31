from __future__ import annotations

import logging
import math
import time

from automation.config import validate_sizes
from automation.cancellation import check_cancelled, sleep
from automation.vision import SceneConflict
from device.coordinates import frame_to_device

logger = logging.getLogger(__name__)


class AutomationError(RuntimeError):
    """The automation cannot proceed safely or verify its result."""


class FrameTimeout(AutomationError):
    """The stream did not produce a suitable frame in time."""


class ActionDeadlineExceeded(AutomationError):
    """A control was not sent because its deadline expired."""


class Runtime:
    def __init__(self, plan, session, detector, *, frame_timeout=5, poll=0.1, max_frame_age=2,
                 cancel_event=None):
        for name, value in (("frame_timeout", frame_timeout), ("poll", poll),
                            ("max_frame_age", max_frame_age)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.plan = plan
        self.session = session
        self.detector = detector
        self.frame_timeout = frame_timeout
        self.poll = poll
        self.max_frame_age = max_frame_age
        self.cancel_event = cancel_event
        self._scene = None
        self._last_wait_frame_id = None

    def snapshot(self, after_id=None, timeout=None):
        if timeout is None:
            timeout = self.frame_timeout
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        while True:
            check_cancelled(self.cancel_event)
            try:
                self.session.check_stream_health()
                frame = self.session.get_frame_snapshot()
            except OSError:
                raise
            except RuntimeError as exc:
                raise AutomationError(f"Video stream unavailable: {exc}") from exc
            if frame is not None:
                try:
                    validate_sizes(self.plan, frame)
                    if (type(frame.frame_id) is not int or frame.frame_id < 0 or
                            type(frame.decode_time) not in (int, float) or
                            not math.isfinite(frame.decode_time) or
                            frame.frame is None or
                            tuple(frame.frame.shape[:2]) != tuple(reversed(frame.frame_size))):
                        raise ValueError("invalid frame metadata")
                except (ValueError, TypeError, AttributeError) as exc:
                    raise AutomationError(f"Invalid frame: {exc}") from exc
                now = time.monotonic()
                age = now - frame.decode_time
                if 0 <= age <= self.max_frame_age and (after_id is None or frame.frame_id > after_id):
                    return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FrameTimeout("Timed out waiting for a healthy new frame")
            sleep(min(self.poll, remaining), self.cancel_event)

    def observe(self, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()
        try:
            scene = self.detector.detect(snapshot)
        except SceneConflict as exc:
            self._scene = None
            raise AutomationError(f"Scene conflict: {exc}") from exc
        if scene is not None and scene not in self.plan.scenes:
            raise AutomationError(f"Unknown detected scene: {scene}")
        self._scene = scene
        return snapshot, scene

    def wait_scene(self, scene_names, timeout, after_id=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise AutomationError("Scene timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        # Without a supplied baseline, a scene wait still requires a subsequent frame.
        if after_id is None:
            try:
                initial = self.snapshot(timeout=max(0, deadline - time.monotonic()))
            except FrameTimeout as exc:
                raise AutomationError(f"Timed out waiting for scenes: {scene_names}") from exc
            after_id = initial.frame_id
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AutomationError(f"Timed out waiting for scenes: {scene_names}")
            try:
                frame = self.snapshot(after_id=after_id, timeout=remaining)
            except FrameTimeout as exc:
                raise AutomationError(f"Timed out waiting for scenes: {scene_names}") from exc
            after_id = frame.frame_id
            _, scene = self.observe(frame)
            if time.monotonic() > deadline:
                raise AutomationError(f"Timed out waiting for scenes: {scene_names}")
            if scene in scene_names:
                self._last_wait_frame_id = frame.frame_id
                return scene

    def execute(self, actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
        return self._execute(actions, expected_scene=expected_scene, before_action=before_action,
                             action_deadline=action_deadline, after_id=after_id)

    def _execute(self, actions, *, guarded=False, expected_scene=None, before_action=None,
                 action_deadline=None, after_id=None):
        last_id = None
        wait_after_id = None
        expected = expected_scene
        if guarded:
            pending = False
            for action in actions:
                kind = next(iter(action))
                if kind in ("click", "swipe", "back"):
                    if pending:
                        raise AutomationError("Exit controls must be separated by wait_scene")
                    pending = True
                elif kind == "wait_scene":
                    pending = False
            if not actions or "wait_scene" not in actions[-1]:
                raise AutomationError("Exit must end with wait_scene")
        for action in actions:
            check_cancelled(self.cancel_event)
            kind, value = next(iter(action.items()))
            if kind == "wait_scene":
                matched = self.wait_scene(value["scenes"], value["timeout"], after_id=wait_after_id)
                last_id = self._last_wait_frame_id
                wait_after_id = last_id
                if guarded or expected_scene is not None:
                    expected = matched
                continue
            if kind == "wait":
                sleep(value, self.cancel_event)
                try:
                    last_id = self.snapshot().frame_id
                except FrameTimeout:
                    # A long wait (e.g. an offline rally task) can end while the device
                    # screen is off, so the stream is silent. Nothing after an
                    # unconditional wait relies on a pre-wait frame; a later state that
                    # actually needs one will still stop on silence.
                    logger.warning(
                        "video stream silent after wait of %ss; no fresh frame within %.3fs "
                        "(the screen may be off)", value, self.frame_timeout)
                    last_id = None
                continue
            if after_id is not None:
                timeout = None if action_deadline is None else max(0, action_deadline - time.monotonic())
                frame = self.snapshot(after_id=after_id, timeout=timeout)
                after_id = None
                frame, scene = self.observe(frame)
            else:
                frame, scene = self.observe()
            if before_action is not None:
                proceed = before_action(frame, scene)
                if not 0 <= time.monotonic() - frame.decode_time <= self.max_frame_age:
                    raise AutomationError("Action frame is no longer fresh")
                if not proceed:
                    return frame.frame_id
            if scene is None:
                raise AutomationError("Cannot act without a uniquely identified scene")
            if expected is not None and scene != expected:
                raise AutomationError(f"Scene changed after verification: expected {expected}, got {scene}")
            if kind == "click":
                resource = self.plan.resources[value["target"]]
                if value["mode"] == "fixed":
                    point = resource.get("device") if resource["action"] == "point" else None
                    if point is None:
                        raise AutomationError("Fixed click requires a point with device coordinates")
                    coords = point["x"], point["y"]
                elif value["mode"] == "match":
                    match = self.detector.match_unique(frame, value["target"])
                    if match is None:
                        raise AutomationError(f"Click target not found: {value['target']}")
                    coords = frame_to_device(*match.center, frame.frame_size, frame.device_size)
                    if coords is None:
                        raise AutomationError("Matched click is outside the frame")
                else:
                    raise AutomationError(f"Unknown click mode: {value['mode']}")
                control = lambda: self.session.controller.tap(*coords)
            elif kind == "swipe":
                start = self.plan.resources[value["from"]]["device"]
                end = self.plan.resources[value["to"]]["device"]
                control = lambda: self.session.controller.swipe(
                    start["x"], start["y"], end["x"], end["y"],
                    duration_ms=max(1, round(value["duration"] * 1000)))
            elif kind == "back":
                control = self.session.controller.back
            else:
                raise AutomationError(f"Unknown action: {kind}")
            if not 0 <= time.monotonic() - frame.decode_time <= self.max_frame_age:
                raise AutomationError("Action frame is no longer fresh")
            if action_deadline is not None and time.monotonic() >= action_deadline:
                raise ActionDeadlineExceeded("Action deadline expired before control")
            check_cancelled(self.cancel_event)
            control()
            # Frames decoded while a control was in progress cannot prove its result.
            baseline = self.snapshot()
            last_id = max(frame.frame_id, baseline.frame_id)
            wait_after_id = last_id
        return last_id

    def prepare(self, allowed_scenes):
        allowed = set(allowed_scenes)
        if not allowed or any(name not in self.plan.scenes or self.plan.scenes[name]["type"] != "main"
                              for name in allowed):
            raise AutomationError("Allowed scenes must be known main scenes")
        visited = set()
        expected = None
        while True:
            _, scene = self.observe()
            if scene is None:
                raise AutomationError("Cannot prepare from an unknown scene")
            if expected is not None and scene != expected:
                raise AutomationError(f"Scene changed after exit verification: expected {expected}, got {scene}")
            expected = None
            if self.plan.scenes[scene]["type"] == "sub":
                if scene in visited:
                    raise AutomationError(f"Repeated exit scene: {scene}")
                visited.add(scene)
                actions = self.plan.scenes[scene].get("exit")
                if not actions:
                    raise AutomationError(f"Scene has no exit: {scene}")
                self._execute(actions, guarded=True, expected_scene=scene)
                expected = self._scene
                continue
            if scene in allowed:
                return
            switches = [switch for switch in self.plan.switches
                        if switch["from"] == scene and switch["to"] in allowed]
            if len(switches) != 1:
                raise AutomationError(f"No unique switch from {scene} to {sorted(allowed)}")
            switch = switches[0]
            actions = switch["actions"]
            if (not actions or "wait_scene" not in actions[-1] or
                    actions[-1]["wait_scene"]["scenes"] != [switch["to"]]):
                raise AutomationError("Switch must verify its target")
            self._execute(actions, guarded=True, expected_scene=scene)
            _, actual = self.observe()
            if actual != switch["to"]:
                raise AutomationError(f"Switch target not confirmed: expected {switch['to']}, got {actual}")
            return

    def finish(self):
        try:
            return self.prepare(name for name, scene in self.plan.scenes.items() if scene["type"] == "main")
        except FrameTimeout:
            # Final recovery takes a frame to identify the scene. If the stream is
            # silent (the screen turned off after a long wait), there is nothing to
            # recover to and nothing left to verify, so warn and report the skip
            # instead of failing the whole list. A live stream that cannot reach a
            # main scene still raises.
            logger.warning(
                "final recovery skipped: video stream silent (the screen may be off); "
                "device state was not verified")
            return "video stream silent"
