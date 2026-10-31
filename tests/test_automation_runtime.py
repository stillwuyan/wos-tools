import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from automation.config import Plan
from automation.engine import run_task
from automation.runtime import ActionDeadlineExceeded, AutomationError, FrameTimeout, Runtime
from automation.vision import Detector, Match, SceneConflict


class FakeController:
    def __init__(self, session):
        self.session = session
        self.calls = []
        self.next_scenes = []
        self.error = None

    def _call(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        if self.next_scenes:
            self.session.scene = self.next_scenes.pop(0)

    def tap(self, *args):
        self._call("tap", *args)

    def swipe(self, *args, **kwargs):
        self._call("swipe", *args, kwargs["duration_ms"])

    def back(self):
        self._call("back")


class FakeSession:
    def __init__(self):
        self.frame_id = 0
        self.scene = "town"
        self.advance = True
        self.available = True
        self.age = 0
        self.frame_size = (100, 50)
        self.device_size = (200, 100)
        self.error = None
        self.controller = FakeController(self)

    def check_stream_health(self):
        if self.error:
            raise self.error

    def get_frame_snapshot(self):
        if not self.available:
            return None
        if self.advance:
            self.frame_id += 1
        return SimpleNamespace(frame=np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8),
                               frame_id=self.frame_id, decode_time=time.monotonic() - self.age,
                               frame_size=self.frame_size, device_size=self.device_size, scene=self.scene)


class FakeDetector:
    def __init__(self):
        self.result = Match(10, 5, 6, 4, 1)

    def detect(self, snapshot):
        if snapshot.scene == "conflict":
            raise SceneConflict("ambiguous")
        return snapshot.scene

    def match(self, snapshot, name):
        return self.result

    def match_unique(self, snapshot, name):
        return self.result


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        size = {"frame_size": {"width": 100, "height": 50},
                "device_size": {"width": 200, "height": 100}}
        self.resources = {
            "a": dict(size, action="point", device={"x": 8, "y": 9}),
            "b": dict(size, action="point", device={"x": 20, "y": 30}),
            "image": dict(size, action="template"),
        }
        self.scenes = {"town": {"type": "main"}, "wild": {"type": "main"},
                       "popup": {"type": "sub"}, "other": {"type": "sub"}}
        self.plan = Plan(self.resources, self.scenes, [], [])
        self.session = FakeSession()
        self.detector = FakeDetector()
        self.runtime = Runtime(self.plan, self.session, self.detector,
                               frame_timeout=0.015, poll=0.001, max_frame_age=0.05)

    @staticmethod
    def wait_for(*scenes):
        return {"wait_scene": {"scenes": list(scenes), "timeout": 0.02}}

    def test_snapshot_timeout_age_health_and_strict_id(self):
        self.session.available = False
        with self.assertRaises(FrameTimeout):
            self.runtime.snapshot()
        self.session.available = True
        self.session.age = 1
        with self.assertRaises(AutomationError):
            self.runtime.snapshot()
        self.session.age = 0
        self.session.advance = False
        frame = self.runtime.snapshot()
        with self.assertRaises(AutomationError):
            self.runtime.snapshot(after_id=frame.frame_id)
        self.session.error = RuntimeError("closed")
        with self.assertRaisesRegex(AutomationError, "closed"):
            self.runtime.snapshot()
        self.session.error = ConnectionError("socket")
        with self.assertRaises(ConnectionError):
            self.runtime.snapshot()

    def test_dynamic_size_validation_even_for_old_frame(self):
        self.runtime.snapshot()
        self.session.device_size = (100, 100)
        with self.assertRaisesRegex(AutomationError, "device_size"):
            self.runtime.snapshot(after_id=100)

    def test_observe_tracks_actual_scene_and_clears_on_conflict(self):
        self.assertEqual(self.runtime.observe()[1], "town")
        self.assertEqual(self.runtime._scene, "town")
        self.session.scene = "popup"
        self.assertEqual(self.runtime.observe()[1], "popup")
        self.assertEqual(self.runtime._scene, "popup")
        self.session.scene = "town"
        self.runtime.observe()
        self.session.controller.next_scenes = ["popup"]
        self.runtime.execute([{"back": True}])
        self.assertEqual(self.runtime.observe()[1], "popup")
        self.assertEqual(self.runtime._scene, "popup")
        self.session.scene = "other"
        self.assertEqual(self.runtime.observe()[1], "other")
        self.assertEqual(self.runtime._scene, "other")
        self.session.scene = "conflict"
        with self.assertRaisesRegex(AutomationError, "conflict"):
            self.runtime.observe()
        self.assertIsNone(self.runtime._scene)

    def test_all_controls_and_post_action_frame(self):
        actions = [{"click": {"target": "a", "mode": "fixed"}},
                   {"click": {"target": "image", "mode": "match"}},
                   {"swipe": {"from": "a", "to": "b", "duration": 0.25}},
                   {"wait": 0}, {"back": True}]
        last_id = self.runtime.execute(actions)
        self.assertEqual(last_id, self.session.frame_id)
        self.assertEqual(self.session.controller.calls,
                         [("tap", 8, 9), ("tap", 26, 14),
                          ("swipe", 8, 9, 20, 30, 250), ("back",)])
        self.assertIsNone(self.runtime.execute([]))

    def test_fixed_click_rejects_template_and_missing_point_coordinates(self):
        self.resources["image"]["device_center"] = {"x": 40, "y": 50}
        self.resources["image"]["device"] = {"x": 40, "y": 50}
        self.resources["a"].pop("device")
        for target in ("image", "a"):
            with self.subTest(target=target), self.assertRaisesRegex(AutomationError, "requires a point"):
                self.runtime.execute([{"click": {"target": target, "mode": "fixed"}}])
        self.assertEqual(self.session.controller.calls, [])

    def test_reject_unknown_conflict_missing_match_and_propagate_control_errors(self):
        self.session.scene = None
        with self.assertRaises(AutomationError):
            self.runtime.execute([{"back": True}])
        self.session.scene = "conflict"
        with self.assertRaises(AutomationError):
            self.runtime.execute([{"back": True}])
        self.session.scene = "town"
        self.detector.result = None
        with self.assertRaises(AutomationError):
            self.runtime.execute([{"click": {"target": "image", "mode": "match"}}])
        self.assertEqual(self.session.controller.calls, [])
        self.session.controller.error = BrokenPipeError("broken")
        with self.assertRaises(BrokenPipeError):
            self.runtime.execute([{"back": True}])
        self.session.controller.error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.runtime.execute([{"back": True}])

    def test_real_detector_rejects_duplicate_click_targets(self):
        template = np.array(
            [[15, 90, 23, 55], [47, 210, 72, 131], [199, 31, 166, 3]], dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            path.write_bytes(encoded.tobytes())
            self.resources["image"]["image"] = path
            self.resources["image"]["grayscale"] = template
            scenes = {"town": {"type": "main", "template": "image"}}
            plan = Plan(self.resources, scenes, [], [])
            detector = Detector(plan.resources, scenes)
            runtime = Runtime(plan, self.session, detector, max_frame_age=2)
            original_snapshot = self.session.get_frame_snapshot
            duplicate = True

            def snapshot():
                frame = original_snapshot()
                image = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                frame.frame[8:11, 12:16] = image
                if duplicate:
                    frame.frame[25:28, 60:64] = image
                return frame

            action = [{"click": {"target": "image", "mode": "match"}}]
            with patch.object(self.session, "get_frame_snapshot", side_effect=snapshot), patch.object(
                    self.session.controller, "tap", wraps=self.session.controller.tap) as tap:
                self.assertIsNotNone(detector.match(snapshot(), "image"))
                with self.assertRaisesRegex(ValueError, "Multiple matches.*image"):
                    runtime.execute(action, expected_scene="town")
                tap.assert_not_called()
                duplicate = False
                runtime.execute(action, expected_scene="town")
                tap.assert_called_once_with(28, 18)

    def test_control_returns_baseline_without_waiting_for_next_frame(self):
        def stop_after_control():
            self.session.frame_id += 1
            self.session.advance = False

        with patch.object(self.session.controller, "back", side_effect=stop_after_control):
            self.assertEqual(self.runtime.execute([{"back": True}]), self.session.frame_id)

    def test_wait_scene_checks_first_frame_after_control_baseline(self):
        self.session.controller.next_scenes = ["wild"]
        self.runtime.execute([{"back": True}, self.wait_for("wild")], expected_scene="town")
        self.assertEqual(self.runtime._last_wait_frame_id, 3)
        self.assertEqual(self.session.frame_id, 3)

    def test_wait_scene_requires_new_frames_and_bounds_unknown(self):
        self.session.advance = False
        with self.assertRaisesRegex(AutomationError, "waiting for scenes.*town"):
            self.runtime.wait_scene(["town"], 0.01)
        self.session.advance = True
        self.session.scene = None
        with self.assertRaises(AutomationError):
            self.runtime.wait_scene(["town"], 0.01)
        self.session.scene = "conflict"
        with self.assertRaisesRegex(AutomationError, "conflict"):
            self.runtime.wait_scene(["town"], 0.01)
        self.session.scene = "town"
        self.assertEqual(self.runtime.wait_scene(["town"], 0.02), "town")

    def test_each_control_uses_latest_snapshot_and_declared_scene(self):
        for action in ({"back": True}, {"click": {"target": "a", "mode": "fixed"}},
                       {"swipe": {"from": "a", "to": "b", "duration": 0.1}}):
            with self.subTest(action=action):
                self.session.scene = "town"
                self.runtime.observe()
                self.session.scene = "wild"
                with self.assertRaisesRegex(AutomationError, "expected town, got wild"):
                    self.runtime.execute([action], expected_scene="town")
                self.assertEqual(self.session.controller.calls, [])
        self.session.scene = "town"
        self.session.controller.next_scenes = ["wild"]
        with self.assertRaisesRegex(AutomationError, "expected town, got wild"):
            self.runtime.execute([{"back": True}, {"back": True}], expected_scene="town")
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_wait_only_does_not_require_a_recognized_scene(self):
        self.session.scene = None
        self.assertIsNotNone(self.runtime.execute([{"wait": 0}]))

    def test_wait_returns_none_when_stream_is_silent_after_sleep(self):
        self.session.available = False
        with self.assertLogs("automation.runtime", level="WARNING") as logs:
            self.assertIsNone(self.runtime.execute([{"wait": 0}]))
        self.assertTrue(any("video stream silent" in s for s in logs.output))

    def test_finish_tolerates_silent_stream_after_long_wait(self):
        self.session.available = False
        with self.assertLogs("automation.runtime", level="WARNING") as logs:
            self.assertEqual(self.runtime.finish(), "video stream silent")
        self.assertTrue(any("final recovery skipped" in s for s in logs.output))
        self.assertEqual(self.session.controller.calls, [])

    def test_engine_retry_rechecks_actual_frame_before_control(self):
        task = {"task": "retry", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.005,
                      "on_timeout": "failed", "transitions": [{"when": {"scene": "other"}, "to": "done"}],
                      "retry": {"count": 1, "interval": 0.001, "require_scene": "town"}},
            "done": {"result": "verified"}}}
        original_execute = self.runtime.execute
        executions = []

        def execute(actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
            executions.append(expected_scene)
            if len(executions) == 2:
                self.session.scene = "wild"
            return original_execute(actions, expected_scene=expected_scene, before_action=before_action,
                                    action_deadline=action_deadline, after_id=after_id)

        with patch.object(self.runtime, "execute", side_effect=execute):
            with self.assertRaisesRegex(AutomationError, "expected town, got wild"):
                run_task(self.runtime, task)
        self.assertEqual(executions, ["town", "town"])
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_engine_detects_only_first_post_baseline_frame(self):
        task = {"task": "first", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.02,
                      "on_timeout": "failed", "transitions": [{"when": {"target": "success"}, "to": "done"}]},
            "done": {"result": "verified"}}}
        with patch.object(self.detector, "match", side_effect=lambda frame, name: object() if frame.frame_id == 3 else None):
            self.assertEqual(run_task(self.runtime, task).result, "verified")
        self.assertEqual(self.session.frame_id, 3)
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_engine_retry_detects_target_on_actual_control_frame(self):
        task = {"task": "retry", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.005,
                      "on_timeout": "failed", "transitions": [{"when": {"target": "success"}, "to": "done"}],
                      "retry": {"count": 1, "interval": 0.001, "require_scene": "town"}},
            "done": {"result": "verified"}}}
        original_execute = self.runtime.execute
        calls = []

        def execute(actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
            calls.append(before_action)
            return original_execute(actions, expected_scene=expected_scene, before_action=before_action,
                                    action_deadline=action_deadline, after_id=after_id)

        with patch.object(self.runtime, "execute", side_effect=execute), patch.object(
                self.detector, "match", side_effect=lambda frame, name: object() if len(calls) == 2 else None):
            self.assertEqual(run_task(self.runtime, task).result, "verified")
        self.assertEqual(self.session.controller.calls, [("back",)])
        self.assertEqual(self.session.scene, "town")

    def test_retry_callback_rejects_frame_that_ages_during_matching(self):
        def callback(frame, scene):
            frame.decode_time -= 1
            return False

        with self.assertRaisesRegex(AutomationError, "no longer fresh"):
            self.runtime.execute([{"back": True}], expected_scene="town", before_action=callback)
        self.assertEqual(self.session.controller.calls, [])

    def test_execute_waits_for_new_action_frame_until_deadline(self):
        now = [0.0]
        self.session.advance = False
        self.runtime.max_frame_age = 10
        self.runtime.poll = 0.01
        frames = []

        def sleep(duration):
            now[0] += duration
            if now[0] >= 0.05:
                self.session.frame_id = 1

        with patch("automation.runtime.time.monotonic", side_effect=lambda: now[0]), patch(
                "automation.runtime.time.sleep", side_effect=sleep):
            self.runtime.execute([{"back": True}], expected_scene="town", after_id=0,
                                 action_deadline=0.1,
                                 before_action=lambda frame, scene: frames.append(frame.frame_id) or True)
        self.assertEqual(frames, [1])
        self.assertEqual(self.session.controller.calls, [("back",)])
        self.assertGreaterEqual(now[0], 0.05)

        now[0] = 0
        self.session.frame_id = 0
        self.session.controller.calls.clear()
        with patch("automation.runtime.time.monotonic", side_effect=lambda: now[0]), patch(
                "automation.runtime.time.sleep", side_effect=lambda duration: now.__setitem__(0, now[0] + duration)):
            with self.assertRaises(FrameTimeout):
                self.runtime.execute([{"back": True}], expected_scene="town", after_id=0,
                                     action_deadline=0.03)
        self.assertEqual(self.session.controller.calls, [])

    def test_real_runtime_retry_with_15fps_stream(self):
        action = {"click": {"target": "a", "mode": "fixed"}}
        task = {"task": "retry", "initial_state": "start", "states": {
            "start": {"actions": [action], "require_scene": "town", "timeout": 0.21,
                      "on_timeout": "failed", "transitions": [{"when": {"target": "success"}, "to": "done"}],
                      "retry": {"count": 1, "interval": 0.08, "require_scene": "town"}},
            "done": {"result": "verified"}}}
        for delayed_result in (False, True):
            with self.subTest(delayed_result=delayed_result):
                now = [0.0]
                self.session.controller.calls.clear()
                self.runtime.poll = 0.01
                self.runtime.frame_timeout = 0.1
                self.runtime.max_frame_age = 1
                reads = []
                action_watermarks = []
                matches = []
                original_execute = self.runtime.execute

                def snapshot():
                    frame_id = int((now[0] + 1e-9) * 15)
                    reads.append(frame_id)
                    return SimpleNamespace(frame=np.zeros((50, 100, 3), dtype=np.uint8), frame_id=frame_id,
                                           decode_time=frame_id / 15, frame_size=(100, 50),
                                           device_size=(200, 100), scene="town")

                def execute(actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
                    if before_action is not None:
                        action_watermarks.append(after_id)
                    return original_execute(actions, expected_scene=expected_scene, before_action=before_action,
                                            action_deadline=action_deadline, after_id=after_id)

                def match(frame, name):
                    matches.append(frame.frame_id)
                    return object() if delayed_result and action_watermarks and frame.frame_id > action_watermarks[-1] else None

                def sleep(duration):
                    now[0] += duration

                with patch("automation.runtime.time.monotonic", side_effect=lambda: now[0]), patch(
                        "automation.runtime.time.sleep", side_effect=sleep), patch.object(
                            self.session, "get_frame_snapshot", side_effect=snapshot), patch.object(
                                self.runtime, "execute", side_effect=execute), patch.object(
                                    self.detector, "match", side_effect=match):
                    result = run_task(self.runtime, task)
                self.assertEqual(result.result, "verified" if delayed_result else "failed")
                self.assertEqual(len(self.session.controller.calls), 1 if delayed_result else 2)
                self.assertEqual(len(action_watermarks), 1)
                self.assertIn(action_watermarks[0], reads)
                self.assertIn(action_watermarks[0] + 1, matches)
                self.assertGreater(reads.count(action_watermarks[0]), 1)

    def test_engine_loading_frames_with_real_runtime(self):
        task = {"task": "loading", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.02,
                      "on_timeout": "failed", "transitions": [{"when": {"scene": "wild"}, "to": "done"}]},
            "done": {"result": "verified"}}}
        original = self.session.get_frame_snapshot

        def snapshot():
            frame = original()
            if frame.frame_id >= 3:
                frame.scene = None if frame.frame_id < 5 else "wild"
            return frame

        with patch.object(self.session, "get_frame_snapshot", side_effect=snapshot):
            self.assertEqual(run_task(self.runtime, task).result, "verified")
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_retry_action_frame_stall_and_touch_failure_stop_safely(self):
        task = {"task": "retry", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.2,
                      "on_timeout": "failed", "transitions": [{"when": {"scene": "other"}, "to": "done"}],
                      "retry": {"count": 1, "interval": 0.1, "require_scene": "town"}},
            "done": {"result": "verified"}}}
        for stall in (True, False):
            with self.subTest(stall=stall):
                now = [0.0]
                self.session.advance = True
                self.session.controller.calls.clear()
                self.runtime.poll = 0.1
                self.runtime.max_frame_age = 10
                original_execute = self.runtime.execute

                def execute(actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
                    if before_action is not None:
                        if stall:
                            self.session.advance = False
                        else:
                            self.session.controller.error = BrokenPipeError("retry touch failed")
                    return original_execute(actions, expected_scene=expected_scene, before_action=before_action,
                                            action_deadline=action_deadline, after_id=after_id)

                def sleep(duration):
                    now[0] += duration

                self.session.controller.error = None
                with patch("automation.runtime.time.monotonic", side_effect=lambda: now[0]), patch(
                        "automation.runtime.time.sleep", side_effect=sleep), patch.object(
                            self.runtime, "execute", side_effect=execute):
                    with self.assertRaises(FrameTimeout if stall else BrokenPipeError):
                        run_task(self.runtime, task)
                self.assertEqual(len(self.session.controller.calls), 1 if stall else 2)
                self.session.controller.error = None

    def test_action_deadline_is_checked_after_matching_for_all_controls(self):
        now = [10.0]
        self.runtime.max_frame_age = 10
        actions = [
            {"click": {"target": "image", "mode": "match"}},
            {"click": {"target": "a", "mode": "fixed"}},
            {"swipe": {"from": "a", "to": "b", "duration": 0.1}},
            {"back": True},
        ]
        with patch("automation.runtime.time.monotonic", side_effect=lambda: now[0]):
            for action in actions:
                with self.subTest(action=action):
                    with self.assertRaises(ActionDeadlineExceeded):
                        self.runtime.execute([action], expected_scene="town", action_deadline=10.0)
            self.assertEqual(self.session.controller.calls, [])
            self.runtime.execute([{"back": True}], expected_scene="town", action_deadline=11.0)
            self.assertEqual(self.session.controller.calls, [("back",)])

    def test_retry_slow_unique_match_cannot_send_control_after_deadline(self):
        now = [0.0]
        self.runtime.max_frame_age = 10
        self.runtime.poll = 0.1
        action = {"click": {"target": "image", "mode": "match"}}
        task = {"task": "retry", "initial_state": "start", "states": {
            "start": {"actions": [action], "require_scene": "town", "timeout": 0.2,
                      "on_timeout": "failed", "transitions": [{"when": {"scene": "other"}, "to": "done"}],
                      "retry": {"count": 1, "interval": 0.1, "require_scene": "town"}},
            "done": {"result": "verified"}}}
        original_match = self.detector.match_unique
        calls = []

        def match(frame, target):
            calls.append(frame.frame_id)
            if len(calls) == 2:
                now[0] += 0.3
            return original_match(frame, target)

        def sleep(duration):
            now[0] += duration

        with patch("automation.runtime.time.monotonic", side_effect=lambda: now[0]), patch(
                "automation.runtime.time.sleep", side_effect=sleep), patch.object(
                    self.detector, "match_unique", side_effect=match):
            result = run_task(self.runtime, task)
        self.assertEqual(result.result, "uncertain")
        self.assertIn("deadline", result.reason)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.session.controller.calls, [("tap", 26, 14)])

    def test_first_post_frame_transition_cannot_bypass_next_scene_guard(self):
        task = {"task": "guard", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.02,
                      "on_timeout": "failed", "transitions": [{"when": {"target": "success"}, "to": "next"}]},
            "next": {"actions": [{"back": True}], "require_scene": "wild", "next": "done"},
            "done": {"result": "verified"}}}
        with self.assertRaisesRegex(AutomationError, "expected wild, got town"):
            run_task(self.runtime, task)
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_engine_with_real_snapshot_times_out_after_stream_stalls(self):
        task = {"task": "stall", "initial_state": "start", "states": {
            "start": {"actions": [], "timeout": 0.01, "on_timeout": "failed",
                      "transitions": [{"when": {"scene": "wild"}, "to": "done"}]},
            "done": {"result": "verified"}}}
        self.session.advance = False
        with self.assertRaises(FrameTimeout):
            run_task(self.runtime, task)
        self.session.error = RuntimeError("disconnected")
        with self.assertRaisesRegex(AutomationError, "disconnected"):
            run_task(self.runtime, task)

    def test_engine_condition_timeout_with_advancing_stream_is_failed(self):
        task = {"task": "timeout", "initial_state": "start", "states": {
            "start": {"actions": [], "timeout": 0.01, "on_timeout": "failed",
                      "transitions": [{"when": {"scene": "wild"}, "to": "done"}]},
            "done": {"result": "verified"}}}
        self.assertEqual(run_task(self.runtime, task).result, "failed")
        self.assertGreater(self.session.frame_id, 1)

    def test_engine_stalled_after_action_stops_instead_of_failed(self):
        task = {"task": "stall", "initial_state": "start", "states": {
            "start": {"actions": [{"back": True}], "require_scene": "town", "timeout": 0.005,
                      "on_timeout": "failed", "transitions": [{"when": {"scene": "wild"}, "to": "done"}],
                      "retry": {"count": 1, "interval": 0.001, "require_scene": "town"}},
            "done": {"result": "verified"}}}
        def stop():
            self.session.controller.calls.append(("back",))
            self.session.advance = False

        with patch.object(self.session.controller, "back", side_effect=stop):
            with self.assertRaises(FrameTimeout):
                run_task(self.runtime, task)
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_prepare_exit_and_actual_main_then_switch(self):
        self.scenes["popup"]["exit"] = [{"back": True}, self.wait_for("town", "wild")]
        self.plan.switches = [{"from": "wild", "to": "town", "actions":
                               [{"back": True}, self.wait_for("town")]}]
        self.session.scene = "popup"
        self.session.controller.next_scenes = ["wild", "town"]
        self.runtime.prepare(["town"])
        self.assertEqual(self.session.scene, "town")
        self.assertEqual(self.session.controller.calls, [("back",), ("back",)])

    def test_exit_must_be_guarded_and_have_exit(self):
        self.session.scene = "popup"
        with self.assertRaisesRegex(AutomationError, "no exit"):
            self.runtime.finish()
        self.scenes["popup"]["exit"] = [{"back": True}, {"wait": 0}, {"back": True}, self.wait_for("town")]
        with self.assertRaisesRegex(AutomationError, "separated"):
            self.runtime.finish()
        self.assertEqual(self.session.controller.calls, [])

    def test_exit_intermediate_scene_must_remain_verified(self):
        self.scenes["popup"]["exit"] = [{"back": True}, self.wait_for("other"),
                                        {"back": True}, self.wait_for("town")]
        self.session.scene = "popup"
        self.session.controller.next_scenes = ["other", "town"]
        original = self.runtime.wait_scene

        def change_after_wait(*args, **kwargs):
            result = original(*args, **kwargs)
            self.session.scene = "wild"
            return result

        with patch.object(self.runtime, "wait_scene", side_effect=change_after_wait):
            with self.assertRaisesRegex(AutomationError, "Scene changed"):
                self.runtime.finish()
        self.assertEqual(self.session.controller.calls, [("back",)])

    def test_multi_step_exit_and_changed_final_scene(self):
        self.scenes["popup"]["exit"] = [{"back": True}, self.wait_for("other"),
                                        {"back": True}, self.wait_for("town")]
        self.session.scene = "popup"
        self.session.controller.next_scenes = ["other", "town"]
        self.runtime.finish()
        self.assertEqual(self.session.controller.calls, [("back",), ("back",)])
        self.session.scene = "popup"
        self.session.controller.next_scenes = ["other", "town"]
        original = self.runtime._execute

        def change_after_exit(*args, **kwargs):
            result = original(*args, **kwargs)
            self.session.scene = "wild"
            return result

        with patch.object(self.runtime, "_execute", side_effect=change_after_exit):
            with self.assertRaisesRegex(AutomationError, "changed after exit"):
                self.runtime.finish()

    def test_exit_cycles_and_ambiguous_switches_are_bounded(self):
        self.scenes["popup"]["exit"] = [{"back": True}, self.wait_for("popup")]
        self.session.scene = "popup"
        with self.assertRaisesRegex(AutomationError, "Repeated"):
            self.runtime.finish()
        self.session.scene = "wild"
        switch = {"from": "wild", "to": "town", "actions": [{"back": True}, self.wait_for("town")]}
        self.plan.switches = [switch, switch]
        calls = len(self.session.controller.calls)
        with self.assertRaisesRegex(AutomationError, "unique switch"):
            self.runtime.prepare(["town"])
        self.assertEqual(len(self.session.controller.calls), calls)

    def test_switch_also_requires_intermediate_verification(self):
        self.session.scene = "wild"
        self.plan.switches = [{"from": "wild", "to": "town", "actions":
                               [{"back": True}, {"back": True}, self.wait_for("town")]}]
        with self.assertRaisesRegex(AutomationError, "separated"):
            self.runtime.prepare(["town"])
        self.assertEqual(self.session.controller.calls, [])

    def test_switch_intermediate_scene_cannot_change_before_control(self):
        self.session.scene = "wild"
        self.plan.switches = [{"from": "wild", "to": "town", "actions":
                               [{"back": True}, self.wait_for("other"),
                                {"back": True}, self.wait_for("town")]}]
        self.session.controller.next_scenes = ["other", "town"]
        original = self.runtime.wait_scene

        def change_after_wait(*args, **kwargs):
            result = original(*args, **kwargs)
            self.session.scene = "wild"
            return result

        with patch.object(self.runtime, "wait_scene", side_effect=change_after_wait):
            with self.assertRaisesRegex(AutomationError, "expected other, got wild"):
                self.runtime.prepare(["town"])
        self.assertEqual(self.session.controller.calls, [("back",)])


if __name__ == "__main__":
    unittest.main()
