import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from automation.config import load_plan
from automation.runner import run_list
from automation.runtime import AutomationError, FrameTimeout, Runtime
from automation.vision import Detector


class FakeStream:
    """Publish a new frame only when virtual time advances, as a video stream would."""

    def __init__(self, images):
        self.images = images
        self.now = 100.0
        self.decode_time = self.now
        self.frame_id = 1
        self.scene = "town"
        self.pending = None
        self.calls = []
        self.controller = self
        self.frame_size = (96, 64)
        self.device_size = (192, 128)
        self.resize_on_open = None
        self.conflicting_targets = False
        self.conflicting_scenes = False

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.frame_id += 1
        self.decode_time = self.now
        if self.pending is not None:
            self.scene, self.pending = self.pending, None

    def check_stream_health(self):
        pass

    def get_frame_snapshot(self):
        width, height = self.frame_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        names = {
            "town": ["town"],
            "wild": ["wild", "open"],
            "popup": ["wild", "popup", "old_result"],
            "completed": ["wild", "popup", "success"],
        }[self.scene][:]
        if self.conflicting_targets and self.scene == "completed":
            names.append("old_result")
        if self.conflicting_scenes:
            names.append("wild" if self.scene == "town" else "town")
        for name in names:
            x, y, image = self.images[name]
            if y + 8 <= height and x + 8 <= width:
                frame[y:y + 8, x:x + 8] = image
        return SimpleNamespace(frame=frame, frame_id=self.frame_id, decode_time=self.decode_time,
                               frame_size=self.frame_size, device_size=self.device_size)

    def tap(self, x, y):
        self.calls.append(("tap", x, y, self.scene, self.frame_id))
        if (x, y, self.scene) == (120, 100, "town"):
            self.pending = "wild"
        elif (x, y, self.scene) == (88, 48, "wild"):
            self.pending = "popup"
            if self.resize_on_open == "frame":
                self.frame_size = (97, 64)
            elif self.resize_on_open == "device":
                self.device_size = (193, 128)
        elif (x, y, self.scene) == (140, 100, "popup"):
            self.pending = "completed"
        else:
            raise AssertionError(f"Unexpected tap {(x, y, self.scene)}")

    def back(self):
        self.calls.append(("back", self.scene, self.frame_id))
        if self.scene not in ("popup", "completed"):
            raise AssertionError(f"Unexpected back from {self.scene}")
        self.pending = "wild"


class AutomationIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "tasks" / "templates").mkdir(parents=True)
        rng = np.random.default_rng(42)
        positions = {"town": (2, 2), "wild": (14, 2), "popup": (26, 2),
                     "open": (40, 20), "old_result": (52, 20), "success": (64, 20)}
        self.images = {}
        for name, (x, y) in positions.items():
            image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
            self.images[name] = (x, y, image)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            (self.root / "tasks" / "templates" / f"{name}.png").write_bytes(encoded.tobytes())
        self.common = {"device": {"frame_size": [96, 64], "device_size": [192, 128]},
                       "scenes": {
            "town": {"type": "main", "image": "tasks/templates/town.png", "threshold": 0.99},
            "wild": {"type": "main", "image": "tasks/templates/wild.png", "threshold": 0.99},
            "popup": {"type": "sub", "image": "tasks/templates/popup.png", "threshold": 0.99,
                      "covers": ["wild"], "exit": [
                          {"back": True}, {"wait_scene": {"scenes": ["wild"], "timeout": 1}}]},
        }, "switches": [{"from": "town", "to": "wild", "steps": [
            {"click": [120, 100]},
            {"wait_scene": {"scenes": ["wild"], "timeout": 1}},
        ]}]}
        self.stream = FakeStream(self.images)
        clock = patch("time.monotonic", self.stream.monotonic)
        sleep = patch("time.sleep", self.stream.sleep)
        clock.start()
        sleep.start()
        self.addCleanup(clock.stop)
        self.addCleanup(sleep.stop)

    def write(self, name, data):
        (self.root / name).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def task(self, name="expedition", fail_after_open=False):
        failure = {"result": "failed", "reason": "old result or deliberate failure"}
        steps = [{"click_image": "tasks/templates/open.png", "require_scene": "wild",
                  "expect": {"scene": "popup"}, "timeout": 1, "on_timeout": "failed"}]
        if fail_after_open:
            branches = [{"when": {"scene": "popup"}, **failure}]
        else:
            steps.append({"click": [140, 100], "require_scene": "popup"})
            branches = [{"when": {"image": "tasks/templates/old_result.png"}, **failure},
                        {"when": {"image": "tasks/templates/success.png"}, "result": "verified",
                         "reason": "recognized success"}]
        steps.append({"detect": {"branches": branches, "timeout": 1, "on_timeout": "failed"}})
        return {"task": name, "start": ["wild"], "steps": steps}

    def make_runtime(self, entries, yaml_dir=Path()):
        (self.root / yaml_dir).mkdir(parents=True, exist_ok=True)
        self.write(yaml_dir / "common.yaml", self.common)
        task_entries = []
        for index, (task, policy) in enumerate(entries):
            filename = yaml_dir / f"task-{index}.yaml"
            self.write(filename, task)
            task_entries.append({"file": filename.as_posix(), "on_failure": policy})
        list_path = yaml_dir / "list.yaml"
        self.write(list_path, {"common": (yaml_dir / "common.yaml").as_posix(), "tasks": task_entries})
        plan = load_plan(list_path, home=self.root)
        detector = Detector(plan.resources, plan.scenes)
        runtime = Runtime(plan, self.stream, detector, frame_timeout=1, poll=0.01)
        return plan, runtime

    def test_end_to_end_switch_round_recognition_and_safe_exit(self):
        plan, runtime = self.make_runtime([(self.task(), "stop")])
        self.assertEqual(list(self.root.rglob("*marker*")), [])
        with self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertTrue(run_list(runtime, plan))
        self.assertTrue(any("result=verified reason=recognized success" in line for line in logs.output))
        self.assertEqual(list(self.root.rglob("*marker*")), [])
        self.assertEqual([call[:-1] for call in self.stream.calls], [
            ("tap", 120, 100, "town"), ("tap", 88, 48, "wild"),
            ("tap", 140, 100, "popup"), ("back", "completed")])
        ids = [call[-1] for call in self.stream.calls]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(self.stream.scene, "wild")
        self.assertEqual(runtime.observe()[1], "wild")

    def test_replaced_images_after_loading_do_not_change_execution(self):
        plan, runtime = self.make_runtime([(self.task(), "stop")])
        names = list(self.images)
        for index, name in enumerate(names):
            replacement = self.images[names[(index + 1) % len(names)]][2]
            (self.root / "tasks" / "templates" / f"{name}.png").write_bytes(cv2.imencode(".png", replacement)[1].tobytes())
        self.assertTrue(run_list(runtime, plan))
        self.assertEqual([call[:-1] for call in self.stream.calls], [
            ("tap", 120, 100, "town"), ("tap", 88, 48, "wild"),
            ("tap", 140, 100, "popup"), ("back", "completed")])

    def test_failed_skip_recovers_before_next_task(self):
        plan, runtime = self.make_runtime([(self.task("first", True), "skip"),
                                           (self.task("second"), "stop")])
        with self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertTrue(run_list(runtime, plan))
        self.assertTrue(any("task=first" in line and "result=failed reason=old result or deliberate failure" in line
                            for line in logs.output))
        self.assertTrue(any("task=second" in line and "result=verified reason=recognized success" in line
                            for line in logs.output))
        self.assertEqual([call[:-1] for call in self.stream.calls], [
            ("tap", 120, 100, "town"), ("tap", 88, 48, "wild"), ("back", "popup"),
            ("tap", 88, 48, "wild"), ("tap", 140, 100, "popup"), ("back", "completed")])
        self.assertEqual(self.stream.scene, "wild")

    def test_conflicting_transition_targets_stop_without_recovery_or_skip(self):
        self.stream.conflicting_targets = True
        plan, runtime = self.make_runtime([(self.task(), "skip"), (self.task("next"), "stop")])
        with self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertFalse(run_list(runtime, plan))
        self.assertTrue(any("result=uncertain" in line and "conflicting transitions" in line
                            for line in logs.output))
        self.assertFalse(any("task=next" in line for line in logs.output))
        self.assertEqual([call[0] for call in self.stream.calls], ["tap", "tap", "tap"])
        self.assertEqual(self.stream.scene, "completed")

    def test_scene_conflict_stops_before_any_control(self):
        self.stream.conflicting_scenes = True
        plan, runtime = self.make_runtime([(self.task(), "skip")])
        with self.assertLogs("automation.runner", level="ERROR") as logs:
            self.assertFalse(run_list(runtime, plan))
        self.assertTrue(any("Scene conflict" in line for line in logs.output))
        self.assertEqual(self.stream.calls, [])

    def test_size_change_after_control_stops_before_further_controls(self):
        for kind in ("frame", "device"):
            with self.subTest(kind=kind):
                self.stream = FakeStream(self.images)
                self.stream.resize_on_open = kind
                # The clock patches bind the original stream, so use scoped replacements.
                with patch("time.monotonic", self.stream.monotonic), patch("time.sleep", self.stream.sleep):
                    plan, runtime = self.make_runtime([(self.task(), "skip"), (self.task("next"), "stop")])
                    with self.assertLogs("automation", level="ERROR") as logs:
                        self.assertFalse(run_list(runtime, plan))
                self.assertTrue(any(f"{kind}_size" in line for line in logs.output))
                self.assertEqual([call[:-1] for call in self.stream.calls], [
                    ("tap", 120, 100, "town"), ("tap", 88, 48, "wild")])

    def test_stale_frames_do_not_satisfy_after_id(self):
        _, runtime = self.make_runtime([(self.task(), "stop")])
        frame = runtime.snapshot()
        self.assertEqual(runtime.snapshot(after_id=frame.frame_id).frame_id, frame.frame_id + 1)
        self.assertGreater(self.stream.now, 100.0)
        latest = self.stream.frame_id

        def no_new_frame(seconds):
            self.stream.now += seconds

        with patch("time.sleep", no_new_frame):
            with self.assertRaises(FrameTimeout):
                runtime.snapshot(after_id=latest, timeout=0.05)
        self.stream.device_size = (193, 128)
        with self.assertRaisesRegex(AutomationError, "device_size"):
            runtime.snapshot(after_id=999)

    def test_paths_resolve_from_home_not_yaml_directories(self):
        plan, runtime = self.make_runtime([(self.task(), "stop")], Path("nested/config"))
        images = {record["image"] for record in plan.resources.values() if record["action"] == "template"}
        self.assertEqual(images, {self.root / "tasks" / "templates" / f"{name}.png" for name in self.images})
        self.assertTrue(run_list(runtime, plan))
        self.assertEqual(self.stream.scene, "wild")


if __name__ == "__main__":
    unittest.main()
