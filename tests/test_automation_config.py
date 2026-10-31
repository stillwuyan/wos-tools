import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from automation import Plan, load_plan, validate_sizes
from automation.config import _actions
from automation.engine import run_task
from automation.vision import Detector, SceneConflict
from home_paths import PROJECT_HOME


class AutomationConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        for folder in ("lists", "tasks", "tasks/templates"):
            (self.root / folder).mkdir()
        self.list_path = self.root / "lists" / "daily.yaml"
        self.common_path = self.root / "tasks" / "common.yaml"
        self.task_path = self.root / "tasks" / "intel.yaml"
        self.image = self.root / "tasks" / "templates" / "target.png"
        self.write_image(np.arange(30, dtype=np.uint8).reshape(5, 6))
        (self.root / "tasks" / "templates" / "popup.png").write_bytes(
            cv2.imencode(".png", np.arange(30, dtype=np.uint8).reshape(5, 6)[::-1])[1].tobytes())
        self.daily = {"common": "tasks/common.yaml", "tasks": [{"file": "tasks/intel.yaml", "on_failure": "stop"}]}
        self.common = {"device": {"frame_size": [100, 200], "device_size": [200, 400]},
                       "scenes": {"town": {"type": "main", "image": "tasks/templates/target.png", "threshold": 0.8,
                                           "region": [0, 0, 50, 50]},
                                  "wild": {"type": "main", "image": "tasks/templates/target.png"},
                                  "popup": {"type": "sub", "image": "tasks/templates/popup.png", "covers": ["town"],
                                            "exit": [{"back": True}, {"wait_scene": {"scenes": ["town"], "timeout": 2}}]}},
                       "switches": [{"from": "town", "to": "wild", "steps": [
                           {"click_image": "tasks/templates/target.png"}, {"wait_scene": {"scenes": ["wild"], "timeout": 3}}]}]}
        self.task = {"task": "intel", "start": "wild", "steps": [
            {"click": [199, 399], "require_scene": "wild", "expect": {"scene": "popup"},
             "retry": {"count": 2, "interval": 1}},
            {"wait": 2}, {"wait_scene": {"scenes": ["town"], "timeout": 4}},
            {"swipe": {"from": [0, 0], "to": [100, 200], "duration": 0.5}, "require_scene": "town"}]}
        self.write_all()

    def write_image(self, image):
        self.image.write_bytes(cv2.imencode(".png", image)[1].tobytes())

    def write_all(self):
        for path, data in ((self.list_path, self.daily), (self.common_path, self.common), (self.task_path, self.task)):
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def load(self):
        self.write_all()
        return load_plan("lists/daily.yaml", home=self.root)

    def invalid(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.load()

    def test_list_scrcpy_options_are_validated_for_tasks_and_loop(self):
        self.assertEqual(self.load().scrcpy, {})
        for loop in (False, True):
            with self.subTest(loop=loop):
                if loop:
                    self.loop_daily({"count": 1, "file": "tasks/intel.yaml",
                                     "success": {"result": "unverified"}})
                self.daily["scrcpy"] = {"screen_on": True, "stay_awake": False}
                self.assertEqual(self.load().scrcpy, self.daily["scrcpy"])
                self.daily["scrcpy"] = {"stay_awake": True}
                self.assertEqual(self.load().scrcpy, {"stay_awake": True})
                for bad in (None, [], "true"):
                    self.daily["scrcpy"] = bad
                    self.invalid("scrcpy: expected a mapping")
                for key in ("screen_on", "stay_awake"):
                    for bad in (None, 1, "true"):
                        self.daily["scrcpy"] = {key: bad}
                        self.invalid(f"scrcpy.{key} must be a boolean")
                self.daily["scrcpy"] = {"unknown": True}
                self.invalid("unknown fields")
                self.daily["scrcpy"] = {"turn_screen_off": True}
                self.invalid("turn_screen_off.*obsolete.*screen_on")

    def test_paths_resources_and_sizes(self):
        plan = self.load()
        self.assertIsInstance(plan, Plan)
        image_id = plan.scenes["town"]["template"]
        self.assertEqual(image_id, plan.scenes["wild"]["template"])
        template = plan.resources[image_id]
        self.assertEqual(template["image"], self.image)
        np.testing.assert_array_equal(template["grayscale"], np.arange(30, dtype=np.uint8).reshape(5, 6))
        with self.assertRaises(ValueError):
            template["grayscale"][0, 0] = 255
        with self.assertRaises(ValueError):
            template["grayscale"].setflags(write=True)
        self.assertEqual(template["frame_rect"], {"x1": 0, "y1": 0, "x2": 6, "y2": 5})
        self.assertEqual(set(template), {"action", "image", "grayscale", "frame_rect", "frame_size", "device_size"})
        self.assertEqual(plan.switches[0]["actions"][0]["click"], {"target": image_id, "mode": "match"})
        task = plan.tasks[0]["task"]
        first = task["states"][task["initial_state"]]
        point = plan.resources[first["actions"][0]["click"]["target"]]
        self.assertEqual(point["device"], {"x": 199, "y": 399})
        self.assertEqual(set(point), {"action", "device", "frame_size", "device_size"})
        self.assertIsNot(point["frame_size"], template["frame_size"])
        self.assertEqual(first["retry"], {"count": 2, "interval": 1, "require_scene": "wild"})
        self.assertEqual((first["timeout"], first["on_timeout"]), (10, "uncertain"))
        self.assertEqual(task["start_scenes"], ["wild"])
        snapshot = SimpleNamespace(frame_size=(100, 200), device_size=(200, 400))
        validate_sizes(plan, snapshot)
        snapshot.frame_size = (200, 100)
        with self.assertRaisesRegex(ValueError, "frame_size"):
            validate_sizes(plan, snapshot)
        snapshot.frame_size = (100, 200)
        snapshot.device_size = (200, 401)
        with self.assertRaisesRegex(ValueError, "device_size"):
            validate_sizes(plan, snapshot)
        Detector(plan.resources, plan.scenes)

        def assert_same_resources(other):
            self.assertEqual(other.resources.keys(), plan.resources.keys())
            for name, record in plan.resources.items():
                actual = other.resources[name]
                self.assertEqual({k: v for k, v in actual.items() if k != "grayscale"},
                                 {k: v for k, v in record.items() if k != "grayscale"})
                if "grayscale" in record:
                    np.testing.assert_array_equal(actual["grayscale"], record["grayscale"])

        with patch("automation.config.PROJECT_HOME", self.root), \
                patch("automation.config.Path.cwd", side_effect=AssertionError):
            assert_same_resources(load_plan("lists/daily.yaml"))
        assert_same_resources(load_plan(self.list_path, home=self.root))
        self.daily["common"] = str(self.common_path)
        self.daily["tasks"][0]["file"] = str(self.task_path)
        self.common["scenes"]["town"]["image"] = str(self.image)
        assert_same_resources(self.load())

    def test_default_home_from_another_directory(self):
        original = Path.cwd()
        try:
            os.chdir(self.root)
            with patch("automation.config._read_yaml", side_effect=ValueError("stop")) as read:
                with self.assertRaisesRegex(ValueError, "stop"):
                    load_plan("lists/daily.yaml")
            read.assert_called_once_with(PROJECT_HOME / "lists" / "daily.yaml")
        finally:
            os.chdir(original)

    @unittest.skipUnless(sys.platform == "win32", "Windows relative path semantics")
    def test_windows_ambiguous_paths_are_rejected(self):
        for value in (r"C:relative.yaml", "C:", r"\root-relative.yaml", "/root-relative.yaml"):
            with self.subTest(value=value, field="list"):
                with self.assertRaisesRegex(ValueError, "Invalid list path"):
                    load_plan(value, home=self.root)
            for field in ("common", "task", "image"):
                with self.subTest(value=value, field=field):
                    daily, common = copy.deepcopy(self.daily), copy.deepcopy(self.common)
                    if field == "common":
                        self.daily["common"] = value
                    elif field == "task":
                        self.daily["tasks"][0]["file"] = value
                    else:
                        self.common["scenes"]["town"]["image"] = value
                    try:
                        self.invalid("invalid path")
                    finally:
                        self.daily, self.common = daily, common

    def test_cover_rejects_identical_grayscale_pixels_direct_and_transitive(self):
        self.common["scenes"]["popup"]["image"] = "tasks/templates/target.png"
        self.invalid("popup.covers: same grayscale image.*town")

        duplicate = self.root / "tasks" / "templates" / "duplicate.bmp"
        duplicate.write_bytes(cv2.imencode(".bmp", np.arange(30, dtype=np.uint8).reshape(5, 6))[1].tobytes())
        self.common["scenes"]["popup"]["image"] = "tasks/templates/duplicate.bmp"
        self.invalid("popup.covers: same grayscale image.*town")

        self.common["scenes"]["middle"] = {"type": "sub", "image": "tasks/templates/popup.png", "covers": ["town"]}
        self.common["scenes"]["popup"]["covers"] = ["middle"]
        self.invalid("popup.covers: same grayscale image.*town")
        self.common["scenes"]["popup"]["image"] = "tasks/templates/popup.png"
        self.invalid("popup.covers: same grayscale image.*middle")
        different = self.root / "tasks" / "templates" / "different.png"
        different.write_bytes(cv2.imencode(".png", np.arange(30, dtype=np.uint8).reshape(6, 5))[1].tobytes())
        self.common["scenes"]["popup"]["image"] = "tasks/templates/different.png"
        self.load()
        self.common["scenes"]["popup"]["image"] = "tasks/templates/target.png"
        self.common["scenes"]["popup"]["covers"] = []
        self.load()

    def test_sequence_and_nested_branches(self):
        self.task["steps"] = [{"detect": {"branches": [
            {"when": {"scene": "wild"}, "steps": [
                {"back": True, "require_scene": "wild"},
                {"detect": {"branches": [{"when": {"image": "tasks/templates/target.png"}, "result": "failed", "reason": "bad"},
                                          {"when": {"scene": "town"}, "steps": [{"wait": 1}]}],
                            "timeout": 2, "on_timeout": "failed"}}]},
            {"when": {"scene": "popup"}, "result": "verified", "reason": "manual"}],
            "timeout": 3, "on_timeout": "uncertain"}}, {"wait": 4}]
        task = self.load().tasks[0]["task"]
        states = task["states"]
        detect = states[task["initial_state"]]
        back = states[detect["transitions"][0]["to"]]
        nested = states[back["next"]]
        nested_wait = states[nested["transitions"][1]["to"]]
        final_wait = states[nested_wait["next"]]
        self.assertEqual(final_wait["actions"], [{"wait": 4}])
        self.assertEqual(states[final_wait["next"]], {"result": "unverified"})
        self.assertEqual(states[detect["transitions"][1]["to"]], {"result": "verified", "reason": "manual"})
        self.task["result"] = "verified"
        self.invalid("verified result requires a final expect or wait_scene")
        self.task["steps"].append({"wait_scene": {"scenes": ["town"], "timeout": 1}})
        self.assertIn({"result": "verified"}, self.load().tasks[0]["task"]["states"].values())

    def test_action_and_wait_compilation(self):
        self.task["steps"] = [
            {"click_image": "tasks/templates/target.png", "require_scene": "wild", "expect": {"image": "tasks/templates/target.png"},
             "timeout": 3, "on_timeout": "failed", "retry": {"count": 100, "interval": 0.1}},
            {"wait_scene": {"scenes": ["town"], "timeout": 5}},
            {"wait": 0},
        ]
        plan = self.load()
        task = plan.tasks[0]["task"]
        states = task["states"]
        first = states[task["initial_state"]]
        image_id = plan.scenes["town"]["template"]
        self.assertEqual(first["actions"], [{"click": {"target": image_id, "mode": "match"}}])
        self.assertEqual(first["transitions"][0]["when"], {"target": image_id})
        self.assertEqual((first["timeout"], first["on_timeout"]), (3, "failed"))
        detection = states[first["transitions"][0]["to"]]
        self.assertEqual(detection["actions"], [])
        self.assertEqual(detection["transitions"][0]["when"], {"scene": "town"})
        self.assertEqual((detection["timeout"], detection["on_timeout"]), (5, "uncertain"))
        waiting = states[detection["transitions"][0]["to"]]
        self.assertEqual(waiting["actions"], [{"wait": 0}])
        self.assertEqual(states[waiting["next"]], {"result": "unverified"})

    def test_image_errors_and_downscaling(self):
        self.common["device"]["device_size"] = [1, 1]
        self.task["steps"] = [{"wait": 0}]
        plan = self.load()
        template = plan.resources[plan.scenes["town"]["template"]]
        self.assertEqual(template["frame_rect"], {"x1": 0, "y1": 0, "x2": 6, "y2": 5})
        self.assertEqual(template["device_size"], {"width": 1, "height": 1})
        self.write_image(np.zeros((5, 6), dtype=np.uint8))
        self.invalid("constant grayscale")
        self.write_image(np.arange(10000, dtype=np.uint8).reshape(100, 100))
        self.invalid("smaller than frame")
        self.image.write_bytes(b"not an image")
        self.invalid("image")
        self.image.unlink()
        self.invalid("image")

    def test_click_validation_requires_resource_matching_mode(self):
        plan = self.load()
        template = plan.scenes["town"]["template"]
        task = plan.tasks[0]["task"]
        point = task["states"][task["initial_state"]]["actions"][0]["click"]["target"]
        for target, mode in ((template, "fixed"), (point, "match")):
            with self.subTest(target=target, mode=mode), self.assertRaisesRegex(ValueError, "wrong type"):
                _actions([{"click": {"target": target, "mode": mode}}], plan.resources, plan.scenes, "test")

    def test_scene_validation(self):
        original = copy.deepcopy(self.common)
        wait_town = {"wait_scene": {"scenes": ["town"], "timeout": 1}}
        cases = [
            (lambda c: c["scenes"]["town"].update(type="sub"), "town and wild"),
            (lambda c: c["scenes"]["town"].update(threshold=1.1), "threshold"),
            (lambda c: c["scenes"]["town"].update(region=[0, 0, 5, 5]), "region"),
            (lambda c: c["scenes"]["popup"].update(covers=["popup"]), "cycle"),
            (lambda c: c["scenes"]["popup"].update(covers=["missing"]), "covers"),
            (lambda c: c["scenes"]["popup"].update(exit=[{"back": True}, {"back": True}, wait_town]), "separated"),
            (lambda c: c["scenes"]["popup"].update(exit=[wait_town]), "side effect"),
            (lambda c: c["switches"][0]["steps"][-1]["wait_scene"].update(scenes=["town"]), "target scene"),
            (lambda c: c["switches"].append(copy.deepcopy(c["switches"][0])), "pair"),
            (lambda c: c["switches"][0]["steps"][0].update(extra=1), "single-key"),
        ]
        for mutate, error in cases:
            with self.subTest(error=error):
                self.common = copy.deepcopy(original)
                mutate(self.common)
                self.invalid(error)

    def test_invalid_steps(self):
        cases = [
            ({"click": [200, 0], "require_scene": "wild"}, "point"),
            ({"click": [True, 0], "require_scene": "wild"}, "point"),
            ({"click": [0, 0]}, "require_scene"),
            ({"click": [0, 0], "require_scene": "missing"}, "require_scene"),
            ({"swipe": {"from": [0, 0], "to": [1, 1], "duration": 1e308}, "require_scene": "wild"}, "duration"),
            ({"back": 1, "require_scene": "wild"}, "back"),
            ({"wait": -1}, "finite"),
            ({"wait": 1, "expect": {"scene": "wild"}}, "wait cannot"),
            ({"back": True, "require_scene": "wild", "timeout": 1}, "require expect"),
            ({"back": True, "require_scene": "wild", "expect": {"scene": "town"}, "timeout": 0}, "timeout"),
            ({"back": True, "require_scene": "wild", "expect": {"scene": "town"}, "on_timeout": "verified"}, "on_timeout"),
            ({"back": True, "require_scene": "wild", "expect": {"scene": "town"}, "retry": {"count": 101, "interval": 1}}, "retry"),
            ({"back": True, "require_scene": "wild", "expect": {"scene": "town"}, "retry": {"count": True, "interval": 1}}, "retry"),
            ({"back": True, "require_scene": "wild", "expect": {"scene": "town"}, "retry": {"count": 1, "interval": 0}}, "retry"),
            ({"wait_scene": {"scenes": ["town", "wild"], "timeout": 1}}, "exactly one"),
            ({"wait_scene": {"scenes": ["town"], "timeout": 0}}, "timeout"),
            ({"back": True, "require_scene": "wild", "unexpected": 1}, "unknown fields"),
        ]
        for step, error in cases:
            with self.subTest(step=step):
                self.task["steps"] = [step]
                self.invalid(error)

    def test_branch_validation_and_duplicate_conditions(self):
        detect = {"branches": [{"when": {"scene": "town"}, "result": "failed"}], "timeout": 1, "on_timeout": "failed"}
        cases = [
            (lambda d: d.update(branches=[]), "nonempty"),
            (lambda d: d["branches"].append({"when": {"scene": "town"}, "result": "verified"}), "duplicate when"),
            (lambda d: d.update(branches=[{"when": {"image": "tasks/templates/target.png"}, "result": "failed"},
                                          {"when": {"image": "tasks/templates/../templates/target.png"}, "result": "failed"}]), "duplicate when"),
            (lambda d: d["branches"][0].update(steps=[]), "exactly one"),
            (lambda d: d["branches"].__setitem__(0, {"when": {"scene": "town"}, "steps": []}), "nonempty"),
            (lambda d: d["branches"][0].update(reason=2), "reason"),
            (lambda d: d["branches"][0].update(result="bad"), "result"),
            (lambda d: d["branches"][0].update(when={"scene": "town", "image": "tasks/templates/target.png"}), "exactly one"),
            (lambda d: d.update(on_timeout="verified"), "on_timeout"),
            (lambda d: d.update(timeout=float("inf")), "timeout"),
        ]
        for mutate, error in cases:
            with self.subTest(error=error):
                value = copy.deepcopy(detect)
                mutate(value)
                self.task["steps"] = [{"detect": value}]
                self.invalid(error)

    def test_nonempty_tasks_and_top_level_steps(self):
        for value in ([], None, {}):
            with self.subTest(field="tasks", value=value):
                original = self.daily["tasks"]
                self.daily["tasks"] = value
                self.invalid("tasks must be a nonempty list")
                self.daily["tasks"] = original
        for result in (None, "unverified", "verified"):
            for value in ([], None, {}):
                with self.subTest(result=result, steps=value):
                    self.task["steps"] = value
                    if result is None:
                        self.task.pop("result", None)
                    else:
                        self.task["result"] = result
                    self.invalid("steps: expected a nonempty list")

    def test_root_verified_requires_final_evidence(self):
        self.task["result"] = "verified"
        click = {"click": [0, 0], "require_scene": "wild"}
        expect = {**click, "expect": {"scene": "wild"}}
        image_expect = {"click_image": "tasks/templates/target.png", "require_scene": "wild",
                        "expect": {"image": "tasks/templates/target.png"}}
        wait_scene = {"wait_scene": {"scenes": ["town"], "timeout": 1}}
        detect = {"detect": {"branches": [{"when": {"scene": "town"}, "result": "verified"}],
                             "timeout": 1, "on_timeout": "failed"}}
        for steps in ([{"wait": 1}], [click], [expect, {"wait": 1}],
                      [wait_scene, click], [detect], [detect, {"wait": 1}]):
            with self.subTest(steps=steps):
                self.task["steps"] = steps
                self.invalid("verified result requires a final expect or wait_scene")
        for steps in ([expect], [image_expect], [wait_scene], [detect, expect], [detect, wait_scene]):
            with self.subTest(steps=steps):
                self.task["steps"] = steps
                task = self.load().tasks[0]["task"]
                self.assertIn({"result": "verified"}, task["states"].values())
        # suspended is a branch-level result; the root still only allows verified/unverified.
        self.task["result"] = "unverified"
        self.task["steps"] = [{"detect": {"branches": [
            {"when": {"scene": "town"}, "result": "suspended", "reason": "no queue"}],
            "timeout": 1, "on_timeout": "uncertain"}}]
        task = self.load().tasks[0]["task"]
        self.assertIn({"result": "suspended", "reason": "no queue"}, task["states"].values())
        self.task["result"] = "suspended"
        self.invalid("result must be verified or unverified")
        for step in ({**click, "expect": {"scene": "missing"}},
                     {"wait_scene": {"scenes": ["missing"], "timeout": 1}},
                     {"wait": 1, "expect": {"scene": "wild"}}):
            with self.subTest(invalid_evidence=step):
                self.task["steps"] = [step]
                with self.assertRaises(ValueError):
                    self.load()

    def test_branch_verified_condition_and_recursive_steps(self):
        branch = {"when": {"scene": "town"}, "result": "verified"}
        detect = {"detect": {"branches": [branch], "timeout": 1, "on_timeout": "failed"}}
        self.task["steps"] = [detect]
        task = self.load().tasks[0]["task"]
        state = task["states"][task["initial_state"]]
        self.assertEqual(state["transitions"][0]["when"], {"scene": "town"})
        self.assertEqual(task["states"][state["transitions"][0]["to"]], {"result": "verified"})
        branch["when"] = {"image": "tasks/templates/target.png"}
        self.load()
        branch["when"] = {"scene": "missing"}
        self.invalid("unknown or wrong type")
        branch.clear()
        branch.update(when={"scene": "town"}, steps=[{"detect": {"branches": [
            {"when": {"scene": "wild"}, "steps": []}], "timeout": 1, "on_timeout": "failed"}}])
        self.invalid("nonempty")

    def test_invalid_new_schema_never_connects(self):
        import app

        args = SimpleNamespace(task_list="lists/daily.yaml", home=self.root)
        cases = [([], [{"wait": 1}], "unverified"),
                 (self.daily["tasks"], [], "unverified"),
                 (self.daily["tasks"], [{"wait": 1}], "verified")]
        for tasks, steps, result in cases:
            with self.subTest(tasks=tasks, steps=steps, result=result):
                self.daily["tasks"] = tasks
                self.task.update(steps=steps, result=result)
                self.write_all()
                with patch.object(app, "ScrcpySession") as session, \
                        patch.object(app, "load_server_jar") as server, \
                        self.assertLogs("app", level="ERROR"):
                    self.assertEqual(app.run_automation_mode(args), 1)
                    session.assert_not_called()
                    server.assert_not_called()

    def test_runtime_conflicts(self):
        self.task["steps"] = [{"detect": {"branches": [
            {"when": {"scene": "town"}, "result": "verified"},
            {"when": {"image": "tasks/templates/target.png"}, "result": "failed"}],
            "timeout": 1, "on_timeout": "failed"}}]
        plan = self.load()
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[10:15, 10:16] = np.arange(30, dtype=np.uint8).reshape(5, 6, 1)
        with self.assertRaises(SceneConflict):
            Detector(plan.resources, plan.scenes).detect(SimpleNamespace(frame=frame))

        class FakeRuntime:
            detector = SimpleNamespace(match=lambda snapshot, target: object())
            poll = 0.01

            def snapshot(self, after_id=None, timeout=None):
                return SimpleNamespace(frame_id=1)

            def observe(self, snapshot):
                return snapshot, "town"

        result = run_task(FakeRuntime(), plan.tasks[0]["task"])
        self.assertEqual((result.result, result.reason), ("uncertain", "conflicting transitions"))

    def test_schema_paths_and_yaml(self):
        self.daily["tasks"][0]["file"] = "missing.yaml"
        self.invalid("Cannot load YAML")
        self.daily["tasks"][0]["file"] = "tasks/intel.yaml"
        self.daily["common"] = "missing.yaml"
        self.invalid("Cannot load YAML")
        self.daily["common"] = "tasks/common.yaml"
        self.daily["tasks"][0]["on_failure"] = "continue"
        self.invalid("on_failure")
        self.daily = {"resources": "old.yaml", "scenes": "old.yaml", "tasks": []}
        self.invalid("unknown fields")
        self.daily = {"common": "tasks/common.yaml", "tasks": [{"file": "tasks/intel.yaml", "on_failure": "stop"}]}
        self.task = {"task": "old", "start_scenes": ["town"], "initial_state": "x", "states": {}}
        self.invalid("unknown fields")
        self.common = {"marker_file": "old.yaml", "resources": {}}
        self.invalid("unknown fields")
        self.write_all()
        self.list_path.write_text("common: a\ncommon: b\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            load_plan(self.list_path, home=self.root)
        self.list_path.write_text("!!python/object/apply:os.system ['exit 1']", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot load YAML"):
            load_plan(self.list_path, home=self.root)

    def test_invalid_sizes_and_task_fields(self):
        for value in (None, [1], [1, True], [0, 1], [1.0, 2], {"width": 1, "height": 2}):
            with self.subTest(value=value):
                self.common["device"]["frame_size"] = value
                self.invalid("frame_size")
        self.common["device"]["frame_size"] = [100, 200]
        self.task["start"] = ["popup"]
        self.invalid("main")
        self.task["start"] = "wild"
        self.task["result"] = "failed"
        self.invalid("result")

    def loop_daily(self, loop):
        self.daily = {"common": "tasks/common.yaml", "loop": loop}

    def test_loop_plan_loads_with_compiled_task(self):
        self.loop_daily({"count": 2, "file": "tasks/intel.yaml", "wait": 1,
                         "success": {"result": "unverified"},
                         "retry": [{"result": "suspended", "wait": 5}],
                         "max_retries": 3})
        plan = self.load()
        self.assertEqual(plan.tasks, [])
        self.assertEqual(plan.loop["count"], 2)
        self.assertEqual(plan.loop["on_failure"], "stop")
        self.assertEqual(plan.loop["wait"], 1)
        self.assertEqual(plan.loop["success"], {"result": "unverified", "reason": ""})
        self.assertEqual(plan.loop["retry"], [{"result": "suspended", "wait": 5}])
        self.assertEqual(plan.loop["max_retries"], 3)
        self.assertEqual(plan.loop["task"]["task"], "intel")

    def test_loop_and_tasks_are_mutually_exclusive(self):
        self.loop_daily({"count": 1, "file": "tasks/intel.yaml", "success": {"result": "unverified"}})
        self.daily["tasks"] = [{"file": "tasks/intel.yaml", "on_failure": "stop"}]
        self.invalid("exactly one")
        del self.daily["loop"]
        del self.daily["tasks"]
        self.invalid("exactly one")

    def test_loop_field_validation(self):
        def loop(**overrides):
            value = {"count": 2, "file": "tasks/intel.yaml",
                     "success": {"result": "unverified", "reason": ""},
                     "retry": [{"result": "suspended", "wait": 5}],
                     "max_retries": 3}
            value.update(overrides)
            return value
        self.loop_daily(loop(count=0))
        self.invalid("loop.count")
        self.loop_daily(loop(count=True))
        self.invalid("loop.count")
        self.loop_daily(loop(on_failure="continue"))
        self.invalid("loop.on_failure")
        self.loop_daily(loop(wait=-1))
        self.invalid("loop.wait")
        self.loop_daily(loop(success={"result": "failed"}))
        self.invalid("loop.success.result")
        self.loop_daily(loop(success={}))
        self.invalid("missing fields")
        self.loop_daily(loop(retry=[{"result": "suspended"}, {"result": "suspended"}]))
        self.invalid("duplicated retry result")
        self.loop_daily(loop(retry=[{"result": "failed"}]))
        self.invalid(r"loop.retry\[0\].result: must be unverified or suspended")
        self.loop_daily(loop(retry=[{}]))
        self.invalid("missing fields")
        self.loop_daily(loop(max_retries=0))
        self.invalid("loop.max_retries must be positive")
        self.loop_daily(loop(max_retries=-1))
        self.invalid("loop.max_retries")
        self.loop_daily(loop(extra=1))
        self.invalid("unknown fields")


if __name__ == "__main__":
    unittest.main()
