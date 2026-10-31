import unittest
from types import SimpleNamespace
from unittest.mock import patch

from automation.engine import TaskResult, run_task
from automation.runtime import FrameTimeout
from automation.vision import SceneConflict


CLICK = {"click": {"target": "button", "mode": "match"}}


def transition(kind, name, to="done"):
    return {"when": {kind: name}, "to": to}


def task(*, actions=None, transitions=None, timeout=0.3, on_timeout="failed", retry=None):
    state = {"actions": [CLICK] if actions is None else actions,
             "transitions": [transition("scene", "done")] if transitions is None else transitions,
             "timeout": timeout, "on_timeout": on_timeout, "require_scene": "town"}
    if retry is not None:
        state["retry"] = retry
    return {"task": "example", "initial_state": "start", "states": {
        "start": state, "done": {"result": "verified", "reason": "complete"}}}


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class FakeRuntime:
    poll = 0.1

    def __init__(self, clock, frames=(), execute_frame=0):
        self.clock = clock
        self.frames = list(frames)
        self.frame_id = 0
        self.execute_frame = execute_frame
        self.executions = []
        self.expected_scenes = []
        self.snapshots = []
        self.observed = []
        self.matches = []
        self.detector = self
        self.plan = SimpleNamespace(resources={}, scenes={})

    def snapshot(self, after_id=None, timeout=None):
        self.snapshots.append((after_id, timeout))
        self.frame_id = max(self.frame_id + 1, (after_id or 0) + 1)
        frame = self.frames.pop(0) if self.frames else ("town", set())
        return SimpleNamespace(frame_id=self.frame_id, scene=frame[0], targets=frame[1])

    def observe(self, snapshot=None):
        self.observed.append(snapshot)
        if isinstance(snapshot.scene, BaseException):
            raise snapshot.scene
        return snapshot, snapshot.scene

    def match(self, snapshot, name):
        self.matches.append((snapshot, name))
        return object() if name in snapshot.targets else None

    def execute(self, actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
        self.expected_scenes.append(expected_scene)
        if isinstance(self.execute_frame, BaseException):
            self.executions.append(actions)
            raise self.execute_frame
        if before_action is not None:
            frame = self.snapshot(after_id=after_id)
            _, scene = self.observe(frame)
            if not before_action(frame, scene):
                return frame.frame_id
            if scene != expected_scene:
                raise RuntimeError("scene changed")
        self.executions.append(actions)
        return self.execute_frame if self.execute_frame is None else max(self.frame_id, self.execute_frame)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.monotonic = patch("automation.engine.time.monotonic", self.clock.monotonic)
        self.sleep = patch("automation.engine.time.sleep", self.clock.sleep)
        self.monotonic.start()
        self.sleep.start()
        self.addCleanup(self.monotonic.stop)
        self.addCleanup(self.sleep.stop)

    def runtime(self, frames=(), execute_frame=0):
        return FakeRuntime(self.clock, frames, execute_frame)

    def test_verified_and_unverified_terminal_results(self):
        for result in ("verified", "unverified"):
            with self.subTest(result=result):
                config = task()
                config["states"]["done"]["result"] = result
                runtime = self.runtime([("town", set()), ("done", set())], execute_frame=10)
                self.assertEqual(run_task(runtime, config), TaskResult(result, "complete", "done"))
                self.assertEqual(runtime.executions, [[CLICK]])
                self.assertEqual(runtime.snapshots[0][0], 10)
                self.assertGreater(runtime.observed[0].frame_id, 10)
                self.assertEqual(runtime.snapshots[1][0], runtime.observed[0].frame_id)

    def test_target_and_scene_use_same_frame_and_conflict_even_same_destination(self):
        config = task(transitions=[transition("scene", "town"), transition("target", "button")])
        runtime = self.runtime([("town", {"button"})])
        result = run_task(runtime, config)
        self.assertEqual((result.result, result.state), ("uncertain", "start"))
        self.assertIn("conflict", result.reason)
        self.assertIs(runtime.matches[0][0], runtime.observed[0])
        self.assertEqual(len(runtime.executions), 1)

    def test_two_targets_conflict_and_no_match_can_be_retested_only_on_new_frames(self):
        config = task(transitions=[transition("target", "a"), transition("target", "b")])
        runtime = self.runtime([("town", set()), ("town", {"a", "b"})])
        self.assertEqual(run_task(runtime, config).result, "uncertain")
        self.assertEqual([item[1] for item in runtime.matches], ["a", "b", "a", "b"])
        self.assertGreater(runtime.observed[1].frame_id, runtime.observed[0].frame_id)

    def test_timeout_without_retry_never_reexecutes(self):
        for outcome in ("failed", "uncertain"):
            with self.subTest(outcome=outcome):
                runtime = self.runtime()
                result = run_task(runtime, task(on_timeout=outcome))
                self.assertEqual(result, TaskResult(outcome, "state timeout", "start"))
                self.assertEqual(len(runtime.executions), 1)
                self.assertTrue(all(timeout >= 0 for _, timeout in runtime.snapshots))

    def test_deadline_checks_last_new_frame_without_retry_or_recovery(self):
        for retry in (None, {"count": 1, "interval": 0.5, "require_scene": "town"}):
            for oversleep, expected in ((0, "verified"), (0.01, "uncertain")):
                with self.subTest(retry=retry, oversleep=oversleep):
                    self.clock.now = 0
                    runtime = self.runtime([("town", set()), ("done", set())])
                    original_sleep = self.clock.sleep

                    def sleep(duration):
                        original_sleep(duration + oversleep)

                    with patch("automation.engine.time.sleep", sleep):
                        result = run_task(runtime, task(timeout=0.1, retry=retry))
                    self.assertEqual(result.result, expected)
                    self.assertEqual(len(runtime.executions), 1)
                    self.assertEqual(runtime.snapshots[-1], (1, 0))
                    self.assertEqual([frame.frame_id for frame in runtime.observed], [1, 2])

    def test_deadline_without_new_frame_stops_without_recovery(self):
        for retry in (None, {"count": 1, "interval": 0.5, "require_scene": "town"}):
            with self.subTest(retry=retry):
                self.clock.now = 0
                runtime = self.runtime()
                original_snapshot = runtime.snapshot

                def snapshot(after_id=None, timeout=None):
                    if timeout == 0:
                        raise FrameTimeout("no fresh frame")
                    return original_snapshot(after_id=after_id, timeout=timeout)

                runtime.snapshot = snapshot
                result = run_task(runtime, task(timeout=0.1, retry=retry))
                self.assertEqual(result.result, "uncertain")
                self.assertEqual(len(runtime.observed), 1)
                self.assertEqual(len(runtime.executions), 1)
                self.assertNotIn(0.5, self.clock.sleeps)

    def test_deadline_frame_detects_conflicting_transitions_on_same_frame(self):
        runtime = self.runtime([("town", set()), ("town", {"a", "b"})])
        result = run_task(runtime, task(timeout=0.1, transitions=[
            transition("target", "a"), transition("target", "b")]))
        self.assertEqual(result.result, "uncertain")
        self.assertIn("conflicting", result.reason)
        self.assertIs(runtime.matches[-1][0], runtime.matches[-2][0])
        self.assertEqual(len(runtime.executions), 1)

    def test_unknown_scene_on_retry_precondition_cannot_branch(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        runtime = self.runtime([("town", set())] * 3 + [(None, {"button"})] * 10)
        result = run_task(runtime, task(timeout=0.2, retry=retry,
                                        transitions=[transition("target", "button")]))
        self.assertEqual(result.result, "uncertain")
        self.assertEqual(len(runtime.executions), 1)
        self.assertTrue(all(frame.scene is not None for frame, _ in runtime.matches))

    def test_unknown_scene_on_retry_action_frame_cannot_branch(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        runtime = self.runtime([("town", set())] * 4 + [(None, {"button"})] * 10)
        result = run_task(runtime, task(timeout=0.2, retry=retry,
                                        transitions=[transition("target", "button")]))
        self.assertEqual(result.result, "uncertain")
        self.assertEqual(len(runtime.executions), 1)
        self.assertTrue(all(frame.scene is not None for frame, _ in runtime.matches))

    def test_loading_frames_then_transition(self):
        runtime = self.runtime([(None, {"button"}), (None, {"button"}), ("town", {"button"})])
        result = run_task(runtime, task(transitions=[transition("target", "button")]))
        self.assertEqual(result.result, "verified")
        self.assertEqual(len(runtime.executions), 1)
        self.assertEqual([frame.frame_id for frame, _ in runtime.matches], [3])

    def test_unknown_scene_at_deadline_never_fails_or_retries(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        for configured_retry in (None, retry):
            with self.subTest(retry=configured_retry):
                self.clock.now = 0
                runtime = self.runtime([(None, {"button"})] * 10)
                result = run_task(runtime, task(timeout=0.2, on_timeout="failed", retry=configured_retry,
                                                transitions=[transition("target", "button")]))
                self.assertEqual(result.result, "uncertain")
                self.assertEqual(runtime.matches, [])
                self.assertEqual(len(runtime.executions), 1)

    def test_retry_action_does_not_reuse_precondition_frame(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        runtime = self.runtime()
        original_execute = runtime.execute

        def execute(actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
            if before_action is not None:
                frame = runtime.observed[-1]
                self.assertFalse(before_action(frame, frame.scene))
                return frame.frame_id
            return original_execute(actions, expected_scene=expected_scene)

        runtime.execute = execute
        result = run_task(runtime, task(timeout=0.2, retry=retry,
                                        transitions=[transition("target", "button")]))
        self.assertEqual(result.result, "uncertain")
        self.assertEqual(len(runtime.executions), 1)
        self.assertEqual(len(runtime.matches), len(runtime.observed))

    def test_empty_actions_use_current_frame_without_executing(self):
        runtime = self.runtime([("done", set())])
        self.assertEqual(run_task(runtime, task(actions=[])).result, "verified")
        self.assertEqual(runtime.executions, [])
        self.assertIsNone(runtime.snapshots[0][0])

    def test_next_executes_wait_and_preserves_watermark_through_empty_state(self):
        config = task(actions=[])
        config["states"]["start"] = {"actions": [{"wait": 1}], "next": "middle"}
        config["states"]["middle"] = {"actions": [], "transitions": [transition("scene", "done")],
                                     "timeout": 1, "on_timeout": "failed"}
        runtime = self.runtime([("done", set())], execute_frame=8)
        self.assertEqual(run_task(runtime, config).result, "verified")
        self.assertEqual(runtime.executions, [[{"wait": 1}]])
        self.assertEqual(runtime.snapshots[0][0], 8)

    def test_none_action_watermark_requires_an_extra_fresh_frame(self):
        runtime = self.runtime([("done", set()), ("done", set())], execute_frame=None)
        self.assertEqual(run_task(runtime, task()).result, "verified")
        self.assertEqual(len(runtime.snapshots), 2)
        self.assertEqual(runtime.snapshots[1][0], 1)
        self.assertEqual(runtime.observed[0].frame_id, 2)

    def test_retry_only_after_timeout_interval_and_matching_scene(self):
        retry = {"count": 2, "interval": 0.5, "require_scene": "town"}
        runtime = self.runtime()
        result = run_task(runtime, task(timeout=0.2, on_timeout="uncertain", retry=retry))
        self.assertEqual(result.result, "uncertain")
        self.assertEqual(len(runtime.executions), 3)
        self.assertEqual(runtime.expected_scenes, ["town", "town", "town"])
        self.assertEqual(self.clock.sleeps.count(0.5), 2)
        self.assertAlmostEqual(self.clock.now, 1.6)

    def test_delayed_transition_precedes_retry_even_when_scene_changes(self):
        retry = {"count": 1, "interval": 0.5, "require_scene": "town"}
        runtime = self.runtime([("town", set()), ("town", set()), ("town", set()), ("done", set())])
        self.assertEqual(run_task(runtime, task(timeout=0.2, retry=retry)).result, "verified")
        self.assertEqual(len(runtime.executions), 1)
        self.assertIn(0.5, self.clock.sleeps)

    def test_retry_scene_mismatch_or_unknown_stops_without_another_action(self):
        retry = {"count": 1, "interval": 0.5, "require_scene": "town"}
        for scene in (None, "other"):
            with self.subTest(scene=scene):
                runtime = self.runtime([("town", set())] * 3 + [(scene, set())] * 10)
                result = run_task(runtime, task(timeout=0.2, on_timeout="uncertain", retry=retry))
                self.assertEqual(result.result, "uncertain")
                self.assertIn("scene", result.reason)
                self.assertEqual(len(runtime.executions), 1)

    def test_unsafe_retries_are_rejected_before_any_action(self):
        retry = {"count": 1, "interval": 0.5, "require_scene": "town"}
        for actions in ([], [CLICK, CLICK], [CLICK, {"wait": 1}], [{"wait": 1}]):
            with self.subTest(actions=actions):
                runtime = self.runtime()
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    run_task(runtime, task(actions=actions, retry=retry))
                self.assertEqual(runtime.executions, [])

    def test_observation_and_execution_exceptions_propagate(self):
        for error in (SceneConflict("conflict"), OSError("detector failed"), KeyboardInterrupt()):
            with self.subTest(error=error):
                runtime = self.runtime([(error, set())])
                with self.assertRaises(type(error)):
                    run_task(runtime, task())
                self.assertEqual(len(runtime.executions), 1)

    def test_target_transition_with_unknown_scene_never_branches(self):
        for retry in (None, {"count": 1, "interval": 0.1, "require_scene": "town"}):
            for destination in ("done", "middle"):
                with self.subTest(retry=retry, destination=destination):
                    config = task(transitions=[transition("target", "button", destination)], retry=retry)
                    config["states"]["middle"] = {"actions": [CLICK], "next": "done", "require_scene": "town"}
                    runtime = self.runtime([(None, {"button"})] * 10)
                    self.assertEqual(run_task(runtime, config).result, "uncertain")
                    self.assertEqual(runtime.matches, [])
                    self.assertEqual(len(runtime.executions), 1)

    def test_snapshot_and_detector_errors_propagate(self):
        runtime = self.runtime()
        runtime.snapshot = lambda **kwargs: (_ for _ in ()).throw(TimeoutError("frame unavailable"))
        with self.assertRaisesRegex(TimeoutError, "frame unavailable"):
            run_task(runtime, task())
        runtime = self.runtime([("town", set())])
        runtime.match = lambda *args: (_ for _ in ()).throw(OSError("match failed"))
        with self.assertRaisesRegex(OSError, "match failed"):
            run_task(runtime, task(transitions=[transition("target", "button")]))
        self.assertEqual(len(runtime.executions), 1)
        for error in (OSError("touch failed"), KeyboardInterrupt()):
            with self.subTest(error=error):
                runtime = self.runtime(execute_frame=error)
                with self.assertRaises(type(error)):
                    run_task(runtime, task())
                self.assertEqual(len(runtime.executions), 1)

    def test_stale_frames_and_invalid_next_are_rejected(self):
        runtime = self.runtime(execute_frame=2)
        runtime.snapshot = lambda after_id=None, timeout=None: SimpleNamespace(frame_id=2)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            run_task(runtime, task())
        config = task()
        config["states"]["start"]["next"] = "done"
        runtime = self.runtime()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_task(runtime, config)
        self.assertEqual(runtime.executions, [])

    def test_cycle_is_bounded_even_if_loader_did_not_validate(self):
        config = task()
        config["states"]["start"] = {"actions": [], "next": "start"}
        with self.assertRaisesRegex(ValueError, "cycle"):
            run_task(self.runtime(), config)

    def test_frame_timeout_propagates_for_scene_and_target_transitions(self):
        for kind in ("scene", "target"):
            for retry_count in (0, 2):
                with self.subTest(kind=kind, retries=retry_count):
                    self.clock.now = 0
                    runtime = self.runtime()

                    def timeout(after_id=None, timeout=None):
                        runtime.snapshots.append((after_id, timeout))
                        self.clock.now += timeout
                        raise FrameTimeout("no fresh frame")

                    runtime.snapshot = timeout
                    retry = {"count": retry_count, "interval": 0.5, "require_scene": "town"} if retry_count else None
                    with self.assertRaises(FrameTimeout):
                        run_task(runtime, task(transitions=[transition(kind, "done")], timeout=0.2,
                                               on_timeout="failed", retry=retry))
                    self.assertEqual(len(runtime.executions), 1)
                    self.assertEqual(len(runtime.snapshots), 1)
                    self.assertAlmostEqual(self.clock.now, 0.2)

    def test_timeout_from_side_effect_propagates_even_if_frame_timeout(self):
        runtime = self.runtime(execute_frame=FrameTimeout("after control"))
        with self.assertRaisesRegex(FrameTimeout, "after control"):
            run_task(runtime, task())
        self.assertEqual(len(runtime.executions), 1)

    def test_retry_precondition_and_next_scene_are_enforced(self):
        config = task(retry={"count": 1, "interval": 0.1, "require_scene": "other"})
        runtime = self.runtime()
        with self.assertRaisesRegex(ValueError, "must match"):
            run_task(runtime, config)
        self.assertEqual(runtime.executions, [])
        config = task()
        config["states"]["start"] = {"actions": [CLICK], "next": "done"}
        with self.assertRaisesRegex(ValueError, "require_scene"):
            run_task(runtime, config)
        config["states"]["start"]["require_scene"] = "wild"
        self.assertEqual(run_task(runtime, config).result, "verified")
        self.assertEqual(runtime.expected_scenes, ["wild"])

    def test_retry_action_frame_checks_transition_and_conflict(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        for targets, expected in (({"a"}, "verified"), ({"a", "b"}, "uncertain")):
            with self.subTest(targets=targets):
                self.clock.now = 0
                runtime = self.runtime([("town", set()), ("town", set()), ("town", set()),
                                        ("town", set()), ("town", targets)])
                config = task(timeout=0.2, retry=retry,
                              transitions=[transition("target", "a"), transition("target", "b")])
                self.assertEqual(run_task(runtime, config).result, expected)
                self.assertEqual(len(runtime.executions), 1)

    def test_late_retry_action_frame_cannot_trigger_action_or_failed(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        for targets, reason in (({"a"}, "after deadline"), ({"a", "b"}, "conflicting")):
            with self.subTest(targets=targets):
                self.clock.now = 0
                runtime = self.runtime([("town", set()), ("town", set()), ("town", set()),
                                        ("town", set()), ("town", targets)])
                original_match = runtime.match

                def match(frame, name):
                    result = original_match(frame, name)
                    if frame.targets:
                        self.clock.now += 1
                    return result

                runtime.match = match
                config = task(timeout=0.2, retry=retry,
                              transitions=[transition("target", "a"), transition("target", "b")])
                result = run_task(runtime, config)
                self.assertEqual(result.result, "uncertain")
                self.assertIn(reason, result.reason)
                self.assertEqual(len(runtime.executions), 1)

    def test_retry_rejects_stale_actual_action_frame(self):
        retry = {"count": 1, "interval": 0.1, "require_scene": "town"}
        runtime = self.runtime()
        original_execute = runtime.execute

        def execute(actions, expected_scene=None, before_action=None, action_deadline=None, after_id=None):
            if before_action is not None:
                before_action(SimpleNamespace(frame_id=0, scene="town", targets=set()), "town")
            return original_execute(actions, expected_scene=expected_scene)

        runtime.execute = execute
        with self.assertRaisesRegex(RuntimeError, "stale action frame"):
            run_task(runtime, task(timeout=0.2, retry=retry))
        self.assertEqual(len(runtime.executions), 1)


if __name__ == "__main__":
    unittest.main()
