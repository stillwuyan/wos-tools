import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from automation.engine import TaskResult
from automation.runner import run_loop
from automation.runtime import Runtime


class SignallingEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


class CancellationTests(unittest.TestCase):
    def assert_wait_cancellable(self, operation, event):
        outcome = []

        def worker():
            try:
                operation()
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(event.waiting.wait(3))
        finally:
            event.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], KeyboardInterrupt)

    def test_long_task_wait_cancels_without_further_controls(self):
        event = SignallingEvent()
        session = Mock()
        runtime = Runtime(SimpleNamespace(), session, Mock(), cancel_event=event)
        self.assert_wait_cancellable(lambda: runtime.execute([{"wait": 60}]), event)
        session.controller.tap.assert_not_called()
        session.controller.back.assert_not_called()
        session.check_stream_health.assert_not_called()

    def test_loop_wait_cancels_before_next_attempt_or_recovery(self):
        event = SignallingEvent()
        runtime = Mock()
        loop = {"task": {"task": "beast", "start_scenes": ["wild"]}, "count": 2,
                "success": {"result": "verified", "reason": ""}, "retry": [],
                "wait": 60, "max_retries": 0, "on_failure": "stop"}
        with patch("automation.runner.run_task", return_value=TaskResult("verified", "", "done")) as task:
            self.assert_wait_cancellable(lambda: run_loop(runtime, loop, cancel_event=event), event)
        task.assert_called_once()
        runtime.prepare.assert_called_once()
        runtime.finish.assert_not_called()

    def test_cancel_during_detection_prevents_control(self):
        event = threading.Event()
        session = Mock()
        plan = SimpleNamespace(resources={"point": {"action": "point", "device": {"x": 1, "y": 2}}})
        runtime = Runtime(plan, session, Mock(), cancel_event=event)
        frame = SimpleNamespace(frame_id=1, decode_time=time.monotonic())

        def observe():
            event.set()
            return frame, "wild"

        with patch.object(runtime, "observe", side_effect=observe), self.assertRaises(KeyboardInterrupt):
            runtime.execute([{"click": {"target": "point", "mode": "fixed"}}])
        session.controller.tap.assert_not_called()
