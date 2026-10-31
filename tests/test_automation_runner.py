import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from automation.engine import TaskResult
from automation.runtime import FrameTimeout
from automation.runner import run_list, run_loop


def entry(name, policy="stop"):
    return {"name": name, "on_failure": policy,
            "task": {"task": name, "start_scenes": [name + "_main"]}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Mock()
        self.entries = [entry("first"), entry("second")]
        self.plan = SimpleNamespace(tasks=self.entries)
        self.calls = Mock()
        self.calls.attach_mock(self.runtime, "runtime")
        self.task_patcher = patch("automation.runner.run_task")
        self.run_task = self.task_patcher.start()
        self.addCleanup(self.task_patcher.stop)
        self.calls.attach_mock(self.run_task, "run_task")

    def result(self, result, state="end", reason="detail"):
        return TaskResult(result, reason, state)

    def test_success_and_unverified_run_in_order_and_finish(self):
        self.run_task.side_effect = [self.result("verified"), self.result("unverified")]
        with self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertTrue(run_list(self.runtime, self.plan))
        self.assertEqual(self.calls.mock_calls, [
            call.runtime.prepare(["first_main"]),
            call.run_task(self.runtime, self.entries[0]["task"]),
            call.runtime.prepare(["second_main"]),
            call.run_task(self.runtime, self.entries[1]["task"]),
            call.runtime.finish(),
        ])
        self.assertTrue(any("task=first state=end result=verified reason=detail" in s for s in logs.output))
        self.assertTrue(any("task=second state=end result=unverified" in s for s in logs.output))

    def test_failed_stop_never_recovers_or_continues(self):
        self.run_task.return_value = self.result("failed")
        self.assertFalse(run_list(self.runtime, self.plan))
        self.run_task.assert_called_once()
        self.runtime.finish.assert_not_called()
        self.runtime.prepare.assert_called_once_with(["first_main"])

    def test_failed_skip_recovers_before_next_prepare(self):
        self.entries[0]["on_failure"] = "skip"
        self.run_task.side_effect = [self.result("failed"), self.result("verified")]
        self.assertTrue(run_list(self.runtime, self.plan))
        self.assertEqual(self.calls.mock_calls, [
            call.runtime.prepare(["first_main"]),
            call.run_task(self.runtime, self.entries[0]["task"]),
            call.runtime.finish(),
            call.runtime.prepare(["second_main"]),
            call.run_task(self.runtime, self.entries[1]["task"]),
            call.runtime.finish(),
        ])

    def test_last_failed_skip_finishes_once(self):
        self.plan.tasks = [entry("last", "skip")]
        self.run_task.return_value = self.result("failed")
        self.assertTrue(run_list(self.runtime, self.plan))
        self.runtime.finish.assert_called_once_with()

    def test_uncertain_or_unknown_never_skips_or_recovers(self):
        self.entries[0]["on_failure"] = "skip"
        for result in ("uncertain", "unexpected"):
            with self.subTest(result=result):
                self.runtime.reset_mock()
                self.run_task.reset_mock()
                self.run_task.return_value = self.result(result)
                self.assertFalse(run_list(self.runtime, self.plan))
                self.runtime.finish.assert_not_called()
                self.run_task.assert_called_once()

    def test_skip_recovery_failure_stops_before_next_task(self):
        self.entries[0]["on_failure"] = "skip"
        self.run_task.return_value = self.result("failed")
        self.runtime.finish.side_effect = RuntimeError("recovery failed")
        self.assertFalse(run_list(self.runtime, self.plan))
        self.run_task.assert_called_once()
        self.runtime.prepare.assert_called_once()
        self.runtime.finish.assert_called_once()

    def test_final_recovery_failure_is_logged_separately(self):
        self.run_task.return_value = self.result("verified")
        self.runtime.finish.side_effect = RuntimeError("recovery failed")
        with self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertFalse(run_list(self.runtime, self.plan))
        self.assertTrue(any("result=verified" in s for s in logs.output))
        self.assertTrue(any("final recovery failed" in s for s in logs.output))
        self.assertEqual(self.run_task.call_count, 2)

    def test_empty_list_still_finishes(self):
        self.plan.tasks = []
        self.assertTrue(run_list(self.runtime, self.plan))
        self.runtime.finish.assert_called_once_with()
        self.run_task.assert_not_called()

    def test_prepare_silent_stream_stops_before_start_with_clean_warning(self):
        self.runtime.prepare.side_effect = FrameTimeout("Timed out waiting for a healthy new frame")
        with self.assertLogs("automation.runner", level="WARNING") as logs:
            self.assertFalse(run_list(self.runtime, self.plan))
        self.assertTrue(any("stopped before start: video stream silent" in s for s in logs.output))
        self.run_task.assert_not_called()
        self.runtime.finish.assert_not_called()

    def test_final_recovered_with_silent_stream_finishes_unverified(self):
        self.run_task.return_value = self.result("verified")
        self.runtime.finish.return_value = "video stream silent"
        with self.assertLogs("automation.runner", level="WARNING") as logs:
            self.assertTrue(run_list(self.runtime, self.plan))
        self.assertTrue(any("final recovery was not verified" in s for s in logs.output))
        self.assertNotIn("recovered to a main scene", " ".join(logs.output))

    def test_exceptions_are_never_skipped_or_followed_by_recovery(self):
        self.entries[0]["on_failure"] = "skip"
        for source in ("prepare", "task"):
            with self.subTest(source=source):
                self.runtime.reset_mock(side_effect=True)
                self.run_task.reset_mock(side_effect=True)
                if source == "prepare":
                    self.runtime.prepare.side_effect = OSError("stream disconnected")
                else:
                    self.run_task.side_effect = OSError("stream disconnected")
                self.assertFalse(run_list(self.runtime, self.plan))
                self.runtime.finish.assert_not_called()
                self.runtime.prepare.assert_called_once()

    def test_interrupt_propagates_without_further_recovery(self):
        self.entries[0]["on_failure"] = "skip"
        for source in ("prepare", "task", "skip_finish", "final_finish"):
            with self.subTest(source=source):
                self.runtime.reset_mock(side_effect=True)
                self.run_task.reset_mock(side_effect=True)
                self.run_task.return_value = self.result("failed" if source == "skip_finish" else "verified")
                if source == "prepare":
                    self.runtime.prepare.side_effect = KeyboardInterrupt
                elif source == "task":
                    self.run_task.side_effect = KeyboardInterrupt
                else:
                    self.runtime.finish.side_effect = KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    run_list(self.runtime, self.plan)
                self.assertEqual(self.runtime.finish.call_count, int(source.endswith("finish")))
                if source != "final_finish":
                    self.runtime.prepare.assert_called_once()


class RunnerIntegrationTests(unittest.TestCase):
    def test_real_engine_with_fake_runtime_and_skip(self):
        class FakeRuntime:
            def __init__(self):
                self.events = []

            def prepare(self, scenes):
                self.events.append(("prepare", scenes))

            def finish(self):
                self.events.append(("finish",))

        entries = []
        for name, result, policy in (("first", "failed", "skip"), ("second", "verified", "stop")):
            task = {"task": name, "start_scenes": [name], "initial_state": "end",
                    "states": {"end": {"result": result, "reason": name + " reason"}}}
            entries.append({"name": name, "on_failure": policy, "task": task})
        runtime = FakeRuntime()
        with self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertTrue(run_list(runtime, SimpleNamespace(tasks=entries)))
        self.assertEqual(runtime.events, [("prepare", ["first"]), ("finish",),
                                          ("prepare", ["second"]), ("finish",)])
        self.assertTrue(any("task=first state=end result=failed reason=first reason" in s for s in logs.output))
        self.assertTrue(any("task=second state=end result=verified reason=second reason" in s for s in logs.output))


class RunLoopTests(unittest.TestCase):
    NOQUEUE = "no available army queue"

    def loop(self, count=2, wait=30, max_retries=2, on_failure="stop", retry=True):
        retry = [{"result": "suspended", "wait": 10}] if retry else []
        return {"count": count, "on_failure": on_failure, "wait": wait,
                "success": {"result": "unverified", "reason": ""},
                "retry": retry, "max_retries": max_retries,
                "task": {"task": "beast", "start_scenes": ["wild"], "states": {}}}

    def setUp(self):
        self.runtime = Mock()
        self.patcher = patch("automation.runner.run_task")
        self.run_task = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def result(self, result, reason=""):
        return TaskResult(result, reason, "end")

    def loop_verified(self, count=1, wait=30, max_retries=2, retry=True):
        retry = [{"result": "suspended", "wait": 10}] if retry else []
        return {"count": count, "on_failure": "stop", "wait": wait,
                "success": {"result": "verified", "reason": ""},
                "retry": retry, "max_retries": max_retries,
                "task": {"task": "beast", "start_scenes": ["wild"], "states": {}}}

    def test_successes_count_and_wait_between_attempts(self):
        self.run_task.side_effect = [self.result("unverified"), self.result("unverified")]
        with patch("automation.runner.time.sleep") as sleep, \
                self.assertLogs("automation.runner", level="INFO") as logs:
            self.assertTrue(run_loop(self.runtime, self.loop()))
        self.assertEqual(self.run_task.call_count, 2)
        # one 30s success wait between the two attempts, none after the last
        self.assertEqual(sleep.call_args_list, [call(30)])
        self.runtime.prepare.assert_called()
        self.runtime.finish.assert_called_once_with()
        self.assertTrue(any("success=2/2" in s for s in logs.output))

    def test_suspended_retries_by_result_without_counting(self):
        self.run_task.side_effect = [
            self.result("suspended", self.NOQUEUE),
            self.result("suspended", self.NOQUEUE),
            self.result("unverified"),
        ]
        with patch("automation.runner.time.sleep") as sleep:
            self.assertTrue(run_loop(self.runtime, self.loop(count=1)))
        self.assertEqual(self.run_task.call_count, 3)
        # two 10s retry waits, no success wait (single success)
        self.assertEqual(sleep.call_args_list, [call(10), call(10)])
        self.runtime.finish.assert_called_once_with()

    def test_retry_counter_resets_after_success(self):
        # max_retries=1; two separate outages of one retry each around a success
        self.run_task.side_effect = [
            self.result("suspended", self.NOQUEUE),
            self.result("unverified"),
            self.result("suspended", self.NOQUEUE),
            self.result("unverified"),
        ]
        with patch("automation.runner.time.sleep"):
            self.assertTrue(run_loop(self.runtime, self.loop(count=2, max_retries=1)))
        self.assertEqual(self.run_task.call_count, 4)
        self.runtime.finish.assert_called_once_with()

    def test_retry_exhaustion_stops_without_recovery(self):
        self.run_task.return_value = self.result("suspended", self.NOQUEUE)
        self.assertFalse(run_loop(self.runtime, self.loop(count=1, max_retries=2)))
        self.assertEqual(self.run_task.call_count, 3)  # 1 miss + 2 retries
        self.runtime.finish.assert_not_called()

    def test_unverified_not_in_retry_entries_stops(self):
        # retry matches by result: unverified has no retry entry, so it stops.
        self.run_task.return_value = self.result("unverified", "anything")
        self.assertFalse(run_loop(self.runtime, self.loop(count=1)))
        self.run_task.assert_called_once()
        self.runtime.finish.assert_not_called()

    def test_uncertain_stops_even_with_skip(self):
        self.run_task.return_value = self.result("uncertain", "scene not recognized")
        self.assertFalse(run_loop(self.runtime, self.loop(on_failure="skip")))
        self.run_task.assert_called_once()
        self.runtime.finish.assert_not_called()

    def test_failed_skip_continues_without_counting(self):
        self.run_task.side_effect = [self.result("failed", "boom"), self.result("unverified")]
        self.assertTrue(run_loop(self.runtime, self.loop(count=1, on_failure="skip")))
        self.assertEqual(self.run_task.call_count, 2)
        self.runtime.finish.assert_called_once_with()

    def test_failed_stop_stops(self):
        self.run_task.return_value = self.result("failed", "boom")
        self.assertFalse(run_loop(self.runtime, self.loop(count=1, on_failure="stop")))
        self.run_task.assert_called_once()
        self.runtime.finish.assert_not_called()

    def test_prepare_frame_timeout_stops_cleanly(self):
        self.runtime.prepare.side_effect = FrameTimeout("silent")
        self.assertFalse(run_loop(self.runtime, self.loop()))
        self.run_task.assert_not_called()
        self.runtime.finish.assert_not_called()

    def test_interrupt_propagates_without_finish(self):
        self.run_task.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_loop(self.runtime, self.loop())
        self.runtime.finish.assert_not_called()

    def test_success_wait_zero_skips_sleep(self):
        self.run_task.side_effect = [self.result("unverified"), self.result("unverified")]
        with patch("automation.runner.time.sleep") as sleep:
            self.assertTrue(run_loop(self.runtime, self.loop(count=2, wait=0)))
        self.assertEqual(sleep.call_count, 0)

    def test_verified_success_counts_toward_target(self):
        self.run_task.side_effect = [self.result("verified"), self.result("verified")]
        with patch("automation.runner.time.sleep"):
            self.assertTrue(run_loop(self.runtime, self.loop_verified(count=2)))
        self.assertEqual(self.run_task.call_count, 2)
        self.runtime.finish.assert_called_once_with()

    def test_suspended_retries_without_counting_until_success(self):
        self.run_task.side_effect = [
            self.result("suspended", self.NOQUEUE),
            self.result("suspended", self.NOQUEUE),
            self.result("verified"),
        ]
        with patch("automation.runner.time.sleep") as sleep:
            self.assertTrue(run_loop(self.runtime, self.loop_verified()))
        self.assertEqual(self.run_task.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(10), call(10)])
        self.runtime.finish.assert_called_once_with()

    def test_suspended_retry_exhaustion_stops(self):
        self.run_task.return_value = self.result("suspended", self.NOQUEUE)
        self.assertFalse(run_loop(self.runtime, self.loop_verified(max_retries=2)))
        self.assertEqual(self.run_task.call_count, 3)
        self.runtime.finish.assert_not_called()


class RunListSuspendedTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Mock()
        self.patcher = patch("automation.runner.run_task")
        self.run_task = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_suspended_continues_to_next_task_and_finishes(self):
        entries = [{"name": "a", "on_failure": "stop",
                    "task": {"task": "a", "start_scenes": ["a_main"]}},
                   {"name": "b", "on_failure": "stop",
                    "task": {"task": "b", "start_scenes": ["b_main"]}}]
        self.run_task.return_value = TaskResult("suspended", "no available army queue", "end")
        self.assertTrue(run_list(self.runtime, SimpleNamespace(tasks=entries)))
        self.assertEqual(self.run_task.call_count, 2)
        self.runtime.prepare.assert_called()
        self.runtime.finish.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
