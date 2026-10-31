from __future__ import annotations

import logging
import time

from automation.engine import run_task
from automation.cancellation import check_cancelled, sleep
from automation.runtime import FrameTimeout, record_trace_if_enabled, set_trace_context_if_enabled


logger = logging.getLogger(__name__)


def run_list(runtime, plan, *, cancel_event=None) -> bool:
    for index, entry in enumerate(plan.tasks):
        check_cancelled(cancel_event)
        name = entry["name"]
        task = entry["task"]
        try:
            set_trace_context_if_enabled(runtime, task=task["task"], phase="prepare")
            record_trace_if_enabled(runtime, "task_prepare_started", start_scenes=task["start_scenes"])
            logger.info("task=%s preparing start_scenes=%s", name, task["start_scenes"])
            try:
                runtime.prepare(task["start_scenes"])
            except FrameTimeout:
                record_trace_if_enabled(runtime, "task_prepare_failed", reason="video stream silent")
                # Preparing needs a frame to identify the current scene. If the stream
                # is silent (the screen turned off during a previous long wait), no
                # further task can be started safely, so stop with a clean warning
                # instead of an exception traceback.
                logger.warning(
                    "task=%s stopped before start: video stream silent (the screen may be "
                    "off); start the tasks after the device screen is on", name)
                return False
            except Exception as exc:
                record_trace_if_enabled(runtime, "task_prepare_failed",
                                        error=f"{type(exc).__name__}: {exc}")
                raise
            record_trace_if_enabled(runtime, "task_prepare_completed")
            check_cancelled(cancel_event)
            result = (run_task(runtime, task) if cancel_event is None
                      else run_task(runtime, task, cancel_event=cancel_event))
            check_cancelled(cancel_event)
            record_trace_if_enabled(runtime, "task_prepare_and_run_completed", result=result.result,
                                    reason=result.reason, state=result.state)
            logger.info("task=%s state=%s result=%s reason=%s", name, result.state, result.result, result.reason)
            # suspended is a planned pause (e.g. no army queue), not an error.
            if result.result in ("verified", "unverified", "suspended"):
                continue
            if result.result != "failed" or entry["on_failure"] != "skip":
                logger.error("task=%s stopped list: result=%s policy=%s", name, result.result, entry["on_failure"])
                return False
            logger.warning("task=%s failed; skipping after safe recovery", name)
            if index + 1 < len(plan.tasks):
                runtime.finish()
        except Exception:
            logger.exception("task=%s stopped list by exception; no further recovery", name)
            return False

    check_cancelled(cancel_event)
    try:
        recovery = runtime.finish()
    except Exception:
        logger.exception("task list completed but final recovery failed")
        return False
    if recovery == "video stream silent":
        logger.warning("task list completed; final recovery was not verified (screen may be off)")
    else:
        logger.info("task list completed and recovered to a main scene")
    return True


def run_loop(runtime, loop, *, cancel_event=None) -> bool:
    """Repeat one task until the success target is reached.

    A run matching loop.success counts as a success; after a success the loop
    waits loop.wait seconds before the next attempt. A run whose result matches
    a loop.retry entry (unverified or suspended) waits that entry's seconds and
    retries without counting, regardless of the reason. Suspended runs are first
    recovered to a main scene so an open sub-scene does not carry into the wait or
    next attempt. The consecutive retry
    counter resets
    after each success, so loop.max_retries caps how many no-progress retries
    happen between successes (a long queue outage stops the run). Any other
    failed or uncertain result is handled by loop.on_failure (stop, or skip to
    the next attempt). KeyboardInterrupt propagates so the caller can stop the
    session.
    """
    name = loop["task"]["task"]
    success = loop["success"]
    retries = {item["result"]: item["wait"] for item in loop["retry"]}
    attempts = successes = retry_attempts = 0
    while successes < loop["count"]:
        check_cancelled(cancel_event)
        attempts += 1
        set_trace_context_if_enabled(runtime, task=name, attempt=attempts, phase="prepare")
        record_trace_if_enabled(runtime, "task_prepare_started", start_scenes=loop["task"]["start_scenes"])
        logger.info("loop=%s attempt=%d success=%d/%d preparing start_scenes=%s",
                    name, attempts, successes, loop["count"], loop["task"]["start_scenes"])
        try:
            runtime.prepare(loop["task"]["start_scenes"])
        except FrameTimeout:
            record_trace_if_enabled(runtime, "task_prepare_failed", reason="video stream silent")
            logger.warning(
                "loop=%s stopped: video stream silent (the screen may be off); start "
                "after the device screen is on", name)
            return False
        except Exception as exc:
            record_trace_if_enabled(runtime, "task_prepare_failed",
                                    error=f"{type(exc).__name__}: {exc}")
            logger.exception("loop=%s stopped by prepare exception", name)
            return False
        record_trace_if_enabled(runtime, "task_prepare_completed")
        try:
            check_cancelled(cancel_event)
            result = (run_task(runtime, loop["task"]) if cancel_event is None
                      else run_task(runtime, loop["task"], cancel_event=cancel_event))
            check_cancelled(cancel_event)
            record_trace_if_enabled(runtime, "task_prepare_and_run_completed", result=result.result,
                                    reason=result.reason, state=result.state)
            logger.info("task=%s state=%s result=%s reason=%s",
                        name, result.state, result.result, result.reason)
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("loop=%s stopped by exception", name)
            return False
        if result.result == "suspended":
            # A suspended task can leave a recognized sub-scene open (for example,
            # the no-available-army-queue popup). Recover to a main scene before
            # waiting/retrying, rather than carrying that popup into the next attempt.
            try:
                recovery = runtime.finish()
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("loop=%s stopped: recovery after suspended result failed", name)
                return False
            if recovery == "video stream silent":
                logger.warning(
                    "loop=%s stopped after suspended result: could not verify recovery "
                    "because the video stream is silent", name)
                return False
            logger.info("loop=%s recovered to a main scene after suspended result", name)
        if result.result == success["result"] and result.reason == success["reason"]:
            successes += 1
            retry_attempts = 0
            logger.info("loop=%s success=%d/%d", name, successes, loop["count"])
            if successes < loop["count"] and loop["wait"] > 0:
                logger.info("loop=%s waiting %ss before next attempt", name, loop["wait"])
                try:
                    sleep(loop["wait"], cancel_event)
                except KeyboardInterrupt:
                    raise
            continue
        if result.result in retries:
            if retry_attempts >= loop["max_retries"]:
                logger.error("loop=%s stopped: retry attempts exhausted (%d) "
                             "result=%s reason=%s", name, loop["max_retries"],
                             result.result, result.reason)
                return False
            retry_attempts += 1
            logger.info("loop=%s result=%s reason=%s; retrying after %ss (retry %d/%d)",
                        name, result.result, result.reason, retries[result.result],
                        retry_attempts, loop["max_retries"])
            if retries[result.result] > 0:
                try:
                    sleep(retries[result.result], cancel_event)
                except KeyboardInterrupt:
                    raise
            continue
        if result.result == "failed" and loop["on_failure"] == "skip":
            logger.warning("loop=%s failed; skipping to next attempt", name)
            continue
        logger.error("loop=%s stopped: result=%s reason=%s policy=%s",
                     name, result.result, result.reason, loop["on_failure"])
        return False
    check_cancelled(cancel_event)
    try:
        recovery = runtime.finish()
    except Exception:
        logger.exception("loop completed but final recovery failed")
        return False
    if recovery == "video stream silent":
        logger.warning("loop completed; final recovery was not verified (screen may be off)")
    else:
        logger.info("loop completed: %d success in %d attempts; recovered to a main scene",
                    loop["count"], attempts)
    return True
