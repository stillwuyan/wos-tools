from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from automation.runtime import ActionDeadlineExceeded, FrameTimeout
from automation.cancellation import check_cancelled, sleep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskResult:
    result: str
    reason: str
    state: str


def run_task(runtime, task, *, cancel_event=None) -> TaskResult:
    """Run a validated task; runtime and recognition errors are not task results.

    snapshot must honor after_id and timeout, and propagate acquisition errors.
    runtime.poll, if present, is a positive polling interval in seconds.
    """
    name = task["task"]
    states = task["states"]
    state_name = task["initial_state"]
    poll = getattr(runtime, "poll", 0.1)
    if type(poll) not in (int, float) or not math.isfinite(poll) or poll <= 0:
        raise ValueError("runtime.poll must be a finite positive number")

    for label, state in states.items():
        if "result" in state:
            continue
        effects = any(next(iter(action)) in ("click", "swipe", "back") for action in state["actions"])
        if effects and not state.get("require_scene"):
            raise ValueError(f"state {label}: side effects require require_scene")
        if "result" not in state and "next" not in state:
            timeout = state["timeout"]
            if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or
                    state["on_timeout"] not in ("failed", "uncertain")):
                raise ValueError(f"state {label}: invalid timeout or on_timeout")
        if "next" in state and "transitions" in state:
            raise ValueError(f"state {label}: next and transitions are mutually exclusive")
        if "retry" in state:
            actions = state["actions"]
            retry = state["retry"]
            if (len(actions) != 1 or len(actions[0]) != 1 or
                    next(iter(actions[0])) not in ("click", "swipe", "back")):
                raise ValueError(f"state {label}: retry requires exactly one side-effect action")
            if "next" in state:
                raise ValueError(f"state {label}: retry requires transitions")
            if retry["require_scene"] != state.get("require_scene"):
                raise ValueError(f"state {label}: retry require_scene must match state require_scene")
            if (type(retry["count"]) is not int or not 1 <= retry["count"] <= 100 or
                    type(retry["interval"]) not in (int, float) or
                    not math.isfinite(retry["interval"]) or retry["interval"] <= 0):
                raise ValueError(f"state {label}: invalid retry count or interval")

    # A pending action watermark survives unconditional transitions.
    after_id = None
    unknown_watermark = False
    visited = set()

    def recognize(snapshot, transitions, scene=None, *, already_observed=False):
        observed = snapshot
        if not already_observed:
            observed, scene = runtime.observe(snapshot)
        if observed.frame_id != snapshot.frame_id:
            raise RuntimeError("runtime.observe changed the frame")
        matches = []
        if scene is None:
            return scene, matches
        for transition in transitions:
            condition = transition["when"]
            if ("scene" in condition and condition["scene"] == scene or
                    "target" in condition and runtime.detector.match(observed, condition["target"]) is not None):
                matches.append(transition)
        return scene, matches

    def finish(result, reason):
        # The list runner logs the final result; avoid duplicating it here.
        return TaskResult(result, reason, state_name)

    try:
        while True:
            check_cancelled(cancel_event)
            if state_name in visited:
                raise ValueError(f"task {name}: cycle involving state {state_name}")
            visited.add(state_name)
            state = states[state_name]
            logger.info("task=%s state=%s entered", name, state_name)
            if "result" in state:
                return finish(state["result"], state.get("reason", ""))

            actions = state["actions"]
            if actions:
                frame_id = runtime.execute(actions, expected_scene=state.get("require_scene"))
                if frame_id is None:
                    unknown_watermark = True
                else:
                    after_id = frame_id if after_id is None else max(after_id, frame_id)
                    unknown_watermark = False

            if "next" in state:
                state_name = state["next"]
                continue

            timeout = state["timeout"]
            deadline = time.monotonic() + timeout
            retries = 0
            retry = state.get("retry")
            transitions = state["transitions"]

            while True:
                check_cancelled(cancel_event)
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    if unknown_watermark:
                        watermark = runtime.snapshot(timeout=remaining)
                        after_id = watermark.frame_id if after_id is None else max(after_id, watermark.frame_id)
                        unknown_watermark = False
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            continue
                    # A condition deadline alone cannot establish that the video is healthy.
                    snapshot = runtime.snapshot(after_id=after_id, timeout=max(
                        remaining, getattr(runtime, "frame_timeout", 0)))
                    if after_id is not None and snapshot.frame_id <= after_id:
                        raise RuntimeError("runtime.snapshot returned a stale frame")
                    after_id = snapshot.frame_id
                    scene, matches = recognize(snapshot, transitions)
                    if scene is None:
                        if time.monotonic() >= deadline:
                            return finish("uncertain", "scene not recognized")
                        sleep(min(poll, max(0, deadline - time.monotonic())), cancel_event)
                        continue
                    if len(matches) > 1:
                        return finish("uncertain", "conflicting transitions")
                    if time.monotonic() > deadline:
                        if matches:
                            return finish("uncertain", "transition observed after deadline")
                    elif matches:
                        state_name = matches[0]["to"]
                        after_id = None
                        break
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            sleep(min(poll, remaining), cancel_event)
                        continue
                else:
                    # Allow one frame interval to establish stream health at the deadline.
                    try:
                        snapshot = runtime.snapshot(after_id=after_id, timeout=getattr(runtime, "frame_timeout", 0))
                    except FrameTimeout:
                        return finish("uncertain", "no fresh frame at deadline")
                    if after_id is not None and snapshot.frame_id <= after_id:
                        raise RuntimeError("runtime.snapshot returned a stale frame")
                    after_id = snapshot.frame_id
                    if unknown_watermark:
                        unknown_watermark = False
                        return finish("uncertain", "no post-action frame at deadline")
                    scene, matches = recognize(snapshot, transitions)
                    if scene is None:
                        return finish("uncertain", "scene not recognized")
                    if len(matches) > 1:
                        return finish("uncertain", "conflicting transitions")
                    if matches:
                        if time.monotonic() > deadline:
                            return finish("uncertain", "transition observed after deadline")
                        state_name = matches[0]["to"]
                        after_id = None
                        break

                if retry is None or retries >= retry["count"]:
                    return finish(state["on_timeout"], "state timeout")

                retries += 1
                sleep(retry["interval"], cancel_event)
                deadline = time.monotonic() + timeout
                # Check delayed results and the retry precondition on the same fresh frame.
                if unknown_watermark:
                    watermark = runtime.snapshot(timeout=max(0, deadline - time.monotonic()))
                    after_id = watermark.frame_id if after_id is None else max(after_id, watermark.frame_id)
                    unknown_watermark = False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    continue
                snapshot = runtime.snapshot(after_id=after_id, timeout=max(
                    remaining, getattr(runtime, "frame_timeout", 0)))
                if after_id is not None and snapshot.frame_id <= after_id:
                    raise RuntimeError("runtime.snapshot returned a stale frame")
                after_id = snapshot.frame_id
                unknown_watermark = False
                scene, matches = recognize(snapshot, transitions)
                if scene is None:
                    if time.monotonic() >= deadline:
                        return finish("uncertain", "scene not recognized")
                    continue
                if len(matches) > 1:
                    return finish("uncertain", "conflicting transitions")
                if time.monotonic() > deadline:
                    if matches:
                        return finish("uncertain", "transition observed after deadline")
                    continue
                if matches:
                    state_name = matches[0]["to"]
                    after_id = None
                    break
                if scene != retry["require_scene"]:
                    return finish(state["on_timeout"], "retry scene not recognized or changed")

                checked = None

                def before_action(frame, actual_scene):
                    nonlocal checked
                    if frame.frame_id < after_id:
                        raise RuntimeError("runtime.execute returned a stale action frame")
                    if frame.frame_id == after_id:
                        checked = (frame.frame_id, [], False, actual_scene, True)
                        return False
                    _, actual_matches = recognize(frame, transitions, actual_scene, already_observed=True)
                    checked = (frame.frame_id, actual_matches, time.monotonic() > deadline, actual_scene, False)
                    return actual_scene is not None and not actual_matches and not checked[2]

                try:
                    frame_id = runtime.execute(actions, expected_scene=retry["require_scene"],
                                               before_action=before_action, action_deadline=deadline,
                                               after_id=after_id)
                except ActionDeadlineExceeded:
                    return finish("uncertain", "retry action deadline expired")
                if checked is None:
                    raise RuntimeError("runtime.execute did not check the action frame")
                if checked[4]:
                    return finish("uncertain", "no fresh retry action frame")
                if checked[3] is None:
                    if time.monotonic() >= deadline:
                        return finish("uncertain", "scene not recognized")
                    after_id = max(after_id, checked[0])
                    continue
                if len(checked[1]) > 1:
                    return finish("uncertain", "conflicting transitions")
                if checked[2]:
                    if checked[1]:
                        return finish("uncertain", "transition observed after deadline")
                    after_id = max(after_id, checked[0])
                    continue
                if checked[1]:
                    state_name = checked[1][0]["to"]
                    after_id = None
                    break
                if frame_id is None:
                    unknown_watermark = True
                else:
                    after_id = max(after_id, frame_id)
                deadline = time.monotonic() + timeout
    except KeyboardInterrupt:
        raise
    except BaseException:
        logger.exception("task=%s state=%s stopped by exception", name, state_name)
        raise
