from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from home_paths import PROJECT_HOME, resolve_home_path


class _UniqueKeyLoader(yaml.SafeLoader):
    def flatten_mapping(self, node):
        if not hasattr(self, "_merging"):
            self._merging = set()
        if node in self._merging:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                "recursive YAML merge", node.start_mark)
        self._merging.add(node)
        try:
            super().flatten_mapping(node)
        finally:
            self._merging.remove(node)

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "unhashable key", key_node.start_mark) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"duplicate key {key!r}", key_node.start_mark)
        return super().construct_mapping(node, deep=deep)


@dataclass
class Plan:
    resources: dict
    scenes: dict
    switches: list
    tasks: list
    loop: dict = None
    scrcpy: dict = field(default_factory=dict)


def _mapping(value, where, required=(), optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{where}: keys must be strings")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{where}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}")
    return value


def _names(value, where, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{where}: expected {'a nonempty' if nonempty else 'a'} list")
    for item in value:
        _name(item, where)
    return value


def _name(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected a nonempty string")
    return value


def _number(value, where, *, zero=False):
    try:
        valid = (type(value) in (int, float) and math.isfinite(value) and
                 (value >= 0 if zero else value > 0))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{where}: expected a finite {'nonnegative' if zero else 'positive'} number")


def _read_yaml(path):
    try:
        with path.open(encoding="utf-8") as fp:
            return yaml.load(fp, Loader=_UniqueKeyLoader)
    except (OSError, ValueError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load YAML {path}: {exc}") from exc


def _path(base, value, where):
    _name(value, where)
    try:
        return resolve_home_path(value, base)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{where}: invalid path {value!r}") from exc


def _size(value, where):
    _mapping(value, where, ("width", "height"))
    if any(type(value[k]) is not int or value[k] <= 0 for k in ("width", "height")):
        raise ValueError(f"{where}: width and height must be positive integers")
    return value["width"], value["height"]


class _Compiler:
    def __init__(self, home, device):
        _mapping(device, "device", ("frame_size", "device_size"))
        self.home = home
        self.meta = {}
        for field in ("frame_size", "device_size"):
            value = device[field]
            if (not isinstance(value, list) or len(value) != 2 or
                    any(type(n) is not int or n <= 0 for n in value)):
                raise ValueError(f"device.{field}: expected two positive integers")
            self.meta[field] = dict(zip(("width", "height"), value))
        self.frame_size = tuple(device["frame_size"])
        self.device_size = tuple(device["device_size"])
        self.resources = {}
        self.ids = {}

    def add(self, key, record):
        if key not in self.ids:
            name = f"resource_{len(self.ids) + 1:04d}"
            self.ids[key] = name
            self.resources[name] = {**record, **{k: v.copy() for k, v in self.meta.items()}}
        return self.ids[key]

    def image(self, value, where):
        path = _path(self.home, value, where)
        key = ("image", path)
        if key in self.ids:
            return self.ids[key]
        try:
            pixels = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(pixels, cv2.IMREAD_GRAYSCALE) if pixels.size else None
        except (OSError, ValueError, cv2.error) as exc:
            raise ValueError(f"{where}: cannot decode image {path}: {exc}") from exc
        if image is None:
            raise ValueError(f"{where}: invalid image {path}")
        h, w = image.shape
        if w >= self.frame_size[0] or h >= self.frame_size[1]:
            raise ValueError(f"{where}: image must be smaller than frame")
        if image.min() == image.max():
            raise ValueError(f"{where}: constant grayscale image {path} has no texture")
        # Back the validated pixels with immutable bytes, not a mutable array.
        image = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(image.shape)
        return self.add(key, {"action": "template", "image": path, "grayscale": image,
                              "frame_rect": {"x1": 0, "y1": 0, "x2": w, "y2": h}})

    def point(self, value, where):
        if (not isinstance(value, list) or len(value) != 2 or
                any(type(n) is not int or not 0 <= n < self.device_size[i] for i, n in enumerate(value))):
            raise ValueError(f"{where}: point outside device {self.device_size}")
        x, y = value
        return self.add(("point", x, y), {"action": "point",
                                          "device": {"x": x, "y": y}})

    def action(self, action, where, *, wait_scene=False):
        if not isinstance(action, dict) or len(action) != 1:
            raise ValueError(f"{where}: action must be a single-key mapping")
        kind, value = next(iter(action.items()))
        if kind == "click":
            return {"click": {"target": self.point(value, where), "mode": "fixed"}}
        if kind == "click_image":
            return {"click": {"target": self.image(value, where), "mode": "match"}}
        if kind == "swipe":
            _mapping(value, where, ("from", "to", "duration"))
            return {"swipe": {"from": self.point(value["from"], f"{where}.from"),
                              "to": self.point(value["to"], f"{where}.to"), "duration": value["duration"]}}
        if kind in ("wait", "back") or kind == "wait_scene" and wait_scene:
            return {kind: value}
        raise ValueError(f"{where}: unsupported action {kind!r}")

    def condition(self, value, where):
        if not isinstance(value, dict) or len(value) != 1:
            raise ValueError(f"{where}: expected exactly one scene or image")
        if "scene" in value:
            return {"scene": value["scene"]}
        if "image" in value:
            return {"target": self.image(value["image"], where)}
        raise ValueError(f"{where}: expected scene or image")


def _reference(name, objects, where, kind=None):
    _name(name, where)
    if name not in objects or (kind is not None and objects[name].get("action") != kind):
        raise ValueError(f"{where}: unknown or wrong type {kind or 'reference'} {name!r}")


def _acyclic(graph, where):
    # Iterative DFS also handles large, valid graphs without recursion limits.
    done, active = set(), set()
    for root in graph:
        if root in done:
            continue
        stack = [(root, iter(graph[root]))]
        active.add(root)
        while stack:
            node, edges = stack[-1]
            target = next(edges, None)
            if target is None:
                stack.pop()
                active.remove(node)
                done.add(node)
            elif target in active:
                raise ValueError(f"{where}: cycle involving {target!r}")
            elif target not in done:
                active.add(target)
                stack.append((target, iter(graph[target])))


def _actions(actions, resources, scenes, where, *, wait_scene=False):
    if not isinstance(actions, list):
        raise ValueError(f"{where}: actions must be a list")
    for index, action in enumerate(actions):
        loc = f"{where}[{index}]"
        if not isinstance(action, dict) or len(action) != 1:
            raise ValueError(f"{loc}: action must be a single-key mapping")
        kind, value = next(iter(action.items()))
        if kind == "click":
            _mapping(value, loc, ("target", "mode"))
            if value["mode"] not in ("fixed", "match"):
                raise ValueError(f"{loc}: invalid click mode")
            _reference(value["target"], resources, loc, "template" if value["mode"] == "match" else "point")
        elif kind == "swipe":
            _mapping(value, loc, ("from", "to", "duration"))
            for key in ("from", "to"):
                _reference(value[key], resources, f"{loc}.{key}", "point")
            _number(value["duration"], f"{loc}.duration")
            if not math.isfinite(value["duration"] * 1000.0):
                raise ValueError(f"{loc}.duration: milliseconds must be finite")
        elif kind == "wait":
            _number(value, loc, zero=True)
        elif kind == "back":
            if value is not True:
                raise ValueError(f"{loc}: back must be true")
        elif kind == "wait_scene" and wait_scene:
            _mapping(value, loc, ("scenes", "timeout"))
            for scene in _names(value["scenes"], f"{loc}.scenes"):
                _reference(scene, scenes, loc)
            _number(value["timeout"], f"{loc}.timeout")
        else:
            raise ValueError(f"{loc}: unsupported action {kind!r}")


def _scene_actions(actions, resources, scenes, where):
    _actions(actions, resources, scenes, where, wait_scene=True)
    pending = False
    effects = 0
    for action in actions:
        if next(iter(action)) in ("click", "swipe", "back"):
            if pending:
                raise ValueError(f"{where}: side effects must be separated by wait_scene")
            pending = True
            effects += 1
        elif "wait_scene" in action:
            pending = False
    if not effects:
        raise ValueError(f"{where}: requires a side effect")
    if not actions or "wait_scene" not in actions[-1]:
        raise ValueError(f"{where}: must end with wait_scene")


def _scenes(data, resources, where):
    scenes, switches = data["scenes"], data["switches"]
    if not isinstance(scenes, dict):
        raise ValueError(f"{where}: scenes must be a mapping")
    for name, scene in scenes.items():
        _name(name, f"{where}: scene name")
        loc = f"scene {name}"
        _mapping(scene, loc, ("type", "template"), ("threshold", "region", "covers", "exit"))
        if scene["type"] not in ("main", "sub"):
            raise ValueError(f"{loc}: type must be main or sub")
        _reference(scene["template"], resources, loc, "template")
        if "threshold" in scene:
            _number(scene["threshold"], f"{loc}.threshold", zero=True)
            if scene["threshold"] > 1:
                raise ValueError(f"{loc}: threshold must be between 0 and 1")
        if "region" in scene:
            region = scene["region"]
            if not isinstance(region, list) or len(region) != 4 or any(type(n) is not int for n in region):
                raise ValueError(f"{loc}: region must be [x, y, width, height] integers")
            x, y, w, h = region
            record = resources[scene["template"]]
            fw, fh = _size(record["frame_size"], loc)
            rect = record["frame_rect"]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh or w < rect["x2"] - rect["x1"] or h < rect["y2"] - rect["y1"]:
                raise ValueError(f"{loc}: region outside frame or smaller than template")
        if "covers" in scene:
            if scene["type"] != "sub":
                raise ValueError(f"{loc}: only sub scenes can cover scenes")
            _names(scene["covers"], f"{loc}.covers", nonempty=False)
    if not all(name in scenes and scenes[name]["type"] == "main" for name in ("town", "wild")):
        raise ValueError("scenes: town and wild must both be main scenes")
    graph = {}
    for name, scene in scenes.items():
        graph[name] = scene.get("covers", [])
        for covered in graph[name]:
            _reference(covered, scenes, f"scene {name}.covers")
        if "exit" in scene:
            actions = scene["exit"]
            _scene_actions(actions, resources, scenes, f"scene {name}.exit")
            if any(scenes[target]["type"] != "main" for target in actions[-1]["wait_scene"]["scenes"]):
                raise ValueError(f"scene {name}.exit: final wait_scene must only wait for main scenes")
    _acyclic(graph, "scene covers")
    # Different templates can still match the same frame; covers must reflect
    # real UI relationships and must not be used to hide misrecognition.
    for name in graph:
        pixels = resources[scenes[name]["template"]]["grayscale"]
        pending = list(graph[name])
        seen = set()
        while pending:
            covered = pending.pop()
            if covered in seen:
                continue
            seen.add(covered)
            other = resources[scenes[covered]["template"]]["grayscale"]
            if np.array_equal(pixels, other):
                raise ValueError(f"scene {name}.covers: same grayscale image as covered scene {covered}")
            pending.extend(graph[covered])
    if not isinstance(switches, list):
        raise ValueError(f"{where}: switches must be a list")
    pairs = set()
    for index, switch in enumerate(switches):
        loc = f"switch {index}"
        _mapping(switch, loc, ("from", "to", "actions"))
        for key in ("from", "to"):
            _reference(switch[key], scenes, f"{loc}.{key}")
            if scenes[switch[key]]["type"] != "main":
                raise ValueError(f"{loc}.{key}: must reference a main scene")
        pair = (switch["from"], switch["to"])
        if pair[0] == pair[1] or pair in pairs:
            raise ValueError(f"{loc}: switch pair must be distinct and not duplicated")
        pairs.add(pair)
        actions = switch["actions"]
        _scene_actions(actions, resources, scenes, f"{loc}.actions")
        if not actions or "wait_scene" not in actions[-1] or actions[-1]["wait_scene"]["scenes"] != [switch["to"]]:
            raise ValueError(f"{loc}: must end with wait_scene for only the target scene")
    return scenes, switches


def _task(task, path, resources, scenes):
    _mapping(task, str(path), ("task", "start_scenes", "initial_state", "states"))
    _name(task["task"], f"{path}: task")
    for scene in _names(task["start_scenes"], f"{path}: start_scenes"):
        _reference(scene, scenes, f"{path}: start_scenes")
        if scenes[scene]["type"] != "main":
            raise ValueError(f"{path}: start scene {scene!r} must be main")
    states = task["states"]
    if not isinstance(states, dict) or not states:
        raise ValueError(f"{path}: states must be a nonempty mapping")
    for name in states:
        _name(name, f"{path}: state name")
    _reference(task["initial_state"], states, f"{path}: initial_state")
    graph = {}
    for name, state in states.items():
        loc = f"task {task['task']} state {name}"
        if not isinstance(state, dict):
            raise ValueError(f"{loc}: expected a mapping")
        graph[name] = []
        if "result" in state:
            _mapping(state, loc, ("result",), ("reason",))
            if state["result"] not in ("verified", "unverified", "failed", "uncertain", "suspended"):
                raise ValueError(f"{loc}: invalid result")
            if "reason" in state and not isinstance(state["reason"], str):
                raise ValueError(f"{loc}: reason must be a string")
            continue
        if "next" in state:
            _mapping(state, loc, ("actions", "next"), ("require_scene",))
        else:
            _mapping(state, loc, ("actions", "transitions", "timeout", "on_timeout"), ("retry", "require_scene"))
        _actions(state["actions"], resources, scenes, f"{loc}.actions")
        effects = sum(next(iter(action)) in ("click", "swipe", "back") for action in state["actions"])
        if effects > 1:
            raise ValueError(f"{loc}: at most one side effect is allowed per state")
        if effects and "require_scene" not in state:
            raise ValueError(f"{loc}: side effects require require_scene")
        if "require_scene" in state:
            _reference(state["require_scene"], scenes, f"{loc}.require_scene")
        if "next" in state:
            if not state["actions"]:
                raise ValueError(f"{loc}: next requires nonempty actions")
            _reference(state["next"], states, f"{loc}.next")
            graph[name].append(state["next"])
            continue
        _number(state["timeout"], f"{loc}.timeout")
        if state["on_timeout"] not in ("failed", "uncertain"):
            raise ValueError(f"{loc}: invalid on_timeout")
        if not isinstance(state["transitions"], list) or not state["transitions"]:
            raise ValueError(f"{loc}: transitions must be a nonempty list")
        conditions = set()
        for transition in state["transitions"]:
            _mapping(transition, f"{loc}.transition", ("when", "to"))
            condition = transition["when"]
            if not isinstance(condition, dict) or len(condition) != 1:
                raise ValueError(f"{loc}: when must contain exactly one scene or target")
            if "scene" in condition:
                _reference(condition["scene"], scenes, f"{loc}.when.scene")
            elif "target" in condition:
                _reference(condition["target"], resources, f"{loc}.when.target", "template")
            else:
                raise ValueError(f"{loc}: when must contain scene or target")
            condition_key = next(iter(condition.items()))
            if condition_key in conditions:
                raise ValueError(f"{loc}: duplicate when condition {condition_key!r}")
            conditions.add(condition_key)
            _reference(transition["to"], states, f"{loc}.to")
            graph[name].append(transition["to"])
        if "retry" in state:
            retry = _mapping(state["retry"], f"{loc}.retry", ("count", "interval", "require_scene"))
            if type(retry["count"]) is not int or not 1 <= retry["count"] <= 100:
                raise ValueError(f"{loc}.retry: count must be an integer between 1 and 100")
            if effects != 1 or len(state["actions"]) != 1:
                raise ValueError(f"{loc}.retry: requires exactly one side effect and no wait")
            _number(retry["interval"], f"{loc}.retry.interval")
            _reference(retry["require_scene"], scenes, f"{loc}.retry.require_scene")
            if retry["require_scene"] != state["require_scene"]:
                raise ValueError(f"{loc}.retry.require_scene: must match state require_scene")
    _acyclic(graph, f"task {task['task']} states")
    return task


def _compile_common(data, compiler, where):
    _mapping(data, where, ("device", "scenes", "switches"))
    if not isinstance(data["scenes"], dict):
        raise ValueError(f"{where}: scenes must be a mapping")
    scenes = {}
    for name, source in data["scenes"].items():
        _name(name, f"{where}: scene name")
        loc = f"scene {name}"
        _mapping(source, loc, ("type", "image"), ("threshold", "region", "covers", "exit"))
        scene = {k: v for k, v in source.items() if k not in ("image", "exit")}
        scene["template"] = compiler.image(source["image"], f"{loc}.image")
        if "exit" in source:
            if not isinstance(source["exit"], list):
                raise ValueError(f"{loc}.exit: expected a list")
            scene["exit"] = [compiler.action(a, f"{loc}.exit[{i}]", wait_scene=True)
                             for i, a in enumerate(source["exit"])]
        scenes[name] = scene
    if not isinstance(data["switches"], list):
        raise ValueError(f"{where}: switches must be a list")
    switches = []
    for i, source in enumerate(data["switches"]):
        loc = f"switch {i}"
        _mapping(source, loc, ("from", "to", "steps"))
        if not isinstance(source["steps"], list):
            raise ValueError(f"{loc}.steps: expected a list")
        switches.append({"from": source["from"], "to": source["to"],
                         "actions": [compiler.action(a, f"{loc}.steps[{j}]", wait_scene=True)
                                     for j, a in enumerate(source["steps"])]})
    return _scenes({"scenes": scenes, "switches": switches}, compiler.resources, where)


def _compile_task(source, compiler, scenes, path):
    _mapping(source, str(path), ("task", "start", "steps"), ("result",))
    result = source.get("result", "unverified")
    if result not in ("verified", "unverified"):
        raise ValueError(f"{path}: result must be verified or unverified")
    start = [source["start"]] if isinstance(source["start"], str) else source["start"]
    states = {}

    def add(state):
        name = f"state_{len(states) + 1:04d}"
        states[name] = state
        return name

    active = set()

    def sequence(steps, continuation, where, *, nonempty=False):
        if not isinstance(steps, list) or (nonempty and not steps):
            raise ValueError(f"{where}: expected {'a nonempty' if nonempty else 'a'} list")
        if id(steps) in active:
            raise ValueError(f"{where}: recursive steps")
        active.add(id(steps))
        try:
            for index in range(len(steps) - 1, -1, -1):
                step = steps[index]
                loc = f"{where}[{index}]"
                if not isinstance(step, dict):
                    raise ValueError(f"{loc}: expected a mapping")
                if "detect" in step:
                    _mapping(step, loc, ("detect",))
                    detect = _mapping(step["detect"], loc + ".detect", ("branches", "timeout", "on_timeout"))
                    branches = detect["branches"]
                    if not isinstance(branches, list) or not branches:
                        raise ValueError(f"{loc}.branches: expected a nonempty list")
                    transitions = []
                    for i, branch in enumerate(branches):
                        bl = f"{loc}.branches[{i}]"
                        _mapping(branch, bl, ("when",), ("result", "reason", "steps"))
                        if ("result" in branch) == ("steps" in branch):
                            raise ValueError(f"{bl}: expected exactly one result or steps")
                        condition = compiler.condition(branch["when"], bl + ".when")
                        if "steps" in branch:
                            if "reason" in branch:
                                raise ValueError(f"{bl}: reason requires result")
                            target = sequence(branch["steps"], continuation, bl + ".steps", nonempty=True)
                        else:
                            target = add({k: branch[k] for k in ("result", "reason") if k in branch})
                        transitions.append({"when": condition, "to": target})
                    continuation = add({"actions": [], "transitions": transitions,
                                        "timeout": detect["timeout"], "on_timeout": detect["on_timeout"]})
                elif "wait_scene" in step:
                    _mapping(step, loc, ("wait_scene",))
                    wait = _mapping(step["wait_scene"], loc + ".wait_scene", ("scenes", "timeout"))
                    names = _names(wait["scenes"], loc + ".scenes")
                    if len(names) != 1:
                        raise ValueError(f"{loc}: wait_scene requires exactly one scene")
                    continuation = add({"actions": [], "transitions": [{"when": {"scene": names[0]}, "to": continuation}],
                                        "timeout": wait["timeout"], "on_timeout": "uncertain"})
                else:
                    operations = [key for key in ("click", "click_image", "swipe", "back", "wait") if key in step]
                    if len(operations) != 1:
                        raise ValueError(f"{loc}: expected exactly one operation")
                    op = operations[0]
                    _mapping(step, loc, (op,), ("require_scene", "expect", "timeout", "on_timeout", "retry"))
                    control = op != "wait"
                    if control and "require_scene" not in step:
                        raise ValueError(f"{loc}: controls require require_scene")
                    if not control and any(k in step for k in ("require_scene", "expect", "timeout", "on_timeout", "retry")):
                        raise ValueError(f"{loc}: wait cannot have control or expectation fields")
                    if "expect" not in step and any(k in step for k in ("timeout", "on_timeout", "retry")):
                        raise ValueError(f"{loc}: timeout, on_timeout and retry require expect")
                    action = compiler.action({op: step[op]}, loc)
                    state = {"actions": [action]}
                    if "require_scene" in step:
                        state["require_scene"] = step["require_scene"]
                    if "expect" in step:
                        state.update(transitions=[{"when": compiler.condition(step["expect"], loc + ".expect"),
                                                   "to": continuation}],
                                     timeout=step.get("timeout", 10), on_timeout=step.get("on_timeout", "uncertain"))
                        if "retry" in step:
                            retry = _mapping(step["retry"], loc + ".retry", ("count", "interval"))
                            state["retry"] = {**retry, "require_scene": step["require_scene"]}
                    else:
                        state["next"] = continuation
                    continuation = add(state)
            return continuation
        finally:
            active.remove(id(steps))

    end = add({"result": result})
    initial = sequence(source["steps"], end, f"{path}: steps", nonempty=True)
    task = {"task": source["task"], "start_scenes": start, "initial_state": initial, "states": states}
    task = _task(task, path, compiler.resources, scenes)
    if result == "verified":
        # A matching condition is evidence, but is not necessarily independent evidence of success.
        last = source["steps"][-1]
        if "wait_scene" not in last and not (
                "expect" in last and any(op in last for op in ("click", "click_image", "swipe", "back"))):
            raise ValueError(f"{path}: verified result requires a final expect or wait_scene step")
    return task


def _compile_loop(base, source, compiler, scenes, path):
    """Validate a repeat-until-success loop; the looped task is compiled once."""
    _mapping(source, f"{path}: loop", ("count", "file", "success"),
             ("on_failure", "wait", "retry", "max_retries"))
    if type(source["count"]) is not int or source["count"] < 1:
        raise ValueError(f"{path}: loop.count must be a positive integer")
    on_failure = source.get("on_failure", "stop")
    if on_failure not in ("stop", "skip"):
        raise ValueError(f"{path}: loop.on_failure must be stop or skip")
    wait = source.get("wait", 0)
    _number(wait, f"{path}: loop.wait", zero=True)
    success = _mapping(source["success"], f"{path}: loop.success", ("result",), ("reason",))
    if success["result"] not in ("verified", "unverified", "suspended"):
        raise ValueError(f"{path}: loop.success.result must be verified, unverified or suspended")
    if "reason" in success and not isinstance(success["reason"], str):
        raise ValueError(f"{path}: loop.success.reason must be a string")
    retries = []
    if "retry" in source:
        if not isinstance(source["retry"], list):
            raise ValueError(f"{path}: loop.retry must be a list")
        results = set()
        for index, item in enumerate(source["retry"]):
            loc = f"{path}: loop.retry[{index}]"
            _mapping(item, loc, ("result",), ("wait",))
            if item["result"] not in ("unverified", "suspended"):
                raise ValueError(f"{loc}.result: must be unverified or suspended")
            if item["result"] in results:
                raise ValueError(f"{loc}.result: duplicated retry result {item['result']!r}")
            results.add(item["result"])
            _number(item.get("wait", 0), f"{loc}.wait", zero=True)
            retries.append({"result": item["result"], "wait": item.get("wait", 0)})
    max_retries = source.get("max_retries", 0)
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError(f"{path}: loop.max_retries must be a nonnegative integer")
    if retries and max_retries == 0:
        raise ValueError(f"{path}: loop.max_retries must be positive when loop.retry is set")
    task_path = _path(base, source["file"], f"{path}: loop.file")
    task = _compile_task(_read_yaml(task_path), compiler, scenes, task_path)
    return {"count": source["count"], "on_failure": on_failure, "wait": wait,
            "success": {"result": success["result"], "reason": success.get("reason", "")},
            "retry": retries, "max_retries": max_retries, "task": task}


def load_plan(list_path, home=None) -> Plan:
    """Load and validate all configuration and images without connecting a device.

    The list holds exactly one of 'tasks' (an ordered one-shot list) or 'loop'
    (repeat one task until enough successes are reached).
    """
    try:
        base = PROJECT_HOME if home is None else Path(home).resolve()
        path = resolve_home_path(list_path, base)
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid list path: {list_path!r}") from exc
    data = _mapping(_read_yaml(path), str(path), ("common",), ("tasks", "loop", "scrcpy"))
    if ("tasks" in data) == ("loop" in data):
        raise ValueError(f"{path}: exactly one of 'tasks' or 'loop' is required")
    if isinstance(data.get("scrcpy"), dict) and "turn_screen_off" in data["scrcpy"]:
        raise ValueError(f"{path}: scrcpy.turn_screen_off is obsolete; replace it with the inverse scrcpy.screen_on")
    scrcpy = _mapping(data.get("scrcpy", {}), f"{path}: scrcpy",
                      optional=("screen_on", "stay_awake"))
    for key, value in scrcpy.items():
        if type(value) is not bool:
            raise ValueError(f"{path}: scrcpy.{key} must be a boolean")
    common_path = _path(base, data["common"], f"{path}: common")
    common = _mapping(_read_yaml(common_path), str(common_path), ("device", "scenes", "switches"))
    compiler = _Compiler(base, common["device"])
    scenes, switches = _compile_common(common, compiler, str(common_path))
    if "loop" in data:
        loop = _compile_loop(base, data["loop"], compiler, scenes, str(path))
        return Plan(compiler.resources, scenes, switches, [], loop=loop, scrcpy=dict(scrcpy))
    if not isinstance(data["tasks"], list) or not data["tasks"]:
        raise ValueError(f"{path}: tasks must be a nonempty list")
    tasks = []
    for index, entry in enumerate(data["tasks"]):
        loc = f"{path}: tasks[{index}]"
        _mapping(entry, loc, ("file", "on_failure"))
        if entry["on_failure"] not in ("stop", "skip"):
            raise ValueError(f"{loc}: on_failure must be stop or skip")
        task_path = _path(base, entry["file"], f"{loc}.file")
        task = _compile_task(_read_yaml(task_path), compiler, scenes, task_path)
        tasks.append({"name": task["task"], "on_failure": entry["on_failure"], "task": task})
    return Plan(compiler.resources, scenes, switches, tasks, scrcpy=dict(scrcpy))


def validate_sizes(plan: Plan, snapshot) -> None:
    """Compare every annotation against the current snapshot; call again on size changes."""
    for attr in ("frame_size", "device_size"):
        value = getattr(snapshot, attr, None)
        if not isinstance(value, (tuple, list)) or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
            raise ValueError(f"snapshot.{attr}: expected two positive integers")
    for name, record in plan.resources.items():
        for attr in ("frame_size", "device_size"):
            annotated = _size(record.get(attr), f"resource {name}.{attr}")
            actual = tuple(getattr(snapshot, attr))
            if annotated != actual:
                raise ValueError(f"resource {name}.{attr}: annotated {annotated}, snapshot {actual}")
