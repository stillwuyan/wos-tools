"""Optional scene-recognition tracing for `run --trace-scenes`.

Wraps the real Detector: every detect() call is logged with the per-scene
template scores (one JSONL row per frame), a screenshot plus a sidecar JSON is
saved whenever the matched set or the resolution changes (and always on a
scene conflict), and the run ends with a covers audit that reports how many
real frames actually exercised each `covers` declaration.

Every detector call is observed in-place, so each template is matched exactly
once. Scene observations include all per-scene match details; other template
checks and device controls are written chronologically to trace-events.jsonl.

The original detection semantics are preserved: SceneConflict is recorded and
then re-raised, so the runtime still stops exactly as it would without tracing.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cv2


class TraceDetector:
    def __init__(self, base, scenes, resources, output_dir):
        self.base = base
        self.scenes = scenes
        self.resources = resources
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._records = 0
        self._shots = 0
        self._counts = Counter()
        self._cover_hits = Counter()
        self._cover_alone = Counter()
        self._last = None
        self._scene_matches = None
        self._event_seq = 0
        self._event_counts = Counter()
        self._context = {}
        self.base.trace_observer = self._on_match

    def set_context(self, **context):
        self._context = {key: value for key, value in context.items() if value is not None}

    def trace_event(self, event, **details):
        self._write_event(event, details)

    def detect(self, snapshot):
        self._scene_matches = []
        try:
            result = self.base.detect(snapshot)
            conflict = None
        except Exception as exc:
            result = None
            conflict = exc
        finally:
            template_matches = self._scene_matches
            self._scene_matches = None
        matched = {entry["context"]["scene"] for entry in template_matches
                   if entry.get("context", {}).get("scene") and entry.get("accepted")}
        scores = {entry["context"]["scene"]: entry.get("score")
                  for entry in template_matches if entry.get("context", {}).get("scene")}
        result_type = "conflict" if conflict is not None else "scene"
        covered = matched - ({result} if result is not None else set())
        self._counts.update(matched)
        for name in matched:
            for covered_name in self.scenes[name].get("covers", []):
                if covered_name in matched:
                    self._cover_hits[name, covered_name] += 1
                else:
                    self._cover_alone[covered_name] += 1
        row = self._record(snapshot, result, result_type, sorted(matched), sorted(covered), scores,
                           template_matches)
        self._write_event("scene_observation", row)
        if conflict is not None or sorted(matched) != self._last:
            self._save_snapshot(snapshot, row)
        self._last = sorted(matched)
        if conflict is not None:
            raise conflict
        return result

    def _on_match(self, detail):
        context = {**self._context, **(detail.get("context") or {})}
        item = {**detail, "context": context}
        if self._scene_matches is not None and context.get("kind") == "scene":
            self._scene_matches.append(item)
        else:
            self._write_event("template_match", item)

    def match(self, snapshot, resource_name, threshold=0.8, region=None):
        return self.base.match(snapshot, resource_name, threshold, region)

    def match_unique(self, snapshot, resource_name, threshold=0.8, region=None):
        return self.base.match_unique(snapshot, resource_name, threshold, region)

    def _write_event(self, event, details):
        self._event_seq += 1
        self._event_counts[event] += 1
        row = {
            "event_seq": self._event_seq,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "monotonic_time": time.monotonic(),
            "event": event,
            "context": self._context.copy(),
            **details,
        }
        with (self.dir / "trace-events.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def write_summary(self):
        lines = [f"Scene trace summary: {self._records} observations, {self._shots} screenshots in {self.dir}"]
        lines.append("trace events: " + ", ".join(
            f"{name}={count}" for name, count in sorted(self._event_counts.items())))
        lines.append("matched scenes: " + ", ".join(
            f"{name}={count}" for name, count in sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))))
        lines.append("covers audit (A covers B, both matched on the same frame):")
        declarations = sorted(
            (name, covered)
            for name, scene in self.scenes.items()
            for covered in scene.get("covers", []))
        for name, covered in declarations:
            hit = self._cover_hits.get((name, covered), 0)
            note = ""
            if not hit:
                note = "  (never exercised; covered scene matched alone: " \
                    f"{self._cover_alone.get(covered, 0)} frames)"
            lines.append(f"{name} covers {covered}: {hit} frames{note}")
        text = "\n".join(lines)
        (self.dir / "scene-trace-summary.txt").write_text(text + "\n", encoding="utf-8")
        return text

    def _record(self, snapshot, scene, result_type, matched, covered, scores, template_matches):
        self._records += 1
        row = {
            "seq": self._records,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_id": getattr(snapshot, "frame_id", None),
            "scene": scene,
            "result": result_type,
            "matched": matched,
            "covered": covered,
            "scores": scores,
            "template_matches": template_matches,
        }
        with (self.dir / "scene-trace.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def _save_snapshot(self, snapshot, row):
        try:
            image = snapshot.frame
        except AttributeError:
            return
        self._shots += 1
        suffix = "conflict" if row["result"] == "conflict" else (row["scene"] or "unknown")
        image_path = self.dir / f"frame_{row['seq']:06d}_{suffix}.png"
        try:
            cv2.imwrite(str(image_path), image)
            image_path.with_suffix(".json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, cv2.error):
            pass
