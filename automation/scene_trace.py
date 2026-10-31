"""Optional scene-recognition tracing for `run --trace-scenes`.

Wraps the real Detector: every detect() call is logged with the per-scene
template scores (one JSONL row per frame), a screenshot plus a sidecar JSON is
saved whenever the matched set or the resolution changes (and always on a
scene conflict), and the run ends with a covers audit that reports how many
real frames actually exercised each `covers` declaration.

Scoring is done by spying on the detector's own _match_scores during the single
normal detection pass, so each template is matched exactly once (no extra
full-frame matching, keeping action frames fresh) and the recorded per-scene
score is the real best score, including scenes that fell below threshold.

The original detection semantics are preserved: SceneConflict is recorded and
then re-raised, so the runtime still stops exactly as it would without tracing.
"""
from __future__ import annotations

import json
import math
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
        # The detector resolves scenes through its own match(); spying on that
        # method captures every real per-scene score during the single
        # detection pass, instead of matching every template a second time.
        self._template_to_scenes = {}
        for name, scene in scenes.items():
            self._template_to_scenes.setdefault(scene["template"], []).append(name)
        self._spied = None

    def detect(self, snapshot):
        self._spied = {}
        self._scores_original = self.base._match_scores
        self.base._match_scores = self._spying_scores
        try:
            result = self.base.detect(snapshot)
            conflict = None
        except Exception as exc:
            result = None
            conflict = exc
        finally:
            self.base._match_scores = self._scores_original
        matched = set()
        for name in self.scenes:
            scene = self.scenes[name]
            best = self._spied.get(name)
            if best is not None and best >= scene.get("threshold", 0.8):
                matched.add(name)
        scores = {name: (round(score, 4) if score is not None else None)
                  for name, score in sorted(self._spied.items())}
        result_type = "conflict" if conflict is not None else "scene"
        covered = matched - ({result} if result is not None else set())
        self._counts.update(matched)
        for name in matched:
            for covered_name in self.scenes[name].get("covers", []):
                if covered_name in matched:
                    self._cover_hits[name, covered_name] += 1
                else:
                    self._cover_alone[covered_name] += 1
        row = self._record(snapshot, result, result_type, sorted(matched), sorted(covered), scores)
        if conflict is not None or sorted(matched) != self._last:
            self._save_snapshot(snapshot, row)
        self._last = sorted(matched)
        if conflict is not None:
            raise conflict
        return result

    def _spying_scores(self, snapshot, resource_name, threshold, region):
        result = self._scores_original(snapshot, resource_name, threshold, region)
        if result is not None:
            score_array = result[0]
            _, best, _, _ = cv2.minMaxLoc(score_array)
            score = round(float(best), 4) if math.isfinite(best) else None
        else:
            score = None
        for scene_name in self._template_to_scenes.get(resource_name, ()):
            # Each detect() pass calls _match_scores once per scene template,
            # so the last spy hit per scene is that pass's score.
            self._spied[scene_name] = score
        return result

    def match(self, snapshot, resource_name, threshold=0.8, region=None):
        return self.base.match(snapshot, resource_name, threshold, region)

    def match_unique(self, snapshot, resource_name, threshold=0.8, region=None):
        return self.base.match_unique(snapshot, resource_name, threshold, region)

    def write_summary(self):
        lines = [f"Scene trace summary: {self._records} observations, {self._shots} screenshots in {self.dir}"]
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

    def _record(self, snapshot, scene, result_type, matched, covered, scores):
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
