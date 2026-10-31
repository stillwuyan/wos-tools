from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np


@dataclass(frozen=True)
class Match:
    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class SceneConflict(ValueError):
    """Multiple scenes match without a unique covering scene."""


class Detector:
    def __init__(self, resources: dict, scenes: dict):
        self.resources = resources
        self.scenes = scenes
        self.trace_observer = None
        self._trace_context = None

        # Validate the graph even when some of its scenes do not match a frame.
        visited = set()
        active = set()

        for name in scenes:
            if name in visited:
                continue
            pending = [(name, False)]
            while pending:
                current, leaving = pending.pop()
                if leaving:
                    active.remove(current)
                    visited.add(current)
                    continue
                if current in active:
                    raise ValueError(f"Cyclic scene covers: {current}")
                if current in visited:
                    continue
                scene = scenes[current]
                if scene.get("type") not in ("main", "sub"):
                    raise ValueError(f"Invalid scene type: {current}")
                active.add(current)
                pending.append((current, True))
                for covered in reversed(scene.get("covers", [])):
                    if covered not in scenes:
                        raise ValueError(f"Unknown covered scene: {covered}")
                    pending.append((covered, False))

    def _match_scores(self, snapshot, resource_name, threshold, region):
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be finite and between -1 and 1")

        resource = self.resources[resource_name]
        template = resource["grayscale"]
        if template.min() == template.max():
            raise ValueError(f"Constant template has no texture: {resource['image']}")

        frame = snapshot.frame
        height, width = frame.shape[:2]
        if region is None:
            x, y, w, h = 0, 0, width, height
        else:
            if (len(region) != 4 or any(not isinstance(value, int) or isinstance(value, bool) for value in region)):
                raise ValueError("region must contain four integer coordinates")
            x, y, w, h = region
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError("region must be nonempty and within the frame")

        template_height, template_width = template.shape
        if template_width > w or template_height > h:
            return None
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        return scores, x, y, template_width, template_height

    def match(self, snapshot, resource_name, threshold=0.8, region=None) -> Match | None:
        try:
            result = self._match_scores(snapshot, resource_name, threshold, region)
            if result is None:
                self._notify_match(snapshot, resource_name, "match", threshold, region,
                                   score=None, accepted=False, reason="template_larger_than_region")
                return None
            scores, x, y, template_width, template_height = result
            _, score, _, (match_x, match_y) = cv2.minMaxLoc(scores)
            accepted = math.isfinite(score) and score >= threshold
            match = (Match(x + match_x, y + match_y, template_width, template_height, float(score))
                     if accepted else None)
            self._notify_match(snapshot, resource_name, "match", threshold, region,
                               score=score, accepted=accepted,
                               rect=(x + match_x, y + match_y, template_width, template_height))
            return match
        except Exception as exc:
            self._notify_match(snapshot, resource_name, "match", threshold, region,
                               accepted=False, error=exc)
            raise

    def match_unique(self, snapshot, resource_name, threshold=0.8, region=None) -> Match | None:
        """Return the best match only if there are no distinct nearby or distant peaks.

        Very close targets can still be indistinguishable from one object's response;
        broad or noisy responses can also produce false ambiguity.
        """
        try:
            result = self._match_scores(snapshot, resource_name, threshold, region)
            if result is None:
                self._notify_match(snapshot, resource_name, "match_unique", threshold, region,
                                   score=None, accepted=False, reason="template_larger_than_region",
                                   candidate_count=0, peak_count=0)
                return None
            scores, x, y, width, height = result
            candidates = np.isfinite(scores) & (scores >= threshold)
            candidate_count = int(np.count_nonzero(candidates))
            if not candidates.any():
                _, best, _, (match_x, match_y) = cv2.minMaxLoc(scores)
                self._notify_match(snapshot, resource_name, "match_unique", threshold, region,
                                   score=best, accepted=False, reason="below_threshold",
                                   rect=(x + match_x, y + match_y, width, height),
                                   candidate_count=0, peak_count=0)
                return None

            _, score, _, (match_x, match_y) = cv2.minMaxLoc(scores, mask=candidates.astype(np.uint8))
            finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
            local_max = cv2.dilate(finite_scores, np.ones((3, 3), dtype=np.uint8))
            peaks = candidates & (finite_scores == local_max)
            # Merge only close peaks relative to the strongest, never transitively.
            peak_y, peak_x = np.nonzero(peaks)
            rect = (x + match_x, y + match_y, width, height)
            if np.any((np.abs(peak_x - match_x) * 4 > width)
                      | (np.abs(peak_y - match_y) * 4 > height)):
                self._notify_match(snapshot, resource_name, "match_unique", threshold, region,
                                   score=score, accepted=False, reason="multiple_matches", rect=rect,
                                   candidate_count=candidate_count, peak_count=len(peak_x))
                raise ValueError(f"Multiple matches for resource: {resource_name}")
            match = Match(*rect, float(score))
            self._notify_match(snapshot, resource_name, "match_unique", threshold, region,
                               score=score, accepted=True, rect=rect,
                               candidate_count=candidate_count, peak_count=len(peak_x))
            return match
        except Exception as exc:
            if not (isinstance(exc, ValueError) and str(exc).startswith("Multiple matches for resource:")):
                self._notify_match(snapshot, resource_name, "match_unique", threshold, region,
                                   accepted=False, error=exc)
            raise

    def _notify_match(self, snapshot, resource_name, method, threshold, region, *,
                      score=None, accepted=False, rect=None, reason=None, error=None,
                      candidate_count=None, peak_count=None):
        observer = self.trace_observer
        if observer is None:
            return
        resource = self.resources.get(resource_name, {})
        detail = {
            "frame_id": getattr(snapshot, "frame_id", None),
            "context": self._trace_context,
            "method": method,
            "resource": resource_name,
            "image": str(resource.get("image", "")),
            "threshold": threshold,
            "region": list(region) if region is not None else None,
            "score": float(score) if score is not None and math.isfinite(float(score)) else None,
            "accepted": bool(accepted),
            "rect": list(rect) if rect is not None else None,
            "center": [rect[0] + rect[2] // 2, rect[1] + rect[3] // 2] if rect else None,
        }
        if candidate_count is not None:
            detail["candidate_count"] = candidate_count
        if peak_count is not None:
            detail["peak_count"] = peak_count
        if reason is not None:
            detail["reason"] = reason
        if error is not None:
            detail["error"] = f"{type(error).__name__}: {error}"
        try:
            observer(detail)
        except Exception:
            # Trace failures must not change matching or automation semantics.
            pass

    def detect(self, snapshot) -> str | None:
        # Freeze the supplied frame once, including when the caller can mutate it.
        frozen = SimpleNamespace(frame=snapshot.frame.copy(),
                                 frame_id=getattr(snapshot, "frame_id", None))
        matched = set()
        previous_context = self._trace_context
        try:
            for name, scene in self.scenes.items():
                self._trace_context = {"kind": "scene", "scene": name}
                if self.match(frozen, scene["template"], scene.get("threshold", 0.8),
                              scene.get("region")) is not None:
                    matched.add(name)
        finally:
            self._trace_context = previous_context
        covered = set()
        for name in matched:
            pending = list(self.scenes[name].get("covers", []))
            seen = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(self.scenes[current].get("covers", []))
            covered.update(seen)
        remaining = matched - covered
        if len(remaining) > 1:
            raise SceneConflict(f"Conflicting scenes: {', '.join(sorted(remaining))}")
        return next(iter(remaining), None)
