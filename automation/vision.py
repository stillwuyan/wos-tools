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
        result = self._match_scores(snapshot, resource_name, threshold, region)
        if result is None:
            return None
        scores, x, y, template_width, template_height = result
        _, score, _, (match_x, match_y) = cv2.minMaxLoc(scores)
        if not math.isfinite(score) or score < threshold:
            return None
        return Match(x + match_x, y + match_y, template_width, template_height, float(score))

    def match_unique(self, snapshot, resource_name, threshold=0.8, region=None) -> Match | None:
        """Return the best match only if there are no distinct nearby or distant peaks.

        Very close targets can still be indistinguishable from one object's response;
        broad or noisy responses can also produce false ambiguity.
        """
        result = self._match_scores(snapshot, resource_name, threshold, region)
        if result is None:
            return None
        scores, x, y, width, height = result
        candidates = np.isfinite(scores) & (scores >= threshold)
        if not candidates.any():
            return None

        _, score, _, (match_x, match_y) = cv2.minMaxLoc(scores, mask=candidates.astype(np.uint8))
        finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
        local_max = cv2.dilate(finite_scores, np.ones((3, 3), dtype=np.uint8))
        peaks = candidates & (finite_scores == local_max)
        # Merge only close peaks relative to the strongest, never transitively.
        peak_y, peak_x = np.nonzero(peaks)
        if np.any((np.abs(peak_x - match_x) * 4 > width)
                  | (np.abs(peak_y - match_y) * 4 > height)):
            raise ValueError(f"Multiple matches for resource: {resource_name}")
        return Match(x + match_x, y + match_y, width, height, float(score))

    def detect(self, snapshot) -> str | None:
        # Freeze the supplied frame once, including when the caller can mutate it.
        frozen = SimpleNamespace(frame=snapshot.frame.copy())
        matched = {
            name for name, scene in self.scenes.items()
            if self.match(frozen, scene["template"], scene.get("threshold", 0.8), scene.get("region")) is not None
        }
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
