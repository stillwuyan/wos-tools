import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from automation.vision import Detector, SceneConflict
from automation.scene_trace import TraceDetector


def _template(directory, name, rng_seed):
    rng = np.random.default_rng(rng_seed)
    template = rng.integers(0, 256, (12, 12), dtype=np.uint8)
    path = directory / name
    cv2.imwrite(str(path), template)
    return template, path


class SceneTraceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.template, template_path = _template(self.directory, "main.png", 1)
        self.banner, banner_path = _template(self.directory, "banner.png", 2)
        self.resources = {
            "main": {"action": "template", "image": template_path, "grayscale": self.template},
            "banner": {"action": "template", "image": banner_path, "grayscale": self.banner},
        }
        self.scenes = {
            "main": {"type": "main", "template": "main"},
            "banner": {"type": "sub", "template": "banner", "covers": []},
        }
        # A frame showing both the main template and the banner.
        self.frame = np.zeros((40, 48, 3), dtype=np.uint8)
        self.frame[10:22, 10:22] = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        self.frame[2:14, 30:42] = cv2.cvtColor(self.banner, cv2.COLOR_GRAY2BGR)
        self.snapshot = SimpleNamespace(frame=self.frame, frame_id=7)

    def _trace(self, scenes, out):
        base = Detector(self.resources, scenes)
        return base, TraceDetector(base, scenes, self.resources, out)

    def test_detect_matches_base_and_records_scores(self):
        base, trace = self._trace(self.scenes, self.directory / "out")
        # Without covers the two templates conflict; with the declaration it resolves.
        self.scenes["main"]["covers"] = ["banner"]
        self.assertEqual(trace.detect(self.snapshot), "main")
        rows = [json.loads(line) for line in (self.directory / "out" / "scene-trace.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["scene"], row["result"], row["matched"], row["covered"]),
                         ("main", "scene", ["banner", "main"], ["banner"]))
        self.assertEqual(row["frame_id"], 7)
        self.assertEqual(set(row["scores"]), {"main", "banner"})
        self.assertGreaterEqual(row["scores"]["main"], 0.8)
        self.assertGreaterEqual(row["scores"]["banner"], 0.8)
        # The matched set is unchanged, so no second screenshot.
        trace.detect(self.snapshot)
        images = list((self.directory / "out").glob("*.png"))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].name, "frame_000001_main.png")

    def test_screenshot_saved_on_matched_set_change(self):
        base, trace = self._trace(self.scenes, self.directory / "out")
        self.scenes["main"]["covers"] = ["banner"]
        trace.detect(self.snapshot)
        only_main = self.frame.copy()
        only_main[2:14, 30:42] = 0
        trace.detect(SimpleNamespace(frame=only_main, frame_id=8))
        names = sorted(path.name for path in (self.directory / "out").glob("*.png"))
        self.assertEqual(names, ["frame_000001_main.png", "frame_000002_main.png"])
        sidecar = json.loads((self.directory / "out" / "frame_000002_main.json").read_text())
        self.assertEqual(sidecar["matched"], ["main"])

    def test_conflict_is_recorded_and_reraised(self):
        base, trace = self._trace(self.scenes, self.directory / "out")
        with self.assertRaisesRegex(SceneConflict, "banner.*main"):
            trace.detect(self.snapshot)
        rows = [json.loads(line) for line in (self.directory / "out" / "scene-trace.jsonl").read_text().splitlines()]
        self.assertEqual((rows[0]["result"], rows[0]["scene"], rows[0]["matched"]),
                         ("conflict", None, ["banner", "main"]))
        self.assertEqual((self.directory / "out" / "frame_000001_conflict.png").exists(), True)

    def test_cover_audit_counts_exercised_declarations(self):
        base, trace = self._trace(self.scenes, self.directory / "out")
        self.scenes["main"]["covers"] = ["banner"]
        trace.detect(self.snapshot)  # banner matched and covered: declaration exercised.
        only_main = self.frame.copy()
        only_main[2:14, 30:42] = 0
        trace.detect(SimpleNamespace(frame=only_main, frame_id=8))  # banner absent: declaration idle.
        trace.write_summary()
        written = (self.directory / "out" / "scene-trace-summary.txt").read_text()
        self.assertIn("main covers banner: 1 frames", written)
        self.assertIn("2 observations", written)
        self.assertNotIn("never exercised", written)

    def test_summary_flags_never_exercised_declaration(self):
        # A sub scene whose template never co-occurs with the banner it covers.
        ghost, ghost_path = _template(self.directory, "ghost.png", 3)
        self.resources["ghost"] = {"action": "template", "image": ghost_path, "grayscale": ghost}
        self.scenes["ghost"] = {"type": "sub", "template": "ghost", "covers": ["banner"]}
        base, trace = self._trace(self.scenes, self.directory / "out")
        only_main = self.frame.copy()
        only_main[2:14, 30:42] = 0
        trace.detect(SimpleNamespace(frame=only_main, frame_id=1))  # ghost and banner both absent.
        summary = trace.write_summary()
        self.assertIn("ghost covers banner: 0 frames", summary)
        self.assertIn("never exercised", summary)


if __name__ == "__main__":
    unittest.main()
