import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from automation.vision import Detector, Match, SceneConflict


class DetectorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.template = np.array(
            [[15, 90, 23, 55], [47, 210, 72, 131], [199, 31, 166, 3]], dtype=np.uint8
        )
        self.path = self.directory / "template.png"
        self.assertTrue(cv2.imwrite(str(self.path), self.template))
        self.resources = {"button": {"action": "template", "image": self.path, "grayscale": self.template}}
        self.detector = Detector(self.resources, {})
        self.frame = np.zeros((20, 25, 3), dtype=np.uint8)
        self.frame[8:11, 12:16] = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        self.snapshot = SimpleNamespace(frame=self.frame)

    def test_match_coordinates_center_and_threshold(self):
        result = self.detector.match(self.snapshot, "button")
        self.assertIsInstance(result, Match)
        self.assertEqual((result.x, result.y, result.width, result.height), (12, 8, 4, 3))
        self.assertEqual(result.center, (14, 9))
        self.assertAlmostEqual(result.score, 1.0, places=5)
        damaged = self.frame.copy()
        damaged[8, 12] = 255
        snapshot = SimpleNamespace(frame=damaged)
        self.assertIsNotNone(self.detector.match(snapshot, "button", threshold=0.5))
        self.assertIsNone(self.detector.match(snapshot, "button", threshold=0.99))
        for threshold in (float("nan"), float("inf"), 1.1, -1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.detector.match(self.snapshot, "button", threshold=threshold)

    def test_region_limits_search_and_offsets_coordinates(self):
        result = self.detector.match(self.snapshot, "button", region=[10, 7, 8, 5])
        self.assertEqual((result.x, result.y, result.center), (12, 8, (14, 9)))
        self.assertIsNone(self.detector.match(self.snapshot, "button", region=[0, 0, 10, 7]))
        self.assertIsNone(self.detector.match(self.snapshot, "button", region=[12, 8, 3, 3]))
        self.assertIsNone(self.detector.match(SimpleNamespace(frame=self.frame[:2, :3]), "button"))
        for region in ([-1, 0, 5, 5], [0, 0, 26, 5], [0, 19, 5, 2], [0, 0, 0, 5], [0, 0, 2.5, 5], [0, 0, 5]):
            with self.subTest(region=region), self.assertRaises(ValueError):
                self.detector.match(self.snapshot, "button", region=region)

    def test_match_unique_separate_targets_and_region(self):
        self.frame[2:5, 3:7] = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        with self.assertRaisesRegex(ValueError, "Multiple matches.*button"):
            self.detector.match_unique(self.snapshot, "button")
        self.assertIsNotNone(self.detector.match(self.snapshot, "button"))
        result = self.detector.match_unique(self.snapshot, "button", region=[10, 7, 8, 5])
        self.assertEqual((result.x, result.y, result.center), (12, 8, (14, 9)))
        self.assertIsNone(self.detector.match_unique(self.snapshot, "button", region=[0, 5, 8, 5]))
        self.assertIsNone(self.detector.match_unique(self.snapshot, "button", region=[12, 8, 3, 3]))

    def test_match_unique_threshold_and_validation(self):
        damaged = self.template.copy()
        damaged[0, 0] = 255
        self.frame[2:5, 3:7] = cv2.cvtColor(damaged, cv2.COLOR_GRAY2BGR)
        with self.assertRaises(ValueError):
            self.detector.match_unique(self.snapshot, "button", threshold=0.5)
        result = self.detector.match_unique(self.snapshot, "button", threshold=0.99)
        self.assertEqual((result.x, result.y), (12, 8))
        self.frame[:] = 0
        self.assertIsNone(self.detector.match_unique(self.snapshot, "button"))
        for threshold in (float("nan"), float("inf"), 1.1, -1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.detector.match_unique(self.snapshot, "button", threshold=threshold)
        for region in ([-1, 0, 5, 5], [0, 0, 26, 5], [0, 0, 2.5, 5], [0, 0, 5]):
            with self.subTest(region=region), self.assertRaises(ValueError):
                self.detector.match_unique(self.snapshot, "button", region=region)

    def test_match_unique_suppresses_dense_overlapping_peaks(self):
        scores = np.zeros((15, 20), dtype=np.float32)
        scores[5:10, 7:12] = 0.9
        scores[7, 9] = 1.0
        with patch.object(self.detector, "_match_scores", return_value=(scores, 3, 4, 10, 10)):
            result = self.detector.match_unique(self.snapshot, "button")
        self.assertEqual((result.x, result.y, result.width, result.height), (12, 11, 10, 10))
        self.assertAlmostEqual(result.score, 1.0)

    def test_match_unique_with_real_dense_peaks(self):
        rng = np.random.default_rng(42)
        template = cv2.GaussianBlur(rng.integers(0, 256, (25, 25), dtype=np.uint8), (9, 9), 2)
        self.resources["button"]["grayscale"] = template
        frame = rng.integers(0, 256, (60, 70), dtype=np.uint8)
        frame[20:45, 30:55] = template
        scores = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        self.assertGreater(np.count_nonzero(scores >= 0.5), 1)
        snapshot = SimpleNamespace(frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        result = self.detector.match_unique(snapshot, "button", threshold=0.5)
        self.assertEqual((result.x, result.y, result.width, result.height), (30, 20, 25, 25))

    def test_match_unique_rejects_overlapping_local_peaks(self):
        for dx, dy in ((5, 0), (0, 5), (3, 0)):
            with self.subTest(dx=dx, dy=dy):
                scores = np.zeros((15, 15), dtype=np.float32)
                scores[4, 4] = 1.0
                scores[4 + dy, 4 + dx] = 0.95
                with patch.object(self.detector, "_match_scores", return_value=(scores, 0, 0, 10, 10)):
                    with self.assertRaisesRegex(ValueError, "Multiple matches"):
                        self.detector.match_unique(self.snapshot, "button")

    def test_match_unique_rejects_real_overlapping_targets(self):
        rng = np.random.default_rng(42)
        half = rng.integers(0, 256, (10, 5), dtype=np.uint8)
        template = np.tile(half, (1, 2))
        frame = np.tile(half, (1, 3))
        self.resources["button"]["grayscale"] = template
        snapshot = SimpleNamespace(frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        with self.assertRaisesRegex(ValueError, "Multiple matches"):
            self.detector.match_unique(snapshot, "button", threshold=0.99)

    def test_match_unique_merges_nearby_continuous_peaks(self):
        scores = np.zeros((10, 10), dtype=np.float32)
        scores[5, 3:6] = [1.0, 0.85, 0.95]
        with patch.object(self.detector, "_match_scores", return_value=(scores, 0, 0, 10, 10)):
            result = self.detector.match_unique(self.snapshot, "button")
        self.assertEqual((result.x, result.y), (3, 5))

    def test_match_unique_includes_threshold_and_ignores_nonfinite_scores(self):
        scores = np.array([[float("nan"), 0.5, float("inf")]], dtype=np.float32)
        with patch.object(self.detector, "_match_scores", return_value=(scores, 0, 0, 4, 3)):
            result = self.detector.match_unique(self.snapshot, "button", threshold=0.5)
        self.assertEqual((result.x, result.y, result.score), (1, 0, 0.5))

    def test_match_unique_does_not_merge_bridged_or_touching_targets(self):
        scores = np.zeros((5, 25), dtype=np.float32)
        scores[2, 2:13] = 0.85
        scores[2, 2] = 1.0
        scores[2, 12] = 0.95
        with patch.object(self.detector, "_match_scores", return_value=(scores, 0, 0, 10, 10)):
            with self.assertRaisesRegex(ValueError, "Multiple matches"):
                self.detector.match_unique(self.snapshot, "button")
        self.frame[:] = 0
        self.frame[8:11, 4:8] = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        self.frame[8:11, 8:12] = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        with self.assertRaisesRegex(ValueError, "Multiple matches"):
            self.detector.match_unique(self.snapshot, "button", threshold=0.99)

    def test_constant_grayscale_template_is_rejected(self):
        for image in (np.full((3, 4), 123, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)):
            with self.subTest(shape=image.shape):
                self.resources["button"]["grayscale"] = image
                with self.assertRaisesRegex(ValueError, "Constant template"):
                    self.detector.match(self.snapshot, "button")

    def test_cached_template_does_not_read_file(self):
        self.path.unlink()
        with patch("automation.vision.np.fromfile", side_effect=AssertionError("unexpected file read")), \
                patch("automation.vision.cv2.imdecode", side_effect=AssertionError("unexpected decode")):
            result = self.detector.match(self.snapshot, "button", threshold=0.99)
            unique = self.detector.match_unique(self.snapshot, "button", threshold=0.99)
        self.assertEqual((result.x, result.y), (12, 8))
        self.assertEqual((unique.x, unique.y), (12, 8))

    def test_grayscale_is_required_even_when_image_exists(self):
        del self.resources["button"]["grayscale"]
        with patch("automation.vision.np.fromfile", side_effect=AssertionError("unexpected file read")), \
                self.assertRaisesRegex(KeyError, "grayscale"):
            self.detector.match(self.snapshot, "button")

    def test_detect_none_single_and_scene_region_threshold(self):
        scenes = {"home": {"type": "main", "template": "button", "region": [0, 0, 10, 7]}}
        detector = Detector(self.resources, scenes)
        self.assertIsNone(detector.detect(self.snapshot))
        scenes["home"]["region"] = [10, 7, 8, 5]
        self.assertEqual(detector.detect(self.snapshot), "home")
        damaged = self.frame.copy()
        damaged[8, 12] = 255
        scenes["home"]["threshold"] = 0.99
        self.assertIsNone(detector.detect(SimpleNamespace(frame=damaged)))

    def test_multiple_matches_conflict_even_with_different_scores(self):
        other = self.template.copy()
        other[0, 0] = 255
        path = self.directory / "other.png"
        cv2.imwrite(str(path), other)
        self.resources["other"] = {"image": path, "grayscale": other}
        scenes = {
            "home": {"type": "main", "template": "button"},
            "dialog": {"type": "sub", "template": "other", "threshold": 0.5},
        }
        with self.assertRaisesRegex(SceneConflict, "dialog.*home"):
            Detector(self.resources, scenes).detect(self.snapshot)

    def test_specific_scene_covering_generic_banner_wins_over_it(self):
        # Mirrors the polar_terror flow: a specific popup frame also shows a generic
        # sub-task banner. Declaring the banner in the popup's covers resolves the
        # same-frame double match to the specific scene; without it, they conflict.
        other = self.template.copy()
        other[0, 0] = 255
        path = self.directory / "banner.png"
        cv2.imwrite(str(path), other)
        self.resources["banner"] = {"image": path, "grayscale": other}
        frame = self.frame.copy()
        frame[2:5, 3:7] = cv2.cvtColor(other, cv2.COLOR_GRAY2BGR)
        snapshot = SimpleNamespace(frame=frame)
        scenes = {
            "popup": {"type": "sub", "template": "button", "covers": ["banner_scene"]},
            "banner_scene": {"type": "sub", "template": "banner"},
        }
        self.assertEqual(Detector(self.resources, scenes).detect(snapshot), "popup")
        scenes["popup"].pop("covers")
        with self.assertRaisesRegex(SceneConflict, "banner_scene.*popup"):
            Detector(self.resources, scenes).detect(snapshot)

    def test_cover_is_transitive_through_unmatched_scene(self):
        scenes = {
            "home": {"type": "main", "template": "button"},
            "middle": {"type": "sub", "template": "button", "region": [0, 0, 4, 3], "covers": ["home"]},
            "dialog": {"type": "sub", "template": "button", "covers": ["middle"]},
        }
        detector = Detector(self.resources, scenes)
        self.assertEqual(detector.detect(self.snapshot), "dialog")
        scenes["middle"].pop("region")
        self.assertEqual(detector.detect(self.snapshot), "dialog")
        scenes["other"] = {"type": "sub", "template": "button"}
        with self.assertRaises(SceneConflict):
            detector.detect(self.snapshot)

    def test_invalid_cover_graph_rejected(self):
        for scenes in (
            {"a": {"type": "main", "template": "button", "covers": ["missing"]}},
            {"a": {"type": "main", "template": "button", "covers": ["a"]}},
            {"a": {"type": "main", "template": "button", "covers": ["b"]},
             "b": {"type": "sub", "template": "button", "covers": ["a"]}},
            {"a": {"type": "unknown", "template": "button"}},
        ):
            with self.subTest(scenes=scenes), self.assertRaises(ValueError):
                Detector(self.resources, scenes)

    def test_deep_cover_graph_validation(self):
        scenes = {
            str(index): {"type": "sub", "template": "button", "covers": [str(index + 1)]}
            for index in range(3000)
        }
        scenes["3000"] = {"type": "main", "template": "button"}
        Detector(self.resources, scenes)
        scenes["3000"]["covers"] = ["1500"]
        with self.assertRaisesRegex(ValueError, "Cyclic scene covers"):
            Detector(self.resources, scenes)
        scenes["3000"]["covers"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Unknown covered scene"):
            Detector(self.resources, scenes)

    def test_detect_uses_one_frozen_frame(self):
        scenes = {
            "first": {"type": "main", "template": "button"},
            "second": {"type": "sub", "template": "button"},
        }
        detector = Detector(self.resources, scenes)
        original_match = detector.match
        frames = []

        def match(snapshot, *args):
            frames.append(snapshot.frame)
            self.frame[:] = 0
            return original_match(snapshot, *args)

        with patch.object(detector, "match", side_effect=match):
            with self.assertRaises(SceneConflict):
                detector.detect(self.snapshot)
        self.assertIs(frames[0], frames[1])
        self.assertIsNot(frames[0], self.frame)


if __name__ == "__main__":
    unittest.main()
