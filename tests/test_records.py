import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from calibration.records import MarkRecordStore
from home_paths import PROJECT_HOME


SIZES = {"frame_size": {"width": 100, "height": 200},
         "device_size": {"width": 200, "height": 400}}
POINT = {**SIZES, "frame": {"x": 10, "y": 20}, "device": {"x": 20, "y": 40}}
TEMPLATE = {**SIZES, "frame_rect": {"x1": 1, "y1": 1, "x2": 11, "y2": 11},
            "device_rect": {"x1": 2, "y1": 2, "x2": 22, "y2": 22},
            "frame_center": {"x": 5, "y": 5}, "device_center": {"x": 10, "y": 10}}


class RecordStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name).resolve()
        self.store = MarkRecordStore(Path("captures"), home=self.home)
        self.roi = np.zeros((10, 10, 3), dtype=np.uint8)

    def test_same_second_stores_are_unique_and_do_not_overwrite(self):
        timestamp = "20260101_120000"
        existing = self.store.output_dir / f"mark-records_{timestamp}_occupied.yaml"
        existing.write_text("old: data\n", encoding="utf-8")
        with patch("calibration.records.strftime", return_value=timestamp):
            with ThreadPoolExecutor(max_workers=8) as executor:
                stores = list(executor.map(lambda _: MarkRecordStore(self.store.output_dir, home=self.home), range(16)))
        paths = {store.records_file for store in stores}
        self.assertEqual(len(paths), len(stores))
        for path in paths:
            self.assertRegex(path.name, rf"^mark-records_{timestamp}_.+\.yaml$")
            self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8")), {})
        for store in stores:
            self.assertEqual(store.save_point(POINT), "step_001")
        self.assertEqual(existing.read_text(encoding="utf-8"), "old: data\n")
        self.assertEqual(yaml.safe_load(self.store.records_file.read_text(encoding="utf-8")), {})

    def test_existing_name_collision_is_not_overwritten(self):
        timestamp = "20260101_120000"
        existing = self.store.output_dir / f"mark-records_{timestamp}_occupied.yaml"
        existing.write_text("old: data\n", encoding="utf-8")
        with patch("calibration.records.strftime", return_value=timestamp), \
                patch("tempfile._get_candidate_names", return_value=iter(["occupied", "newname"])):
            store = MarkRecordStore(self.store.output_dir, home=self.home)
        self.assertEqual(store.records_file.name, f"mark-records_{timestamp}_newname.yaml")
        self.assertEqual(existing.read_text(encoding="utf-8"), "old: data\n")

    def test_initialization_failure_closes_and_removes_reserved_file(self):
        before = set(self.store.output_dir.glob("*.yaml"))
        with patch("calibration.records.os.fdopen", side_effect=OSError("open failed")):
            with self.assertRaisesRegex(OSError, "open failed"):
                MarkRecordStore(self.store.output_dir, home=self.home)
        self.assertEqual(set(self.store.output_dir.glob("*.yaml")), before)

    def test_temporary_yaml_open_failure_closes_and_removes_file(self):
        original = self.store.records_file.read_bytes()
        with patch("calibration.records.os.fdopen", side_effect=OSError("open failed")):
            with self.assertRaisesRegex(OSError, "open failed"):
                self.store.save_point(POINT)
        self.assertEqual(self.store.records_file.read_bytes(), original)
        self.assertEqual(self.store._undo_history, [])
        self.assertEqual(list(self.store.output_dir.glob("*.yaml.tmp")), [])

    def test_sizes_are_required_and_positive_in_every_save(self):
        bad_sizes = [None, {}, {"width": 1}, {"width": 0, "height": 2},
                     {"width": -1, "height": 2}, {"width": True, "height": 2},
                     {"width": 1.0, "height": 2}, {"width": 1, "height": "2"}]
        self.store.save_point(POINT)
        original = self.store.records_file.read_bytes()
        history = self.store._undo_history.copy()
        for key in SIZES:
            for value in bad_sizes:
                for action in ("point", "template"):
                    with self.subTest(key=key, value=value, action=action):
                        record = {**(POINT if action == "point" else TEMPLATE), key: value}
                        with self.assertRaisesRegex(ValueError, key):
                            if action == "point":
                                self.store.save_point(record)
                            else:
                                self.store.save_template(record, self.roi)
                        self.assertEqual(self.store.records_file.read_bytes(), original)
                        self.assertEqual(self.store._undo_history, history)
                        self.assertEqual(list(self.store.templates_dir.iterdir()), [])

    def test_sizes_are_only_in_meta_and_survive_undo(self):
        self.assertEqual(yaml.safe_load(self.store.records_file.read_text(encoding="utf-8")), {})
        first = self.store.save_template(TEMPLATE, self.roi)
        second = self.store.save_point(POINT)
        data = yaml.safe_load(self.store.records_file.read_text(encoding="utf-8"))
        self.assertEqual(data["meta"], SIZES)
        for step in (first, second):
            self.assertTrue(SIZES.keys().isdisjoint(data["steps"][step]))
        image = self.home / data["steps"][first]["image"]
        self.assertEqual(self.store.undo(), second)
        self.assertEqual(self.store.undo(), first)
        self.assertFalse(image.exists())
        self.assertEqual(yaml.safe_load(self.store.records_file.read_text(encoding="utf-8")),
                         {"meta": SIZES, "steps": {}})
        self.assertEqual(self.store.save_point(POINT), first)

    def test_changed_sizes_are_rejected_without_touching_records_or_images(self):
        first = self.store.save_template(TEMPLATE, self.roi)
        original = self.store.records_file.read_bytes()
        data = yaml.safe_load(original)
        image = self.home / data["steps"][first]["image"]
        original_image = image.read_bytes()
        history = self.store._undo_history.copy()
        for key in SIZES:
            for dimension in ("width", "height"):
                for action in ("point", "template"):
                    with self.subTest(key=key, dimension=dimension, action=action):
                        size = {**SIZES[key], dimension: SIZES[key][dimension] + 1}
                        record = {**(POINT if action == "point" else TEMPLATE), key: size}
                        with self.assertRaisesRegex(ValueError, key):
                            if action == "point":
                                self.store.save_point(record)
                            else:
                                self.store.save_template(record, self.roi)
                        self.assertEqual(self.store.records_file.read_bytes(), original)
                        self.assertEqual(image.read_bytes(), original_image)
                        self.assertEqual(list(self.store.templates_dir.iterdir()), [image])
                        self.assertEqual(self.store._undo_history, history)

        self.assertEqual(self.store.undo(), first)
        original = self.store.records_file.read_bytes()
        for key in SIZES:
            record = {**POINT, key: {**SIZES[key], "width": SIZES[key]["width"] + 1}}
            with self.assertRaisesRegex(ValueError, key):
                self.store.save_point(record)
            self.assertEqual(self.store.records_file.read_bytes(), original)
        self.assertEqual(self.store.save_point(POINT), first)

    def test_existing_steps_without_meta_are_rejected(self):
        original = "steps:\n  step_001: {}\n"
        self.store.records_file.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "meta"):
            self.store.save_point(POINT)
        self.assertEqual(self.store.records_file.read_text(encoding="utf-8"), original)

    def test_yaml_replacement_failure_preserves_records_and_history(self):
        self.store.save_point(POINT)
        original = self.store.records_file.read_bytes()
        history = self.store._undo_history.copy()
        existing = self.store.templates_dir / "existing.png"
        existing.write_bytes(b"preserve this image")

        for action in ("point", "template"):
            with self.subTest(action=action):
                with patch(
                    "calibration.records.os.replace",
                    side_effect=PermissionError("YAML replacement denied"),
                ):
                    with self.assertRaisesRegex(PermissionError, "YAML replacement denied"):
                        if action == "point":
                            self.store.save_point(POINT)
                        else:
                            self.store.save_template(TEMPLATE, self.roi)
                self.assertEqual(self.store.records_file.read_bytes(), original)
                self.assertEqual(self.store._undo_history, history)
                self.assertEqual(list(self.store.templates_dir.iterdir()), [existing])
                self.assertEqual(existing.read_bytes(), b"preserve this image")
                self.assertEqual(list(self.store.output_dir.glob("*.yaml.tmp")), [])

    def test_imwrite_failure_raises_and_removes_partial_image(self):
        self.store.save_point(POINT)
        original = self.store.records_file.read_bytes()
        history = self.store._undo_history.copy()

        def failed_write(path, roi):
            Path(path).write_bytes(b"partial image")
            return False

        with patch("calibration.records.cv2.imwrite", side_effect=failed_write):
            with self.assertRaisesRegex(OSError, "Failed to save ROI image"):
                self.store.save_template(TEMPLATE, self.roi)

        self.assertEqual(self.store.records_file.read_bytes(), original)
        self.assertEqual(self.store._undo_history, history)
        self.assertEqual(list(self.store.templates_dir.iterdir()), [])
        self.assertEqual(list(self.store.output_dir.glob("*.yaml.tmp")), [])

    def test_imwrite_exception_is_reraised_and_image_is_removed(self):
        with patch("calibration.records.cv2.imwrite", side_effect=cv2.error("Invalid ROI")):
            with self.assertRaisesRegex(cv2.error, "Invalid ROI"):
                self.store.save_template(TEMPLATE, self.roi)

        self.assertEqual(yaml.safe_load(self.store.records_file.read_text(encoding="utf-8")), {})
        self.assertEqual(self.store._undo_history, [])
        self.assertEqual(list(self.store.templates_dir.iterdir()), [])

    def test_same_second_templates_are_unique_and_preserve_existing_images(self):
        timestamp = "20260101_120000"
        existing = self.store.templates_dir / f"template_{timestamp}.png"
        existing.write_bytes(b"existing image")

        with patch("calibration.records.strftime", return_value=timestamp):
            first = self.store.save_template(TEMPLATE, self.roi)
            second = self.store.save_template(TEMPLATE, self.roi)

        data = yaml.safe_load(self.store.records_file.read_text(encoding="utf-8"))
        images = [
            self.home / data["steps"][step]["image"]
            for step in (first, second)
        ]
        self.assertEqual((first, second), ("step_001", "step_002"))
        self.assertNotEqual(images[0], images[1])
        for image in images:
            self.assertNotEqual(image, existing)
            self.assertTrue(image.name.startswith(f"template_{timestamp}_"))
            self.assertIsNotNone(cv2.imread(str(image)))
        self.assertEqual(existing.read_bytes(), b"existing image")
        self.assertEqual(len(list(self.store.templates_dir.iterdir())), 3)

    def test_template_name_collision_preserves_existing_image_on_failure_and_undo(self):
        timestamp = "20260101_120000"
        existing = self.home / "captures" / "templates" / f"template_{timestamp}_occupied.png"
        existing.write_bytes(b"existing image")
        with patch("calibration.records.strftime", return_value=timestamp), \
                patch("tempfile._get_candidate_names", return_value=iter(["occupied", "failed"])):
            with patch("calibration.records.cv2.imwrite", return_value=False):
                with self.assertRaisesRegex(OSError, "Failed to save ROI image"):
                    self.store.save_template(TEMPLATE, self.roi)
        self.assertEqual(existing.read_bytes(), b"existing image")
        self.assertEqual(list(self.store.templates_dir.iterdir()), [existing])
        with patch("calibration.records.strftime", return_value=timestamp), \
                patch("tempfile._get_candidate_names", return_value=iter(["occupied", "newname", "yamltemp"])):
            step = self.store.save_template(TEMPLATE, self.roi)
        image = self.store.templates_dir / f"template_{timestamp}_newname.png"
        self.assertTrue(image.is_file())
        self.assertEqual(self.store.undo(), step)
        self.assertFalse(image.exists())
        self.assertEqual(existing.read_bytes(), b"existing image")

    def test_relative_output_directory_saves_and_undoes_template(self):
        store = MarkRecordStore(Path("nested/captures"), home=self.home)
        self.assertEqual(store.output_dir, self.home / "nested" / "captures")
        self.assertEqual(store.records_file.parent, store.output_dir)
        self.assertEqual(store.templates_dir, self.home / "captures" / "templates")
        step = store.save_template(TEMPLATE, self.roi)

        data = yaml.safe_load(store.records_file.read_text(encoding="utf-8"))
        image_path = Path(data["steps"][step]["image"])
        self.assertFalse(image_path.is_absolute())
        self.assertEqual(image_path.parent, Path("captures/templates"))
        self.assertEqual(data["steps"][step]["image"], f"captures/templates/{image_path.name}")
        self.assertFalse((store.output_dir / "templates").exists())
        image = self.home / image_path
        np.testing.assert_array_equal(cv2.imread(str(image)), self.roi)
        self.assertEqual(store.undo(), step)
        self.assertFalse(image.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows relative paths")
    def test_ambiguous_output_is_rejected_without_writing(self):
        before = set(self.home.iterdir())
        with patch("calibration.records.tempfile.mkstemp") as mkstemp, \
                patch.object(Path, "mkdir") as mkdir:
            for value, error in (("D:captures", "drive-relative"), ("D:", "drive-relative"),
                                 (r"\captures", "root-relative"), ("/captures", "root-relative"),
                                 ("\\", "root-relative"), ("/", "root-relative")):
                for output in (value, Path(value)):
                    with self.subTest(output=output), self.assertRaisesRegex(ValueError, error):
                        MarkRecordStore(output, home=self.home)
            mkdir.assert_not_called()
            mkstemp.assert_not_called()
        self.assertEqual(set(self.home.iterdir()), before)

    def test_tilde_output_remains_a_literal_directory(self):
        store = MarkRecordStore(Path("~") / "captures", home=self.home)
        self.assertEqual(store.output_dir, self.home / "~" / "captures")

    def test_absolute_output_shares_home_images_without_deleting_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "captures"
            store = MarkRecordStore(output, home=self.home)
            self.assertEqual(store.output_dir, output)
            self.assertEqual(store.records_file.parent, output)
            self.assertEqual(store.templates_dir, self.home / "captures" / "templates")
            other_step = self.store.save_template(TEMPLATE, self.roi)
            other_data = yaml.safe_load(self.store.records_file.read_text(encoding="utf-8"))
            other_image = self.home / other_data["steps"][other_step]["image"]
            unrelated = self.home / "captures" / "templates" / "unrelated.png"
            unrelated.write_bytes(b"other file")
            step = store.save_template(TEMPLATE, self.roi)
            data = yaml.safe_load(store.records_file.read_text(encoding="utf-8"))
            image_path = Path(data["steps"][step]["image"])
            self.assertEqual(image_path.parent, Path("captures/templates"))
            self.assertEqual(data["steps"][step]["image"], f"captures/templates/{image_path.name}")
            self.assertFalse((output / "templates").exists())
            image = self.home / image_path
            self.assertNotEqual(image, other_image)
            self.assertTrue(image.is_file())
            self.assertEqual(store.undo(), step)
            self.assertFalse(image.exists())
            self.assertTrue(other_image.is_file())
            self.assertEqual(unrelated.read_bytes(), b"other file")

    def test_default_home_is_fixed_at_construction(self):
        with tempfile.TemporaryDirectory() as other:
            original_cwd = Path.cwd()
            try:
                os.chdir(other)
                with patch("calibration.records.PROJECT_HOME", self.home):
                    store = MarkRecordStore(Path("captures-default"))
                    os.chdir(self.home.parent)
                    step = store.save_template(TEMPLATE, self.roi)
                data = yaml.safe_load(store.records_file.read_text(encoding="utf-8"))
                image = self.home / data["steps"][step]["image"]
                self.assertEqual(store.home, self.home)
                self.assertEqual(store.output_dir, self.home / "captures-default")
                self.assertEqual(image.parent, self.home / "captures" / "templates")
                self.assertTrue(image.is_file())
                self.assertEqual(store.undo(), step)
                self.assertFalse(image.exists())
                explicit = MarkRecordStore(Path("explicit"), home=self.home)
                self.assertEqual(explicit.output_dir, self.home / "explicit")
                self.assertEqual(explicit.templates_dir, self.home / "captures" / "templates")
            finally:
                os.chdir(original_cwd)

    def test_default_home_is_project_root_without_writing_there(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.home)
            with patch("calibration.records.resolve_home_path", side_effect=ValueError("stop")) as resolve:
                with self.assertRaisesRegex(ValueError, "stop"):
                    MarkRecordStore(Path("captures"))
            resolve.assert_called_once_with(Path("captures"), PROJECT_HOME)
        finally:
            os.chdir(original_cwd)

    def test_root_metadata_is_preserved_on_save_and_undo(self):
        metadata = {"metadata": {"device": "phone", "tags": ["one", "two"]},
                     "version": 2, "meta": {"source": "marker", **SIZES}}
        self.store.records_file.write_text(
            yaml.safe_dump(metadata), encoding="utf-8"
        )
        point = self.store.save_point(POINT)
        template = self.store.save_template(TEMPLATE, self.roi)

        def assert_metadata():
            data = yaml.safe_load(self.store.records_file.read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"], metadata["metadata"])
            self.assertEqual(data["version"], metadata["version"])
            self.assertEqual(data["meta"], {**metadata["meta"], **SIZES})
            return data

        self.assertEqual(list(assert_metadata()["steps"]), [point, template])
        self.assertEqual(self.store.undo(), template)
        self.assertEqual(list(assert_metadata()["steps"]), [point])
        self.assertEqual(self.store.undo(), point)
        self.assertEqual(assert_metadata(), {**metadata, "meta": {**metadata["meta"], **SIZES}, "steps": {}})

    def test_undo_write_failure_preserves_history_and_template(self):
        step = self.store.save_template(TEMPLATE, self.roi)
        original = self.store.records_file.read_bytes()
        history = self.store._undo_history.copy()
        image = history[-1][1]
        original_image = image.read_bytes()

        with patch(
            "calibration.records.os.replace",
            side_effect=PermissionError("YAML replacement denied"),
        ):
            with self.assertRaisesRegex(PermissionError, "YAML replacement denied"):
                self.store.undo()

        self.assertEqual(self.store.records_file.read_bytes(), original)
        self.assertEqual(self.store._undo_history, history)
        self.assertEqual(image.read_bytes(), original_image)
        self.assertEqual(list(self.store.output_dir.glob("*.yaml.tmp")), [])
        self.assertEqual(self.store.undo(), step)
        self.assertEqual(self.store._undo_history, [])
        self.assertFalse(image.exists())
        self.assertEqual(yaml.safe_load(self.store.records_file.read_text(encoding="utf-8")),
                         {"meta": SIZES, "steps": {}})


if __name__ == "__main__":
    unittest.main()
