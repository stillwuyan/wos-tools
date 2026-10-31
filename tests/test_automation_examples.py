import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

import automation.config as config
from automation import load_plan


class AutomationExamplesTests(unittest.TestCase):
    def test_real_daily_plan_loads_from_home(self):
        source = Path(__file__).resolve().parent.parent
        files = ("tasks/auto_polar_terror.yaml", "tasks/common.yaml", "tasks/polar_terror.yaml")
        documents = {}
        for name in files:
            with (source / name).open(encoding="utf-8") as fp:
                documents[name] = yaml.safe_load(fp)

        images = set()

        def collect_images(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ("image", "click_image"):
                        self.assertIsInstance(child, str)
                        images.add(child)
                    else:
                        collect_images(child)
            elif isinstance(value, list):
                for child in value:
                    collect_images(child)

        for document in documents.values():
            collect_images(document)
        self.assertTrue(images)

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            for name in files:
                destination = home / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination)

            image_paths = set()
            for name in sorted(images):
                relative = Path(name)
                self.assertFalse(relative.is_absolute(), name)
                path = (home / relative).resolve()
                self.assertTrue(path.is_relative_to(home), name)
                self.assertNotIn(path, image_paths, name)
                image_paths.add(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
                pixels = np.random.default_rng(seed).integers(0, 256, (16, 16), dtype=np.uint8)
                self.assertNotEqual(pixels.min(), pixels.max())
                encoded, data = cv2.imencode(path.suffix, pixels)
                self.assertTrue(encoded, name)
                path.write_bytes(data.tobytes())

            before = {path: path.read_bytes() for path in home.rglob("*") if path.is_file()}
            with patch.object(config, "resolve_home_path", wraps=config.resolve_home_path) as resolve, \
                    patch.object(config, "_read_yaml", wraps=config._read_yaml) as read_yaml:
                plan = load_plan("tasks/auto_polar_terror.yaml", home=home)

            self.assertTrue(resolve.call_args_list)
            for call in resolve.call_args_list:
                self.assertEqual(call.args[1], home)
            self.assertEqual({call.args[0] for call in resolve.call_args_list}, set(files) | images)
            self.assertEqual({call.args[0] for call in read_yaml.call_args_list},
                             {home / name for name in files})
            # The list and common are read once each; the looped task reads its file.
            expected_reads = 2 + 1
            self.assertEqual(len(read_yaml.call_args_list), expected_reads)
            self.assertEqual(plan.tasks, [])
            self.assertEqual(plan.loop["task"]["task"], "polar_terror")
            self.assertEqual(plan.loop["task"]["start_scenes"], ["wild"])
            self.assertEqual(plan.loop["on_failure"], "stop")
            self.assertEqual(set(plan.scenes), set(documents["tasks/common.yaml"]["scenes"]))
            templates = [record for record in plan.resources.values() if record["action"] == "template"]
            self.assertEqual({record["image"] for record in templates}, image_paths)
            self.assertEqual(len(templates), len(image_paths))
            self.assertEqual(len({record["grayscale"].tobytes() for record in templates}), len(image_paths))
            self.assertEqual({path: path.read_bytes() for path in home.rglob("*") if path.is_file()}, before)
            self.assertEqual(set(before), {home / name for name in files} | image_paths)
            self.assertFalse(any("marker" in path.name.lower() for path in before))


if __name__ == "__main__":
    unittest.main()
