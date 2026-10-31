from __future__ import annotations

import os
import tempfile
from time import strftime
from pathlib import Path
from typing import Optional

import cv2
import yaml

from log import get_logger
from home_paths import PROJECT_HOME, resolve_home_path

logger = get_logger("marker")


class MarkRecordStore:
    def __init__(self, output_dir: Path, home: Optional[Path] = None):
        self.home = Path(home).resolve() if home is not None else PROJECT_HOME
        self.output_dir = resolve_home_path(output_dir, self.home)
        self.templates_dir = self.home / "captures" / "templates"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(
            dir=self.output_dir,
            prefix=f"mark-records_{strftime('%Y%m%d_%H%M%S')}_",
            suffix=".yaml",
        )
        self.records_file = Path(path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fd = -1
                fp.write("{}\n")
        except BaseException:
            if fd != -1:
                os.close(fd)
            self.records_file.unlink(missing_ok=True)
            raise
        logger.info("Marker records: %s", self.records_file)
        self._undo_history: list[tuple[str, Optional[Path]]] = []

    def save_point(self, record: dict) -> str:
        step_name = self._append_step("point", record)
        self._undo_history.append((step_name, None))
        return step_name

    def save_template(self, record: dict, roi) -> str:
        fd, image_path = tempfile.mkstemp(
            dir=self.templates_dir,
            prefix=f"template_{strftime('%Y%m%d_%H%M%S')}_",
            suffix=".png",
        )
        path = Path(image_path)
        try:
            os.close(fd)
            if not cv2.imwrite(str(path), roi):
                raise OSError(f"Failed to save ROI image: {path}")
            step_name = self._append_step(
                "template", {**record, "image": (Path("captures/templates") / path.name).as_posix()}
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._undo_history.append((step_name, path))
        return step_name

    def undo(self) -> Optional[str]:
        if not self._undo_history:
            logger.info("No saved records to undo in this session.")
            return None
        step_name, image_path = self._undo_history[-1]
        data = self._load()
        steps = data.get("steps")
        if not isinstance(steps, dict) or step_name not in steps:
            raise ValueError(f"Record '{step_name}' not found in {self.records_file}")
        del steps[step_name]
        self._write(data)
        self._undo_history.pop()
        if image_path is not None:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Record was removed, but template image could not be deleted: %s", image_path)
                return step_name
        return step_name

    def _append_step(self, action: str, record: dict) -> str:
        sizes = {}
        for key in ("frame_size", "device_size"):
            size = record.get(key)
            if not isinstance(size, dict) or any(
                type(size.get(dimension)) is not int or size[dimension] <= 0
                for dimension in ("width", "height")
            ):
                raise ValueError(f"'{key}' must contain positive integer width and height")
            sizes[key] = {dimension: size[dimension] for dimension in ("width", "height")}
        data = self._load()
        steps = data.get("steps", {})
        if not isinstance(steps, dict):
            raise ValueError(f"'steps' in {self.records_file} must be a mapping")
        if "meta" not in data:
            if steps:
                raise ValueError(f"'meta' is required for existing steps in {self.records_file}")
            data["meta"] = sizes
        meta = data["meta"]
        if not isinstance(meta, dict):
            raise ValueError(f"'meta' in {self.records_file} must be a mapping")
        for key, size in sizes.items():
            if meta.get(key) != size:
                raise ValueError(f"'{key}' does not match meta in {self.records_file}")
        index = 1
        while f"step_{index:03d}" in steps:
            index += 1
        step_name = f"step_{index:03d}"
        steps[step_name] = {"action": action, **{key: value for key, value in record.items() if key not in sizes}}
        data["steps"] = steps
        self._write(data)
        return step_name

    def _load(self) -> dict:
        if not self.records_file.exists():
            return {}
        with self.records_file.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {self.records_file}")
        return data

    def _write(self, data: dict) -> None:
        fd, temp_path = tempfile.mkstemp(dir=self.output_dir, suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fd = -1
                yaml.safe_dump(data, fp, allow_unicode=True, sort_keys=False)
            os.replace(temp_path, self.records_file)
        finally:
            if fd != -1:
                os.close(fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
