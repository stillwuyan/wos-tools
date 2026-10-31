from __future__ import annotations

import sys
from pathlib import Path, PureWindowsPath


PROJECT_HOME = Path(__file__).resolve().parent


def resolve_home_path(value: str | Path, home: Path) -> Path:
    if sys.platform == "win32":
        windows_path = PureWindowsPath(value)
        if windows_path.drive and not windows_path.root:
            raise ValueError(f"Windows drive-relative path cannot be resolved against home: {value}")
        if windows_path.root and not windows_path.drive:
            raise ValueError(f"Windows root-relative path cannot be resolved against home: {value}")
    return (home / Path(value)).resolve()
