"""Dataset identity and DVC provenance for reproducible NeuroPilots runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetVersion:
    """Immutable dataset provenance recorded with every experiment."""

    dataset_name: str
    path: str
    fingerprint: str
    dvc_revision: str | None
    file_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.path,
            "dataset_fingerprint": self.fingerprint,
            "dvc_revision": self.dvc_revision or "unversioned",
            "file_count": self.file_count,
        }


class DatasetVersioner:
    """Calculates a content fingerprint and reads DVC status without mutation."""

    def identify(self, dataset_path: str | Path, dataset_name: str) -> DatasetVersion:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        digest = hashlib.sha256()
        for item in files:
            relative = item.name if path.is_file() else str(item.relative_to(path))
            stat = item.stat()
            digest.update(f"{relative}:{stat.st_size}:".encode())
            with item.open("rb") as handle:
                digest.update(handle.read(65536))

        return DatasetVersion(
            dataset_name=dataset_name,
            path=str(path),
            fingerprint=digest.hexdigest(),
            dvc_revision=self._dvc_revision(path),
            file_count=len(files),
        )

    @staticmethod
    def _dvc_revision(path: Path) -> str | None:
        """Return the checked-out DVC data version when DVC is configured."""
        working_directory = path if path.is_dir() else path.parent
        try:
            dvc = DatasetVersioner._dvc_executable()
            if dvc is None:
                return None
            process = subprocess.run(
                [dvc, "status", "--json"],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if process.returncode != 0:
                return None
            payload = json.loads(process.stdout)
            return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
            return None

    @staticmethod
    def _dvc_executable() -> str | None:
        # Do not resolve the interpreter symlink: a virtualenv's scripts live
        # beside the symlink, not beside the framework interpreter it targets.
        local_dvc = Path(sys.executable).parent / "dvc"
        if local_dvc.exists():
            return str(local_dvc)
        return shutil.which("dvc")
