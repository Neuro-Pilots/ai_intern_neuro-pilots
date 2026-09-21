"""Explicit DVC operations for datasets selected by a NeuroPilots user."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class DVCManager:
    """Initialize and track data only after a caller explicitly requests it."""

    def track(self, dataset_path: str | Path, repository_path: str | Path) -> Path:
        """Run ``dvc init`` when needed and ``dvc add`` for one exact dataset path."""
        dataset = Path(dataset_path).expanduser().resolve()
        repository = Path(repository_path).expanduser().resolve()
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset}")
        if not (repository / ".git").exists():
            raise ValueError("DVC versioning requires a Git repository path.")
        local_dvc = Path(sys.executable).parent / "dvc"
        dvc = str(local_dvc) if local_dvc.exists() else shutil.which("dvc")
        if not dvc:
            raise RuntimeError("DVC is not installed or not available on PATH.")
        if not (repository / ".dvc").exists():
            self._run([dvc, "init"], repository)
        self._run([dvc, "add", str(dataset)], repository)
        return dataset.with_name(f"{dataset.name}.dvc")

    @staticmethod
    def _run(command: list[str], cwd: Path) -> None:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip() or "Unknown DVC error"
            raise RuntimeError(message)
