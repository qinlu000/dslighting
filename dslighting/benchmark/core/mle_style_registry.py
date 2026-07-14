from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dslighting.benchmark.core.mle_task_contract import (
    MLEStyleCompetition,
    MLETaskContractLoader,
)
from dslighting.benchmark.core.source_catalog import BenchmarkSourceDescriptor

DEFAULT_DATA_DIR = (Path.home() / ".cache" / "mle-bench" / "data").resolve()


@dataclass(frozen=True)
class ResolvedTaskConfig:
    """The task directory and config selected by the registry's override rules."""

    task_dir: Path
    config: dict[str, Any]


class MLEStyleRegistry:
    """Registry adapter shared by all MLE-style benchmark sources."""

    def __init__(
        self,
        descriptor: BenchmarkSourceDescriptor,
        data_dir: Path = DEFAULT_DATA_DIR,
        mode: str = "test",
    ) -> None:
        self.descriptor = descriptor
        self._data_dir = Path(data_dir).resolve()
        self.mode = mode
        self.loader = MLETaskContractLoader(descriptor)

    def _resolve_task_dir(self, competition_id: str) -> Path:
        user_task_dir = self._data_dir / competition_id
        if (user_task_dir / "config.yaml").exists():
            return user_task_dir
        return self.get_competitions_dir() / competition_id

    def resolve_task_config(self, competition_id: str) -> ResolvedTaskConfig:
        """Return the effective task config used to build the competition.

        A task package colocated under the configured data root overrides the
        vendor package.  Control-plane consumers must use this boundary rather
        than reading the vendor registry directly, or they can disagree with
        the competition that the registry actually executes.
        """

        task_dir = self._resolve_task_dir(competition_id).resolve()
        return ResolvedTaskConfig(
            task_dir=task_dir,
            config=self.loader.load_task_config(task_dir),
        )

    def set_mode(self, mode: str = "test") -> "MLEStyleRegistry":
        assert mode in ["test", "validation", "prepare"], "Mode must be in ['test', 'validation', 'prepare']."
        self.mode = mode
        return self

    def get_competition(self, competition_id: str) -> MLEStyleCompetition:
        resolved = self.resolve_task_config(competition_id)
        payload = self.loader.build_competition_payload(
            resolved.task_dir,
            self.get_data_dir(),
            resolved.config,
            self.mode,
        )
        return MLEStyleCompetition.from_dict(payload)

    def get_competitions_dir(self) -> Path:
        return self.descriptor.registry_root

    def get_splits_dir(self) -> Path:
        return self.descriptor.vendor_root.parent.parent / "experiments" / "splits"

    def get_lite_competition_ids(self) -> list[str]:
        lite_competitions_file = self.get_splits_dir() / "low.txt"
        if not lite_competitions_file.exists():
            return []
        return lite_competitions_file.read_text(encoding="utf-8").splitlines()

    def get_data_dir(self) -> Path:
        return self._data_dir

    def set_data_dir(self, new_data_dir: Path) -> "MLEStyleRegistry":
        return MLEStyleRegistry(
            descriptor=self.descriptor,
            data_dir=Path(new_data_dir),
            mode=self.mode,
        )

    def list_competition_ids(self) -> list[str]:
        competition_ids: set[str] = set()
        for cfg in self.get_competitions_dir().rglob("config.yaml"):
            competition_ids.add(cfg.parent.stem)
        return sorted(competition_ids)
