"""Create and load a deterministic family-stratified MoSciBench split."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

DATASETS = (
    "health_spa",
    "massspecgym",
    "pop_genetics",
    "nurse_stress",
    "cyclone",
    "terra",
)
DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "releases"
    / "moscibench_full"
    / "moscibench"
    / "competitions"
)
DEFAULT_SPLIT_MANIFEST = Path(__file__).with_name("split.json")
DEFAULT_EXPLORE_FRACTION = 0.25
DEFAULT_SPLIT_SALT = "datacope-moscibench-v1"
_TASK_PATTERN = re.compile(r"mosci-(.+)-(\d+)")


def task_family(task_id: str) -> str:
    match = _TASK_PATTERN.fullmatch(task_id)
    if match is None or match.group(1) not in DATASETS:
        raise ValueError(f"Invalid MoSciBench task ID: {task_id}")
    return match.group(1)


def discover_tasks(data_root: Path) -> dict[str, tuple[str, ...]]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"MoSciBench data root does not exist: {root}")
    tasks: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    for path in root.glob("mosci-*"):
        if not path.is_dir():
            continue
        family = task_family(path.name)
        if not (path / "description.md").is_file():
            raise ValueError(f"Missing description.md for {path.name}")
        if not (path / "prepared" / "public").is_dir():
            raise ValueError(f"Missing prepared/public for {path.name}")
        tasks[family].append(path.name)
    if any(not task_ids for task_ids in tasks.values()):
        raise ValueError("MoSciBench data root must contain all six task families")
    return {
        family: tuple(sorted(task_ids, key=lambda value: int(value.rsplit("-", 1)[1])))
        for family, task_ids in tasks.items()
    }


def source_sha256(data_root: Path, tasks: dict[str, tuple[str, ...]]) -> str:
    root = Path(data_root).expanduser().resolve()
    digest = hashlib.sha256()
    for family in DATASETS:
        for task_id in tasks[family]:
            digest.update(task_id.encode())
            digest.update(b"\0")
            digest.update((root / task_id / "description.md").read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class MoSciSplit:
    manifest_path: Path
    split_id: str
    source_sha256: str
    explore: dict[str, tuple[str, ...]]
    test: dict[str, tuple[str, ...]]

    def task_ids(self, phase: str) -> tuple[str, ...]:
        table = self.explore if phase == "explore" else self.test if phase == "test" else None
        if table is None:
            raise ValueError(f"Unknown phase: {phase}")
        return tuple(task_id for family in DATASETS for task_id in table[family])

    def query_ids(self, phase: str) -> set[str]:
        return set(self.task_ids(phase))

    def verify_source(self, data_root: Path) -> None:
        tasks = discover_tasks(data_root)
        if source_sha256(data_root, tasks) != self.source_sha256:
            raise ValueError(f"MoSciBench task source changed: {Path(data_root).resolve()}")


def create_split(
    data_root: Path,
    output_path: Path,
    *,
    explore_fraction: float = DEFAULT_EXPLORE_FRACTION,
    salt: str = DEFAULT_SPLIT_SALT,
) -> Path:
    if not 0 < explore_fraction < 1:
        raise ValueError("explore_fraction must be between 0 and 1")
    root = Path(data_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Split manifest already exists: {output}")

    tasks = discover_tasks(root)
    partitions: dict[str, dict[str, list[str]]] = {}
    for family in DATASETS:
        task_ids = tasks[family]
        ordered = sorted(
            task_ids,
            key=lambda task_id: hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest(),
        )
        count = max(1, min(len(task_ids) - 1, int(len(task_ids) * explore_fraction + 0.5)))
        explore_set = set(ordered[:count])
        partitions[family] = {
            "explore": [task_id for task_id in task_ids if task_id in explore_set],
            "test": [task_id for task_id in task_ids if task_id not in explore_set],
        }

    payload = {
        "schema_version": 1,
        "benchmark": "MoSciBench",
        "split_id": salt,
        "explore_fraction": explore_fraction,
        "source_sha256": source_sha256(root, tasks),
        "partitions": partitions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_split(path: Path = DEFAULT_SPLIT_MANIFEST) -> MoSciSplit:
    manifest = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload["schema_version"] != 1 or payload["benchmark"] != "MoSciBench":
            raise ValueError("unsupported manifest")
        partitions = payload["partitions"]
        explore = {
            family: tuple(str(value) for value in partitions[family]["explore"])
            for family in DATASETS
        }
        test = {
            family: tuple(str(value) for value in partitions[family]["test"]) for family in DATASETS
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MoSciBench split manifest {manifest}: {exc}") from exc

    for family in DATASETS:
        if not explore[family] or not test[family] or set(explore[family]) & set(test[family]):
            raise ValueError(f"Invalid explore/test partition for {family}")
        if any(task_family(task_id) != family for task_id in (*explore[family], *test[family])):
            raise ValueError(f"Wrong task family in partition: {family}")
    return MoSciSplit(
        manifest_path=manifest,
        split_id=str(payload["split_id"]),
        source_sha256=str(payload["source_sha256"]),
        explore=explore,
        test=test,
    )
