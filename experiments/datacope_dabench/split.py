"""Frozen family-level explore/test split for the DABench experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAMILY_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "releases"
    / "dabench_perception_clean_v1"
    / "dabench_perception_dataset_manifest.json"
)
SOURCE_MANIFEST_SHA256 = "c22f572c670f7e9914f5d01ae7220c402e074d948f99f6c1d9bae837291ec708"
SPLIT_ID = "dabench-family-1-to-3-seed-42-v1"

# Selected once with seed 42 under two constraints: 13 of 51 families and
# exactly 64 of 257 tasks. Keeping the family IDs here makes the split stable
# and prevents the same raw dataset from appearing in both phases.
EXPLORE_FAMILY_IDS = frozenset(
    {
        "dabench-family-2890f10a92b706c260de39281a814442249716ea757029ae5425f7e12ff1e11b",
        "dabench-family-411cf03455d6026823fbd3ab65e2839075a22f9a5c088b85aef0d272d79cca00",
        "dabench-family-425acc100e1a260e1206d7f12aa03e5186a3895a4c4fa2da5b3116f1350d2f76",
        "dabench-family-55770473647050ba8f11d46108aac81f38df600e8261b105321fffd5c5a6ecec",
        "dabench-family-68a9ce2a7b5c249f01c716d92eb5f7429be1c6185f0160a6b73d6df89e425419",
        "dabench-family-6d0f08343d42a5ad45d1eb2275a99416aa8a0b26be03fe5e74538cff31352292",
        "dabench-family-6fda3c4b0c0cff9bd5f7a46434509179069ed620c8fc7090d872a08d2c1a729a",
        "dabench-family-753bcb0a243256986b0b663666d0bbf5dd273f0de753ad0705be1c9cbacf8f83",
        "dabench-family-83efce074700d226453f08c9b8faddc4703c70ae0e1888e8e8422994416d1ce0",
        "dabench-family-86e7e9f8c381c36441f72800f20c011497351541e1e9a9d3f532844922c447ae",
        "dabench-family-9e6f0a924da3b91a0c7ba24260f773f8648deb2475ca4bbd5d16f1366a0b45a7",
        "dabench-family-b2d37e2153fa661b710170db1c9b2d62cef2baf2910535bd5600c32892f48a45",
        "dabench-family-c9e411ff75df8d966f6e5de9270a4a7276289c478a6b7aa216ce667eac66f540",
    }
)


@dataclass(frozen=True)
class DABenchSplit:
    """Resolved task and family IDs for the frozen split."""

    split_id: str
    source_manifest_sha256: str
    explore_family_ids: tuple[str, ...]
    test_family_ids: tuple[str, ...]
    explore_task_ids: tuple[str, ...]
    test_task_ids: tuple[str, ...]

    def task_ids(self, phase: str) -> tuple[str, ...]:
        if phase == "explore":
            return self.explore_task_ids
        if phase == "test":
            return self.test_task_ids
        raise ValueError(f"Unsupported DABench phase: {phase}")


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read DABench family manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"DABench family manifest must be a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def load_dabench_split(manifest_path: Path = DEFAULT_FAMILY_MANIFEST) -> DABenchSplit:
    """Validate the frozen source manifest and resolve the 64/193 task split."""

    path = Path(manifest_path).expanduser().resolve()
    payload, digest = _load_payload(path)
    if digest != SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "DABench family manifest changed; refusing to silently change the frozen split "
            f"(expected {SOURCE_MANIFEST_SHA256}, got {digest})"
        )
    if payload.get("schema_version") != "dabench_perception_dataset_manifest_v1":
        raise ValueError("Unsupported DABench family manifest schema")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 257:
        raise ValueError("Frozen DABench split requires exactly 257 manifest tasks")

    task_ids: set[str] = set()
    family_to_raw: dict[str, str] = {}
    task_families: list[tuple[str, str]] = []
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("DABench family manifest task entries must be objects")
        task_id = str(item.get("task_id") or "")
        family_id = str(item.get("family_id") or "")
        raw_sha256 = str(item.get("raw_sha256") or "")
        if not task_id.startswith("dabench-") or not family_id.startswith("dabench-family-"):
            raise ValueError("DABench family manifest contains an invalid task or family ID")
        if task_id in task_ids:
            raise ValueError(f"Duplicate DABench task ID in family manifest: {task_id}")
        task_ids.add(task_id)
        previous_raw = family_to_raw.setdefault(family_id, raw_sha256)
        if previous_raw != raw_sha256:
            raise ValueError(f"DABench family {family_id} maps to multiple raw datasets")
        task_families.append((task_id, family_id))

    all_families = set(family_to_raw)
    if len(all_families) != 51 or not EXPLORE_FAMILY_IDS < all_families:
        raise ValueError("Frozen DABench split requires its expected 51 data families")

    test_families = all_families - EXPLORE_FAMILY_IDS
    explore_tasks = sorted(
        task_id for task_id, family_id in task_families if family_id in EXPLORE_FAMILY_IDS
    )
    test_tasks = sorted(
        task_id for task_id, family_id in task_families if family_id in test_families
    )
    if len(explore_tasks) != 64 or len(test_tasks) != 193:
        raise ValueError("Frozen DABench split must resolve to 64 explore and 193 test tasks")

    explore_raw = {family_to_raw[family_id] for family_id in EXPLORE_FAMILY_IDS}
    test_raw = {family_to_raw[family_id] for family_id in test_families}
    if explore_raw & test_raw:
        raise ValueError("A raw DABench dataset appears in both explore and test")

    return DABenchSplit(
        split_id=SPLIT_ID,
        source_manifest_sha256=digest,
        explore_family_ids=tuple(sorted(EXPLORE_FAMILY_IDS)),
        test_family_ids=tuple(sorted(test_families)),
        explore_task_ids=tuple(explore_tasks),
        test_task_ids=tuple(test_tasks),
    )


__all__ = [
    "DABenchSplit",
    "DEFAULT_FAMILY_MANIFEST",
    "EXPLORE_FAMILY_IDS",
    "SOURCE_MANIFEST_SHA256",
    "SPLIT_ID",
    "load_dabench_split",
]
