from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dslighting.benchmark.core.mle_style_registry import MLEStyleRegistry
from dslighting.benchmark.core.source_catalog import BenchmarkSourceDescriptor
from dslighting.core.tasks.errors import TaskLayoutResolutionError
from dslighting.core.tasks.resolver import TaskResolver

TASK_ID = "identity-task"
IDENTITY_FIELD = "task_metadata.source_family"


@pytest.mark.parametrize(
    "leaf",
    [
        None,
        "",
        "   ",
        [],
        {},
        ["family-a"],
        {"name": "family-a"},
        7,
        True,
    ],
)
def test_declared_dataset_identity_requires_non_empty_string(leaf: Any) -> None:
    with pytest.raises(
        TaskLayoutResolutionError,
        match="must be a non-empty single-line string",
    ):
        TaskResolver.resolve_dataset_id(
            task_id=TASK_ID,
            config={"task_metadata": {"source_family": leaf}},
            context_dataset_id_field=IDENTITY_FIELD,
            context_dataset_id_prefix="dataset-",
        )


@pytest.mark.parametrize(
    "leaf",
    ["family-a\nfamily-b", "family-a\rfamily-b", "family-a\u2028family-b"],
)
def test_declared_dataset_identity_requires_one_line(leaf: str) -> None:
    with pytest.raises(
        TaskLayoutResolutionError,
        match="must be a non-empty single-line string",
    ):
        TaskResolver.resolve_dataset_id(
            task_id=TASK_ID,
            config={"task_metadata": {"source_family": leaf}},
            context_dataset_id_field=IDENTITY_FIELD,
            context_dataset_id_prefix="dataset-",
        )


@pytest.mark.parametrize(
    ("leaf", "prefix"),
    [
        ("../family", "dataset-"),
        ("family\\other", "dataset-"),
        ("family\x00other", "dataset-"),
        ("family", "../dataset-"),
        ("family", "dataset-\nother-"),
    ],
)
def test_prefixed_dataset_identity_must_be_a_safe_basename(
    leaf: str,
    prefix: str,
) -> None:
    with pytest.raises(TaskLayoutResolutionError, match="invalid context dataset identity"):
        TaskResolver.resolve_dataset_id(
            task_id=TASK_ID,
            config={"task_metadata": {"source_family": leaf}},
            context_dataset_id_field=IDENTITY_FIELD,
            context_dataset_id_prefix=prefix,
        )


def _write_task_package(task_dir: Path, *, family: str, prepared: bool) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "description.md").write_text(
        "# Identity task\n\n## Dataset Description\n\nSynthetic data.\n",
        encoding="utf-8",
    )
    (task_dir / "prepare.py").write_text(
        "def prepare(raw, public, private):\n    return public\n",
        encoding="utf-8",
    )
    (task_dir / "grade.py").write_text(
        "def grade(submission, answers):\n    return 1.0\n",
        encoding="utf-8",
    )
    (task_dir / "checksums.yaml").write_text("{}\n", encoding="utf-8")
    (task_dir / "leaderboard.csv").write_text("score\n1.0\n", encoding="utf-8")
    (task_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "id": TASK_ID,
                "name": "Identity task",
                "competition_type": "tabular",
                "description": "description.md",
                "task_metadata": {"source_family": family},
                "dataset": {
                    "answers": f"{TASK_ID}/prepared/private/answers.csv",
                    "sample_submission": (f"{TASK_ID}/prepared/public/sample_submission.csv"),
                },
                "grader": {
                    "name": "StandardGrader",
                    "grade_fn": "file:grade.py:grade",
                },
                "preparer": "file:prepare.py:prepare",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    if prepared:
        public_dir = task_dir / "prepared" / "public"
        private_dir = task_dir / "prepared" / "private"
        raw_dir = task_dir / "raw"
        public_dir.mkdir(parents=True)
        private_dir.mkdir(parents=True)
        raw_dir.mkdir()
        (public_dir / "features.csv").write_text("value\n1\n", encoding="utf-8")
        (public_dir / "sample_submission.csv").write_text(
            "prediction\n0\n",
            encoding="utf-8",
        )
        (private_dir / "answers.csv").write_text(
            "prediction\n1\n",
            encoding="utf-8",
        )


def test_registry_and_resolver_use_the_effective_data_root_config(tmp_path: Path) -> None:
    source_id = "identity-source"
    vendor_root = tmp_path / "benchmark" / "vendor" / source_id
    registry_root = vendor_root / "competitions"
    data_root = tmp_path / "data"
    vendor_task_dir = registry_root / TASK_ID
    data_task_dir = data_root / TASK_ID

    _write_task_package(vendor_task_dir, family="family-a", prepared=False)
    _write_task_package(data_task_dir, family="family-b", prepared=True)
    manifest_path = vendor_root / "benchmark.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source_id": source_id,
                "contract_id": "mle_task_contract/v1",
                "engine_id": "mle",
                "registry_root": "competitions",
                "context_dataset": {
                    "id_field": IDENTITY_FIELD,
                    "id_prefix": "dataset-",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    descriptor = BenchmarkSourceDescriptor(
        source_id=source_id,
        contract_id="mle_task_contract/v1",
        engine_id="mle",
        vendor_root=vendor_root,
        registry_root=registry_root,
        manifest_path=manifest_path,
        context_dataset_id_field=IDENTITY_FIELD,
        context_dataset_id_prefix="dataset-",
    )

    registry = MLEStyleRegistry(descriptor=descriptor, data_dir=data_root)
    effective = registry.resolve_task_config(TASK_ID)

    assert effective.task_dir == data_task_dir.resolve()
    assert effective.config["task_metadata"]["source_family"] == "family-b"

    layout = TaskResolver().resolve(
        task_id=TASK_ID,
        data=data_root,
        registry_dir=registry_root,
    )

    assert layout.dataset_id == "dataset-family-b"
    assert layout.task_root == data_task_dir.resolve()
