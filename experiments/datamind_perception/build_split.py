#!/usr/bin/env python3
"""Build the frozen Python-only DataMind perception split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

DATA_ROOT = Path("/data/caoqinlu/datasets/DataMind-Data/rl")
OUTPUT_DIR = Path("/data/caoqinlu/projects/dslighting/artifacts/datamind_perception")
SPLIT_IDS = Path(__file__).with_name("split_ids.json")
TRAIN_SOURCE = "darl/python"
TEST_SOURCES = {"TableBench", "DABench"}
TRAIN_COLUMNS = [
    "data_source",
    "db_id",
    "task_id",
    "trajectory",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
]
TEST_COLUMNS = [column for column in TRAIN_COLUMNS if column != "trajectory"]
SOURCE_RE = re.compile(
    r"data source path is\s*(['\"])(.*?)\1\.?\s*\*\*", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split-ids", type=Path, default=SPLIT_IDS)
    return parser.parse_args()


def task_id(extra_info: dict[str, Any]) -> str:
    value = extra_info.get("index")
    if not value:
        raise ValueError("missing extra_info.index")
    return str(value)


def source_file(prompt: Any) -> str:
    system = next(
        (message.get("content", "") for message in prompt if message.get("role") == "system"),
        "",
    )
    match = SOURCE_RE.search(system)
    if not match:
        raise ValueError("cannot extract source path from prompt")
    return Path(match.group(2)).name


def reference(reward_model: dict[str, Any]) -> str:
    return str(reward_model.get("ground_truth", {}).get("ground_truth", "")).strip()


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_id"] = frame["extra_info"].map(task_id)
    frame["_source"] = frame["prompt"].map(source_file)
    return frame


def select(frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    ids = [identifier for values in groups.values() for identifier in values]
    indexed = frame.set_index("_id", drop=False)
    missing = [identifier for identifier in ids if identifier not in indexed.index]
    if missing:
        raise ValueError(f"missing task IDs: {missing}")
    selected = indexed.loc[ids].copy()
    if len(selected) != len(ids):
        raise ValueError("duplicate task IDs in source data")
    categories = {
        identifier: category
        for category, values in groups.items()
        for identifier in values
    }
    selected["_category"] = selected["_id"].map(categories)
    return selected


def benchmark_rows(rows: pd.DataFrame, origin: str) -> pd.DataFrame:
    output = rows[TEST_COLUMNS].copy()
    output["extra_info"] = output["extra_info"].map(
        lambda value: {key: item for key, item in value.items() if key != "trajectory"}
    )
    output["origin_split"] = origin
    output["perception_category"] = rows["_category"].to_numpy()
    output["source_file"] = rows["_source"].to_numpy()
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_content_leak(
    benchmark_sources: set[str], train_sources: set[str], source_dir: Path
) -> None:
    fingerprints: dict[tuple[int, str], str] = {}
    sizes: set[int] = set()
    for name in benchmark_sources:
        path = source_dir / name
        if not path.is_file():
            raise AssertionError(f"missing benchmark source: {name}")
        fingerprint = (path.stat().st_size, sha256(path))
        if fingerprint in fingerprints:
            raise AssertionError(f"duplicate benchmark source: {name}")
        fingerprints[fingerprint] = name
        sizes.add(fingerprint[0])

    for name in train_sources:
        path = source_dir / name
        if not path.is_file():
            raise AssertionError(f"missing train source: {name}")
        size = path.stat().st_size
        if size in sizes and (size, sha256(path)) in fingerprints:
            raise AssertionError(f"content leak: {name}")


def write_manifests(
    official: pd.DataFrame,
    moved: pd.DataFrame,
    excluded: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    records = []
    for order, (_, row) in enumerate(pd.concat([official, moved]).iterrows(), 1):
        records.append(
            {
                "benchmark_order": order,
                "task_identifier": row["_id"],
                "origin_split": "official_test" if order <= len(official) else "official_train",
                "perception_category": row["_category"],
                "data_source": row["data_source"],
                "source_file": row["_source"],
                "question": row["extra_info"].get("question", ""),
                "reward_style": row["reward_model"].get("style", ""),
                "ground_truth": reference(row["reward_model"]),
            }
        )
    manifest_path = output_dir / "benchmark_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    moved_ids = set(moved["_id"])
    exclusion = excluded[["_id", "data_source", "_source"]].rename(
        columns={"_id": "task_identifier", "_source": "source_file"}
    )
    exclusion["selected_for_benchmark"] = exclusion["task_identifier"].isin(moved_ids)
    exclusion["exclusion_reason"] = exclusion["selected_for_benchmark"].map(
        {
            True: "moved_to_perception_benchmark",
            False: "shares_source_with_moved_benchmark_task",
        }
    )
    exclusion_path = output_dir / "excluded_same_source_manifest.csv"
    exclusion.to_csv(exclusion_path, index=False)
    return [manifest_path, exclusion_path]


def build(data_root: Path, output_dir: Path, split_ids_path: Path) -> dict[str, Any]:
    split_ids = json.loads(split_ids_path.read_text(encoding="utf-8"))
    train = annotate(pd.read_parquet(data_root / "train.parquet"))
    test = annotate(pd.read_parquet(data_root / "test.parquet"))
    official = select(test, split_ids["official_test"])
    moved = select(train, split_ids["official_train"])

    moved_sources = set(moved["_source"])
    if len(moved) != 56 or len(moved_sources) != 56:
        raise AssertionError("expected 56 moved tasks on 56 sources")

    python_train = train[train["data_source"] == TRAIN_SOURCE]
    excluded = python_train[python_train["_source"].isin(moved_sources)]
    training = python_train[~python_train["_source"].isin(moved_sources)][TRAIN_COLUMNS]
    benchmark = pd.concat(
        [benchmark_rows(official, "official_test"), benchmark_rows(moved, "official_train")],
        ignore_index=True,
    )
    remainder = test[
        test["data_source"].isin(TEST_SOURCES) & ~test["_id"].isin(set(official["_id"]))
    ][TEST_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "benchmark_100.parquet",
        output_dir / "train.parquet",
        output_dir / "official_python_test_remainder.parquet",
    ]
    benchmark.to_parquet(paths[0], index=False, compression="zstd")
    training.to_parquet(paths[1], index=False, compression="zstd")
    remainder.to_parquet(paths[2], index=False, compression="zstd")
    paths.extend(write_manifests(official, moved, excluded, output_dir))

    train_sources = set(training["prompt"].map(source_file))
    benchmark_sources = set(benchmark["source_file"])
    if (len(benchmark), len(training), len(remainder)) != (100, 7154, 276):
        raise AssertionError("unexpected split size")
    if set(training["data_source"]) != {TRAIN_SOURCE}:
        raise AssertionError("training split is not Python-only")
    if not set(benchmark["data_source"]) <= TEST_SOURCES | {TRAIN_SOURCE}:
        raise AssertionError("benchmark contains SQL/BIRD rows")
    if benchmark_sources & train_sources:
        raise AssertionError("source filename leak")
    if "trajectory" in benchmark or any("trajectory" in value for value in benchmark["extra_info"]):
        raise AssertionError("trajectory leaked into benchmark")
    assert_no_content_leak(benchmark_sources, train_sources, data_root / "train_files")

    summary = {
        "benchmark": {
            "rows": len(benchmark),
            "official_test": len(official),
            "official_train": len(moved),
            "categories": benchmark["perception_category"].value_counts().sort_index().to_dict(),
        },
        "train": {
            "rows": len(training),
            "source_files": len(train_sources),
            "excluded_for_source_isolation": len(excluded),
            "sql_rows_excluded": int((train["data_source"] != TRAIN_SOURCE).sum()),
        },
        "python_test_remainder": len(remainder),
        "validation": {
            "python_only": True,
            "source_overlap": [],
            "exact_content_overlap": [],
            "trajectory_in_benchmark": False,
        },
        "sha256": {path.name: sha256(path) for path in paths},
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build(args.data_root, args.output_dir, args.split_ids)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
