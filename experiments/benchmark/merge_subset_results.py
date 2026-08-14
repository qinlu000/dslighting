"""Replace a benchmark result subset while preserving the base row order."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "competition_id" not in reader.fieldnames:
            raise ValueError(f"Missing competition_id column: {path}")
        return list(reader.fieldnames), list(reader)


def _index_rows(rows: list[dict[str, str]], *, source: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = row["competition_id"]
        if not task_id or task_id in indexed:
            raise ValueError(f"Missing or duplicate competition_id in {source}: {task_id!r}")
        indexed[task_id] = row
    return indexed


def merge_results(
    *,
    base: Path,
    subset: Path,
    output: Path,
    expected_task_ids: set[str],
) -> None:
    base_fields, base_rows = _read_rows(base)
    subset_fields, subset_rows = _read_rows(subset)
    if subset_fields != base_fields:
        raise ValueError("Base and subset result schemas differ")

    base_by_id = _index_rows(base_rows, source=base)
    subset_by_id = _index_rows(subset_rows, source=subset)
    if set(subset_by_id) != expected_task_ids:
        raise ValueError(
            "Subset task IDs differ from expectation: "
            f"missing={sorted(expected_task_ids - set(subset_by_id))}, "
            f"extra={sorted(set(subset_by_id) - expected_task_ids)}"
        )
    missing_from_base = expected_task_ids - set(base_by_id)
    if missing_from_base:
        raise ValueError(f"Subset tasks missing from base results: {sorted(missing_from_base)}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    merged_rows = [subset_by_id.get(row["competition_id"], row) for row in base_rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=base_fields)
            writer.writeheader()
            writer.writerows(merged_rows)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"merged_rows={len(merged_rows)} replaced_rows={len(subset_by_id)} output={output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    args = parser.parse_args(argv)
    merge_results(
        base=args.base.resolve(),
        subset=args.subset.resolve(),
        output=args.output.resolve(),
        expected_task_ids=set(args.task_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
