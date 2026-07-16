"""Build and verify a clean DABench release for perception ablations.

The existing DABench release is an execution input and can be changed by an
agent that writes into its task directory.  Perception annotations must never
be generated from such a mutable copy.  This utility rebuilds every task from
its single raw CSV with the vendored ``prepare.py`` and publishes a clean,
integrity-attested release only after the complete release has passed checks.

The raw-file SHA-256 is the dataset identity.  Tasks with identical raw bytes
belong to one family, and their rebuilt ``train.csv`` files must also be byte
identical.  Existing prepared files are not read while constructing the clean
release.  Every regular source file in each vendored task contract is attested
as metadata (relative path, byte size, and SHA-256); runtime bytecode caches are
excluded.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "dabench_perception_dataset_manifest_v1"
VENDOR_CONTRACT_SCHEMA_VERSION = "dabench_vendor_contract_v1"
MANIFEST_FILENAME = "dabench_perception_dataset_manifest.json"
DEFAULT_TASK_COUNT = 257
DEFAULT_FAMILY_COUNT = 51
DEFAULT_VENDOR_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dslighting"
    / "benchmark"
    / "vendor"
    / "dabench"
    / "competitions"
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "source_root",
    "clean_root",
    "task_count",
    "family_count",
    "tasks",
    "families",
}
_TASK_KEYS = {
    "task_id",
    "family_id",
    "raw_filename",
    "raw_sha256",
    "public_train_sha256",
    "sample_submission_sha256",
    "private_answer_sha256",
    "vendor_contract_sha256",
    "vendor_contract_files",
}
_VENDOR_FILE_KEYS = {"path", "sha256", "size_bytes"}
_FAMILY_KEYS = {
    "family_id",
    "raw_sha256",
    "public_train_sha256",
    "representative_task_id",
    "task_ids",
}


class PreparationError(RuntimeError):
    """Raised when a clean DABench release cannot be built or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vendor_contract(task_dir: Path, task_id: str) -> tuple[str, list[dict[str, Any]]]:
    """Return a deterministic inventory of the files that define one task.

    Python bytecode is a runtime cache rather than benchmark source, so
    ``__pycache__`` directories and ``*.pyc`` files are deliberately excluded.
    Every other directory entry must be a real directory or regular file;
    symlinks and special files make the contract unsafe and are rejected.
    """

    if not task_dir.is_dir() or task_dir.is_symlink():
        raise PreparationError(f"{task_id}: vendor task directory is missing or unsafe.")

    inventory: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise PreparationError(
                f"{task_id}: could not inspect vendor contract directory {directory}: {error}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(task_dir)
            if entry.is_symlink():
                raise PreparationError(
                    f"{task_id}: vendor contract contains symlink {relative.as_posix()!r}."
                )
            if entry.is_dir(follow_symlinks=False):
                if entry.name != "__pycache__":
                    visit(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise PreparationError(
                    f"{task_id}: vendor contract contains non-regular entry "
                    f"{relative.as_posix()!r}."
                )
            if entry.name.endswith(".pyc"):
                continue
            try:
                size_bytes = entry.stat(follow_symlinks=False).st_size
            except OSError as error:
                raise PreparationError(
                    f"{task_id}: could not stat vendor contract file "
                    f"{relative.as_posix()!r}: {error}"
                ) from error
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": size_bytes,
                }
            )

    visit(task_dir)
    inventory.sort(key=lambda item: item["path"])
    contract = {
        "schema_version": VENDOR_CONTRACT_SCHEMA_VERSION,
        "files": inventory,
    }
    return _canonical_sha256(contract), inventory


def _family_id(raw_sha256: str) -> str:
    return f"dabench-family-{raw_sha256}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_exact_count(label: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise PreparationError(f"Expected {expected} {label}, found {actual}.")


def _task_directories(source_root: Path) -> list[Path]:
    task_dirs = sorted(
        (
            path
            for path in source_root.iterdir()
            if path.is_dir() and not path.is_symlink() and path.name.startswith("dabench-")
        ),
        key=lambda path: path.name,
    )
    if not task_dirs:
        raise PreparationError(f"No dabench-* task directories found under {source_root}.")
    return task_dirs


def _single_raw_csv(task_dir: Path) -> Path:
    raw_dir = task_dir / "raw"
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise PreparationError(f"{task_dir.name}: raw directory is missing or is a symlink.")
    raw_files = sorted(
        path for path in raw_dir.iterdir() if path.is_file() and not path.is_symlink()
    )
    csv_files = [path for path in raw_files if path.suffix.lower() == ".csv"]
    if len(raw_files) != 1 or len(csv_files) != 1:
        names = [path.name for path in raw_files]
        raise PreparationError(
            f"{task_dir.name}: expected exactly one regular raw CSV, found {names!r}."
        )
    return csv_files[0]


def _load_preparer(path: Path, task_id: str) -> Callable[[Path, Path, Path], Any]:
    if not path.is_file() or path.is_symlink():
        raise PreparationError(f"{task_id}: vendor preparer is missing: {path}.")
    module_name = (
        "_dslighting_dabench_clean_"
        + hashlib.sha256(f"{task_id}:{path}".encode("utf-8")).hexdigest()
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"{task_id}: could not load vendor preparer {path}.")
    module = importlib.util.module_from_spec(spec)
    _execute_module(spec.loader.exec_module, module, task_id, path)
    prepare = getattr(module, "prepare", None)
    if not callable(prepare):
        raise PreparationError(f"{task_id}: {path} does not define callable prepare().")
    return prepare


def _execute_module(
    execute: Callable[[ModuleType], None],
    module: ModuleType,
    task_id: str,
    path: Path,
) -> None:
    try:
        execute(module)
    except Exception as error:
        raise PreparationError(f"{task_id}: importing {path} failed: {error}") from error


def _run_preparer(
    prepare: Callable[[Path, Path, Path], Any],
    *,
    task_id: str,
    raw_dir: Path,
    public_dir: Path,
    private_dir: Path,
) -> None:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(captured_stdout),
            contextlib.redirect_stderr(captured_stderr),
        ):
            prepare(raw_dir, public_dir, private_dir)
    except Exception as error:
        details = "\n".join(
            text.strip()
            for text in (captured_stdout.getvalue(), captured_stderr.getvalue())
            if text.strip()
        )
        suffix = f"\nVendor output:\n{details}" if details else ""
        raise PreparationError(f"{task_id}: vendor prepare() failed: {error}{suffix}") from error


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise PreparationError(f"{label} is missing or is not a regular file: {path}.")


def _validate_prepared_task(task_root: Path, task_id: str) -> tuple[Path, Path, Path]:
    public_dir = task_root / "prepared" / "public"
    private_dir = task_root / "prepared" / "private"
    if not public_dir.is_dir() or public_dir.is_symlink():
        raise PreparationError(f"{task_id}: prepared/public is missing or unsafe.")
    if not private_dir.is_dir() or private_dir.is_symlink():
        raise PreparationError(f"{task_id}: prepared/private is missing or unsafe.")
    public_entries = sorted(path.name for path in public_dir.iterdir())
    expected_public = ["sample_submission.csv", "train.csv"]
    if public_entries != expected_public:
        raise PreparationError(
            f"{task_id}: public files must be exactly {expected_public!r}; "
            f"found {public_entries!r}."
        )
    train_path = public_dir / "train.csv"
    sample_submission_path = public_dir / "sample_submission.csv"
    answer_path = private_dir / "answer.csv"
    _regular_file(train_path, f"{task_id}: train.csv")
    _regular_file(sample_submission_path, f"{task_id}: sample_submission.csv")
    _regular_file(answer_path, f"{task_id}: private answer.csv")
    return train_path, sample_submission_path, answer_path


def _strict_json_load(path: Path) -> Mapping[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreparationError(f"Duplicate JSON key {key!r} in {path}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PreparationError(f"Non-finite JSON value {value!r} in {path}.")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except PreparationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"Could not read strict JSON manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PreparationError(f"Manifest {path} must contain a JSON object.")
    return payload


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PreparationError(
            f"{label} keys do not match the schema; "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}."
        )


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreparationError(f"{label} must be a non-empty string.")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PreparationError(f"{label} must be a lowercase SHA-256 hex digest.")
    return digest


def _validate_vendor_inventory(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise PreparationError(f"{label} must be a JSON array.")
    inventory: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, dict):
            raise PreparationError(f"{item_label} must be a JSON object.")
        _require_keys(item, _VENDOR_FILE_KEYS, item_label)
        relative_path = _require_string(item["path"], f"{item_label}.path")
        parsed_path = Path(relative_path)
        if (
            parsed_path.is_absolute()
            or relative_path != parsed_path.as_posix()
            or ".." in parsed_path.parts
            or "." in parsed_path.parts
        ):
            raise PreparationError(f"{item_label}.path must be a normalized relative POSIX path.")
        _require_sha256(item["sha256"], f"{item_label}.sha256")
        size_bytes = item["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise PreparationError(f"{item_label}.size_bytes must be a non-negative integer.")
        paths.append(relative_path)
        inventory.append(item)
    if paths != sorted(set(paths)):
        raise PreparationError(f"{label} paths must be unique and sorted.")
    return inventory


def _validate_manifest(
    payload: Mapping[str, Any],
    *,
    clean_root: Path,
    content_root: Path | None = None,
    expected_task_count: int | None,
    expected_family_count: int | None,
    source_root: Path | None = None,
    vendor_root: Path | None = None,
) -> None:
    _require_keys(payload, _TOP_LEVEL_KEYS, "Manifest")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PreparationError(
            f"Unsupported schema_version {payload['schema_version']!r}; expected {SCHEMA_VERSION!r}."
        )

    recorded_clean_root = Path(_require_string(payload["clean_root"], "clean_root"))
    if recorded_clean_root.resolve() != clean_root.resolve():
        raise PreparationError(
            f"Manifest clean_root {recorded_clean_root} does not match {clean_root.resolve()}."
        )
    recorded_source_root = Path(_require_string(payload["source_root"], "source_root"))
    if source_root is not None and recorded_source_root.resolve() != source_root.resolve():
        raise PreparationError(
            f"Manifest source_root {recorded_source_root} does not match {source_root.resolve()}."
        )
    content_root = content_root or clean_root

    tasks = payload["tasks"]
    families = payload["families"]
    if not isinstance(tasks, list) or not isinstance(families, list):
        raise PreparationError("Manifest tasks and families must be JSON arrays.")
    if isinstance(payload["task_count"], bool) or not isinstance(payload["task_count"], int):
        raise PreparationError("Manifest task_count must be an integer.")
    if isinstance(payload["family_count"], bool) or not isinstance(payload["family_count"], int):
        raise PreparationError("Manifest family_count must be an integer.")
    if payload["task_count"] != len(tasks):
        raise PreparationError("Manifest task_count does not equal the tasks array length.")
    if payload["family_count"] != len(families):
        raise PreparationError("Manifest family_count does not equal the families array length.")
    _require_exact_count("tasks", len(tasks), expected_task_count)
    _require_exact_count("dataset families", len(families), expected_family_count)

    task_by_id: dict[str, Mapping[str, Any]] = {}
    task_ids_in_order: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise PreparationError(f"tasks[{index}] must be a JSON object.")
        _require_keys(task, _TASK_KEYS, f"tasks[{index}]")
        task_id = _require_string(task["task_id"], f"tasks[{index}].task_id")
        family_id = _require_string(task["family_id"], f"tasks[{index}].family_id")
        raw_filename = _require_string(task["raw_filename"], f"tasks[{index}].raw_filename")
        raw_sha256 = _require_sha256(task["raw_sha256"], f"tasks[{index}].raw_sha256")
        train_sha256 = _require_sha256(
            task["public_train_sha256"], f"tasks[{index}].public_train_sha256"
        )
        sample_sha256 = _require_sha256(
            task["sample_submission_sha256"],
            f"tasks[{index}].sample_submission_sha256",
        )
        answer_sha256 = _require_sha256(
            task["private_answer_sha256"], f"tasks[{index}].private_answer_sha256"
        )
        vendor_contract_sha256 = _require_sha256(
            task["vendor_contract_sha256"], f"tasks[{index}].vendor_contract_sha256"
        )
        vendor_contract_files = _validate_vendor_inventory(
            task["vendor_contract_files"], f"tasks[{index}].vendor_contract_files"
        )
        recorded_contract = {
            "schema_version": VENDOR_CONTRACT_SCHEMA_VERSION,
            "files": vendor_contract_files,
        }
        if _canonical_sha256(recorded_contract) != vendor_contract_sha256:
            raise PreparationError(
                f"{task_id}: vendor contract SHA-256 does not match its file inventory."
            )
        if task_id in task_by_id:
            raise PreparationError(f"Duplicate task_id {task_id!r} in manifest.")
        if Path(raw_filename).name != raw_filename:
            raise PreparationError(f"{task_id}: raw_filename must be a basename.")
        if family_id != _family_id(raw_sha256):
            raise PreparationError(f"{task_id}: family_id is not derived from raw_sha256.")

        clean_task_root = content_root / task_id
        clean_raw = clean_task_root / "raw" / raw_filename
        _regular_file(clean_raw, f"{task_id}: clean raw CSV")
        if _sha256(clean_raw) != raw_sha256:
            raise PreparationError(f"{task_id}: clean raw SHA-256 does not match the manifest.")
        train_path, sample_path, answer_path = _validate_prepared_task(clean_task_root, task_id)
        if _sha256(train_path) != train_sha256:
            raise PreparationError(f"{task_id}: public train SHA-256 does not match the manifest.")
        if _sha256(sample_path) != sample_sha256:
            raise PreparationError(
                f"{task_id}: sample submission SHA-256 does not match the manifest."
            )
        if _sha256(answer_path) != answer_sha256:
            raise PreparationError(
                f"{task_id}: private answer SHA-256 does not match the manifest."
            )

        if source_root is not None:
            source_raw = _single_raw_csv(source_root / task_id)
            if source_raw.name != raw_filename or _sha256(source_raw) != raw_sha256:
                raise PreparationError(f"{task_id}: source raw CSV no longer matches the manifest.")

        if vendor_root is not None:
            actual_contract_sha256, actual_contract_files = _vendor_contract(
                vendor_root / task_id, task_id
            )
            if (
                actual_contract_sha256 != vendor_contract_sha256
                or actual_contract_files != vendor_contract_files
            ):
                raise PreparationError(
                    f"{task_id}: vendor contract no longer matches the manifest."
                )

        task_by_id[task_id] = task
        task_ids_in_order.append(task_id)
    if task_ids_in_order != sorted(task_ids_in_order):
        raise PreparationError("Manifest tasks must be sorted by task_id.")

    family_ids_in_order: list[str] = []
    covered_tasks: list[str] = []
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise PreparationError(f"families[{index}] must be a JSON object.")
        _require_keys(family, _FAMILY_KEYS, f"families[{index}]")
        family_id = _require_string(family["family_id"], f"families[{index}].family_id")
        raw_sha256 = _require_sha256(family["raw_sha256"], f"families[{index}].raw_sha256")
        train_sha256 = _require_sha256(
            family["public_train_sha256"], f"families[{index}].public_train_sha256"
        )
        representative = _require_string(
            family["representative_task_id"], f"families[{index}].representative_task_id"
        )
        family_task_ids = family["task_ids"]
        if not isinstance(family_task_ids, list) or not family_task_ids:
            raise PreparationError(f"{family_id}: task_ids must be a non-empty array.")
        if not all(isinstance(task_id, str) and task_id for task_id in family_task_ids):
            raise PreparationError(f"{family_id}: every task_id must be a non-empty string.")
        if family_task_ids != sorted(set(family_task_ids)):
            raise PreparationError(f"{family_id}: task_ids must be unique and sorted.")
        if representative != family_task_ids[0]:
            raise PreparationError(
                f"{family_id}: representative_task_id must be the first sorted task_id."
            )
        if family_id != _family_id(raw_sha256):
            raise PreparationError(f"{family_id}: family_id is not derived from raw_sha256.")
        for task_id in family_task_ids:
            task = task_by_id.get(task_id)
            if task is None:
                raise PreparationError(f"{family_id}: unknown task_id {task_id!r}.")
            if (
                task["family_id"] != family_id
                or task["raw_sha256"] != raw_sha256
                or task["public_train_sha256"] != train_sha256
            ):
                raise PreparationError(f"{family_id}: task {task_id!r} has inconsistent hashes.")
        family_ids_in_order.append(family_id)
        covered_tasks.extend(family_task_ids)
    if family_ids_in_order != sorted(set(family_ids_in_order)):
        raise PreparationError("Manifest families must be unique and sorted by family_id.")
    if sorted(covered_tasks) != sorted(task_by_id):
        raise PreparationError("Manifest families do not cover every task exactly once.")


def verify_clean_release(
    clean_root: Path,
    *,
    source_root: Path | None = None,
    vendor_root: Path | None = None,
    expected_task_count: int | None = DEFAULT_TASK_COUNT,
    expected_family_count: int | None = DEFAULT_FAMILY_COUNT,
) -> Mapping[str, Any]:
    """Verify an already published clean release and return its manifest."""

    clean_root = clean_root.expanduser().resolve()
    if not clean_root.is_dir() or clean_root.is_symlink():
        raise PreparationError(f"Clean root is missing or unsafe: {clean_root}.")
    resolved_source = source_root.expanduser().resolve() if source_root is not None else None
    resolved_vendor = vendor_root.expanduser().resolve() if vendor_root is not None else None
    if resolved_vendor is not None and (
        not resolved_vendor.is_dir() or resolved_vendor.is_symlink()
    ):
        raise PreparationError(f"Vendor root is missing or unsafe: {resolved_vendor}.")
    payload = _strict_json_load(clean_root / MANIFEST_FILENAME)
    _validate_manifest(
        payload,
        clean_root=clean_root,
        source_root=resolved_source,
        vendor_root=resolved_vendor,
        expected_task_count=expected_task_count,
        expected_family_count=expected_family_count,
    )
    return payload


def prepare_clean_release(
    source_root: Path,
    clean_root: Path,
    *,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    expected_task_count: int | None = DEFAULT_TASK_COUNT,
    expected_family_count: int | None = DEFAULT_FAMILY_COUNT,
) -> Mapping[str, Any]:
    """Rebuild DABench into a new clean root and return its integrity manifest."""

    source_root = source_root.expanduser().resolve()
    clean_root = clean_root.expanduser().resolve()
    vendor_root = vendor_root.expanduser().resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise PreparationError(f"Source root is missing or unsafe: {source_root}.")
    if not vendor_root.is_dir() or vendor_root.is_symlink():
        raise PreparationError(f"Vendor root is missing or unsafe: {vendor_root}.")
    if clean_root.exists() or clean_root.is_symlink():
        raise PreparationError(f"Refusing to overwrite existing clean root: {clean_root}.")
    if _is_relative_to(clean_root, source_root) or _is_relative_to(source_root, clean_root):
        raise PreparationError("Source root and clean root must be disjoint directories.")

    task_dirs = _task_directories(source_root)
    _require_exact_count("tasks", len(task_dirs), expected_task_count)
    clean_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{clean_root.name}.staging-", dir=clean_root.parent)
    )
    published = False
    try:
        tasks: list[dict[str, Any]] = []
        family_tasks: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_task_root in task_dirs:
            task_id = source_task_root.name
            raw_source = _single_raw_csv(source_task_root)
            raw_sha256 = _sha256(raw_source)
            family_id = _family_id(raw_sha256)
            vendor_task_root = vendor_root / task_id
            vendor_contract_sha256, vendor_contract_files = _vendor_contract(
                vendor_task_root, task_id
            )

            stage_task_root = stage_root / task_id
            stage_raw_dir = stage_task_root / "raw"
            stage_public_dir = stage_task_root / "prepared" / "public"
            stage_private_dir = stage_task_root / "prepared" / "private"
            stage_raw_dir.mkdir(parents=True)
            stage_public_dir.mkdir(parents=True)
            stage_private_dir.mkdir(parents=True)
            raw_copy = stage_raw_dir / raw_source.name
            shutil.copyfile(raw_source, raw_copy)
            if _sha256(raw_copy) != raw_sha256:
                raise PreparationError(f"{task_id}: raw CSV changed while it was copied.")

            prepare = _load_preparer(vendor_task_root / "prepare.py", task_id)
            _run_preparer(
                prepare,
                task_id=task_id,
                raw_dir=stage_raw_dir,
                public_dir=stage_public_dir,
                private_dir=stage_private_dir,
            )
            train_path, sample_path, answer_path = _validate_prepared_task(stage_task_root, task_id)
            task = {
                "task_id": task_id,
                "family_id": family_id,
                "raw_filename": raw_source.name,
                "raw_sha256": raw_sha256,
                "public_train_sha256": _sha256(train_path),
                "sample_submission_sha256": _sha256(sample_path),
                "private_answer_sha256": _sha256(answer_path),
                "vendor_contract_sha256": vendor_contract_sha256,
                "vendor_contract_files": vendor_contract_files,
            }
            tasks.append(task)
            family_tasks[family_id].append(task)

        families: list[dict[str, Any]] = []
        for family_id in sorted(family_tasks):
            members = sorted(family_tasks[family_id], key=lambda task: task["task_id"])
            raw_hashes = {task["raw_sha256"] for task in members}
            train_hashes = {task["public_train_sha256"] for task in members}
            if len(raw_hashes) != 1:
                raise PreparationError(f"{family_id}: family members have different raw hashes.")
            if len(train_hashes) != 1:
                details = ", ".join(
                    f"{task['task_id']}={task['public_train_sha256']}" for task in members
                )
                raise PreparationError(
                    f"{family_id}: identical raw data produced different train.csv bytes: {details}."
                )
            task_ids = [task["task_id"] for task in members]
            families.append(
                {
                    "family_id": family_id,
                    "raw_sha256": members[0]["raw_sha256"],
                    "public_train_sha256": members[0]["public_train_sha256"],
                    "representative_task_id": task_ids[0],
                    "task_ids": task_ids,
                }
            )
        _require_exact_count("dataset families", len(families), expected_family_count)

        tasks.sort(key=lambda task: task["task_id"])
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_root": str(source_root),
            "clean_root": str(clean_root),
            "task_count": len(tasks),
            "family_count": len(families),
            "tasks": tasks,
            "families": families,
        }
        manifest_path = stage_root / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _validate_manifest(
            payload,
            clean_root=clean_root,
            content_root=stage_root,
            source_root=source_root,
            vendor_root=vendor_root,
            expected_task_count=expected_task_count,
            expected_family_count=expected_family_count,
        )

        # A second existence check protects callers that create the destination
        # while preparation is in progress.  os.rename publishes one complete
        # directory tree; no partial clean root is ever visible.
        if clean_root.exists() or clean_root.is_symlink():
            raise PreparationError(f"Refusing to overwrite existing clean root: {clean_root}.")
        os.rename(stage_root, clean_root)
        published = True
        return payload
    finally:
        if not published and stage_root.exists():
            shutil.rmtree(stage_root)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify clean, integrity-attested DABench perception inputs."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Original DABench task root. Required when building; optional for verify-only.",
    )
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        help=(
            "Vendored DABench competition root. Defaults to the repository copy when "
            "building; optional during verify-only to re-attest the vendor contract."
        ),
    )
    parser.add_argument("--expected-task-count", type=_positive_int, default=DEFAULT_TASK_COUNT)
    parser.add_argument("--expected-family-count", type=_positive_int, default=DEFAULT_FAMILY_COUNT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing clean root without executing vendor preparers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.verify_only:
            payload = verify_clean_release(
                args.clean_root,
                source_root=args.source_root,
                vendor_root=args.vendor_root,
                expected_task_count=args.expected_task_count,
                expected_family_count=args.expected_family_count,
            )
            action = "verified"
        else:
            if args.source_root is None:
                raise PreparationError("--source-root is required unless --verify-only is used.")
            payload = prepare_clean_release(
                args.source_root,
                args.clean_root,
                vendor_root=args.vendor_root or DEFAULT_VENDOR_ROOT,
                expected_task_count=args.expected_task_count,
                expected_family_count=args.expected_family_count,
            )
            action = "published"
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "action": action,
                "clean_root": payload["clean_root"],
                "manifest": str(Path(payload["clean_root"]) / MANIFEST_FILENAME),
                "task_count": payload["task_count"],
                "family_count": payload["family_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
