from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.data_card_ablation import prepare_dabench_perception_data as preparer


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_task(
    source_root: Path,
    vendor_root: Path,
    task_id: str,
    *,
    raw_filename: str,
    raw_bytes: bytes,
    extra_public: bool = False,
    altered_train: bool = False,
    omit_answer: bool = False,
) -> None:
    source_task = source_root / task_id
    raw_dir = source_task / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / raw_filename).write_bytes(raw_bytes)

    # A polluted source prepared tree is intentionally ignored by the builder.
    polluted = source_task / "prepared" / "public"
    polluted.mkdir(parents=True)
    (polluted / "train.csv").write_text("a,ratio\n1,999\n", encoding="utf-8")

    vendor_task = vendor_root / task_id
    vendor_task.mkdir(parents=True)
    train_statement = (
        '(public / "train.csv").write_bytes(source.read_bytes() + b"derived\\n")'
        if altered_train
        else '(public / "train.csv").write_bytes(source.read_bytes())'
    )
    extra_statement = (
        '(public / "unexpected.csv").write_text("leak\\n", encoding="utf-8")'
        if extra_public
        else ""
    )
    answer_statement = (
        ""
        if omit_answer
        else '(private / "answer.csv").write_text("id,answer\\n1,ok\\n", encoding="utf-8")'
    )
    (vendor_task / "prepare.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def prepare(raw: Path, public: Path, private: Path):",
                f"    source = raw / {raw_filename!r}",
                f"    {train_statement}",
                '    (public / "sample_submission.csv").write_text('
                '"id,answer\\n1,pending\\n", encoding="utf-8")',
                f"    {extra_statement}" if extra_statement else "    pass",
                f"    {answer_statement}" if answer_statement else "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (vendor_task / "description.md").write_text(
        f"# {task_id}\n\nCompute the requested result.\n", encoding="utf-8"
    )
    (vendor_task / "config.yaml").write_text("metric: exact_match\n", encoding="utf-8")
    (vendor_task / "evaluate.py").write_text(
        "def evaluate(submission, answer):\n    return submission == answer\n",
        encoding="utf-8",
    )
    (vendor_task / ".contract-source").write_text("included\n", encoding="utf-8")


def _build_two_task_family(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    source_root = tmp_path / "source"
    vendor_root = tmp_path / "vendor"
    clean_root = tmp_path / "clean"
    raw_bytes = b"a,b\n1,2\n3,4\n"
    _write_task(
        source_root,
        vendor_root,
        "dabench-2-second",
        raw_filename="shared.csv",
        raw_bytes=raw_bytes,
    )
    _write_task(
        source_root,
        vendor_root,
        "dabench-1-first",
        raw_filename="shared.csv",
        raw_bytes=raw_bytes,
    )
    manifest = preparer.prepare_clean_release(
        source_root,
        clean_root,
        vendor_root=vendor_root,
        expected_task_count=2,
        expected_family_count=1,
    )
    return source_root, vendor_root, clean_root, dict(manifest)


def test_builds_clean_release_and_stable_family_manifest(tmp_path: Path) -> None:
    source_root, vendor_root, clean_root, manifest = _build_two_task_family(tmp_path)
    raw_bytes = b"a,b\n1,2\n3,4\n"
    raw_hash = _sha256(raw_bytes)
    family_id = f"dabench-family-{raw_hash}"
    sample_hash = _sha256(b"id,answer\n1,pending\n")
    answer_hash = _sha256(b"id,answer\n1,ok\n")
    first_contract_hash, first_contract_files = preparer._vendor_contract(
        vendor_root / "dabench-1-first", "dabench-1-first"
    )
    second_contract_hash, second_contract_files = preparer._vendor_contract(
        vendor_root / "dabench-2-second", "dabench-2-second"
    )

    assert manifest == {
        "schema_version": "dabench_perception_dataset_manifest_v1",
        "source_root": str(source_root.resolve()),
        "clean_root": str(clean_root.resolve()),
        "task_count": 2,
        "family_count": 1,
        "tasks": [
            {
                "task_id": "dabench-1-first",
                "family_id": family_id,
                "raw_filename": "shared.csv",
                "raw_sha256": raw_hash,
                "public_train_sha256": raw_hash,
                "sample_submission_sha256": sample_hash,
                "private_answer_sha256": answer_hash,
                "vendor_contract_sha256": first_contract_hash,
                "vendor_contract_files": first_contract_files,
            },
            {
                "task_id": "dabench-2-second",
                "family_id": family_id,
                "raw_filename": "shared.csv",
                "raw_sha256": raw_hash,
                "public_train_sha256": raw_hash,
                "sample_submission_sha256": sample_hash,
                "private_answer_sha256": answer_hash,
                "vendor_contract_sha256": second_contract_hash,
                "vendor_contract_files": second_contract_files,
            },
        ],
        "families": [
            {
                "family_id": family_id,
                "raw_sha256": raw_hash,
                "public_train_sha256": raw_hash,
                "representative_task_id": "dabench-1-first",
                "task_ids": ["dabench-1-first", "dabench-2-second"],
            }
        ],
    }
    on_disk = json.loads((clean_root / preparer.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert [item["path"] for item in first_contract_files] == [
        ".contract-source",
        "config.yaml",
        "description.md",
        "evaluate.py",
        "prepare.py",
    ]
    for task_id in ("dabench-1-first", "dabench-2-second"):
        public = clean_root / task_id / "prepared" / "public"
        assert sorted(path.name for path in public.iterdir()) == [
            "sample_submission.csv",
            "train.csv",
        ]
        assert (public / "train.csv").read_bytes() == raw_bytes
        assert "ratio" not in (public / "train.csv").read_text(encoding="utf-8")
        assert (clean_root / task_id / "prepared" / "private" / "answer.csv").is_file()


def test_refuses_to_overwrite_an_existing_target(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    vendor_root = tmp_path / "vendor"
    clean_root = tmp_path / "clean"
    _write_task(
        source_root,
        vendor_root,
        "dabench-1-only",
        raw_filename="data.csv",
        raw_bytes=b"x\n1\n",
    )
    clean_root.mkdir()
    sentinel = clean_root / "belongs-to-user.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(preparer.PreparationError, match="Refusing to overwrite"):
        preparer.prepare_clean_release(
            source_root,
            clean_root,
            vendor_root=vendor_root,
            expected_task_count=1,
            expected_family_count=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_rejects_different_rebuilt_train_bytes_for_same_raw_family(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    vendor_root = tmp_path / "vendor"
    clean_root = tmp_path / "clean"
    raw_bytes = b"x\n1\n"
    _write_task(
        source_root,
        vendor_root,
        "dabench-1-normal",
        raw_filename="data.csv",
        raw_bytes=raw_bytes,
    )
    _write_task(
        source_root,
        vendor_root,
        "dabench-2-divergent",
        raw_filename="data.csv",
        raw_bytes=raw_bytes,
        altered_train=True,
    )

    with pytest.raises(preparer.PreparationError, match="different train.csv bytes"):
        preparer.prepare_clean_release(
            source_root,
            clean_root,
            vendor_root=vendor_root,
            expected_task_count=2,
            expected_family_count=1,
        )

    assert not clean_root.exists()
    assert not list(tmp_path.glob(".clean.staging-*"))


@pytest.mark.parametrize(
    ("extra_public", "omit_answer", "message"),
    [
        (True, False, "public files must be exactly"),
        (False, True, "private answer.csv"),
    ],
)
def test_failed_preparer_output_is_never_published(
    tmp_path: Path,
    extra_public: bool,
    omit_answer: bool,
    message: str,
) -> None:
    source_root = tmp_path / "source"
    vendor_root = tmp_path / "vendor"
    clean_root = tmp_path / "clean"
    _write_task(
        source_root,
        vendor_root,
        "dabench-1-invalid",
        raw_filename="data.csv",
        raw_bytes=b"x\n1\n",
        extra_public=extra_public,
        omit_answer=omit_answer,
    )

    with pytest.raises(preparer.PreparationError, match=message):
        preparer.prepare_clean_release(
            source_root,
            clean_root,
            vendor_root=vendor_root,
            expected_task_count=1,
            expected_family_count=1,
        )

    assert not clean_root.exists()
    assert not list(tmp_path.glob(".clean.staging-*"))


def test_verify_only_detects_clean_data_tampering(tmp_path: Path) -> None:
    source_root, vendor_root, clean_root, manifest = _build_two_task_family(tmp_path)

    verified = preparer.verify_clean_release(
        clean_root,
        source_root=source_root,
        vendor_root=vendor_root,
        expected_task_count=2,
        expected_family_count=1,
    )
    assert verified == manifest

    train = clean_root / "dabench-1-first" / "prepared" / "public" / "train.csv"
    train.write_text("a,b,ratio\n1,2,0.5\n", encoding="utf-8")
    with pytest.raises(preparer.PreparationError, match="public train SHA-256"):
        preparer.verify_clean_release(
            clean_root,
            expected_task_count=2,
            expected_family_count=1,
        )


@pytest.mark.parametrize(
    ("relative_path", "expected_message"),
    [
        (
            "prepared/public/sample_submission.csv",
            "sample submission SHA-256",
        ),
        ("prepared/private/answer.csv", "private answer SHA-256"),
    ],
)
def test_verify_only_detects_other_scoring_input_tampering(
    tmp_path: Path,
    relative_path: str,
    expected_message: str,
) -> None:
    _, _, clean_root, _ = _build_two_task_family(tmp_path)
    target = clean_root / "dabench-1-first" / relative_path
    target.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(preparer.PreparationError, match=expected_message):
        preparer.verify_clean_release(
            clean_root,
            expected_task_count=2,
            expected_family_count=1,
        )


def test_verify_only_rejects_symlinked_private_directory(tmp_path: Path) -> None:
    _, _, clean_root, _ = _build_two_task_family(tmp_path)
    private_dir = clean_root / "dabench-1-first" / "prepared" / "private"
    outside = tmp_path / "outside-private"
    private_dir.rename(outside)
    private_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(preparer.PreparationError, match="prepared/private is missing or unsafe"):
        preparer.verify_clean_release(
            clean_root,
            expected_task_count=2,
            expected_family_count=1,
        )


@pytest.mark.parametrize("vendor_relative_path", ["description.md", ".new-source-file"])
def test_verify_only_detects_vendor_contract_tampering(
    tmp_path: Path, vendor_relative_path: str
) -> None:
    _, vendor_root, clean_root, _ = _build_two_task_family(tmp_path)
    target = vendor_root / "dabench-1-first" / vendor_relative_path
    target.write_text("changed\n", encoding="utf-8")

    with pytest.raises(preparer.PreparationError, match="vendor contract no longer matches"):
        preparer.verify_clean_release(
            clean_root,
            vendor_root=vendor_root,
            expected_task_count=2,
            expected_family_count=1,
        )


def test_vendor_contract_excludes_runtime_bytecode_caches(tmp_path: Path) -> None:
    _, vendor_root, clean_root, manifest = _build_two_task_family(tmp_path)
    vendor_task = vendor_root / "dabench-1-first"
    cache = vendor_task / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "prepare.cpython-313.pyc").write_bytes(b"runtime cache")
    (vendor_task / "stray.pyc").write_bytes(b"runtime cache")

    verified = preparer.verify_clean_release(
        clean_root,
        vendor_root=vendor_root,
        expected_task_count=2,
        expected_family_count=1,
    )

    assert verified == manifest


def test_build_rejects_vendor_contract_symlinks(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    vendor_root = tmp_path / "vendor"
    clean_root = tmp_path / "clean"
    task_id = "dabench-1-only"
    _write_task(
        source_root,
        vendor_root,
        task_id,
        raw_filename="data.csv",
        raw_bytes=b"x\n1\n",
    )
    (vendor_root / task_id / "linked-description.md").symlink_to("description.md")

    with pytest.raises(preparer.PreparationError, match="vendor contract contains symlink"):
        preparer.prepare_clean_release(
            source_root,
            clean_root,
            vendor_root=vendor_root,
            expected_task_count=1,
            expected_family_count=1,
        )


@pytest.mark.parametrize("mutation", ["reverse", "duplicate"])
def test_manifest_rejects_unsorted_or_duplicate_vendor_inventory(
    tmp_path: Path, mutation: str
) -> None:
    _, _, clean_root, _ = _build_two_task_family(tmp_path)
    manifest_path = clean_root / preparer.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["tasks"][0]["vendor_contract_files"]
    if mutation == "reverse":
        inventory.reverse()
    else:
        inventory.append(dict(inventory[-1]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(preparer.PreparationError, match="paths must be unique and sorted"):
        preparer.verify_clean_release(
            clean_root,
            expected_task_count=2,
            expected_family_count=1,
        )


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _, _, clean_root, _ = _build_two_task_family(tmp_path)
    manifest_path = clean_root / preparer.MANIFEST_FILENAME
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace('  "task_count": 2,', '  "task_count": 2,\n  "task_count": 2,'),
        encoding="utf-8",
    )

    with pytest.raises(preparer.PreparationError, match="Duplicate JSON key 'task_count'"):
        preparer.verify_clean_release(
            clean_root,
            expected_task_count=2,
            expected_family_count=1,
        )


def test_requires_exactly_one_raw_csv_per_task(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    vendor_root = tmp_path / "vendor"
    clean_root = tmp_path / "clean"
    _write_task(
        source_root,
        vendor_root,
        "dabench-1-invalid",
        raw_filename="first.csv",
        raw_bytes=b"x\n1\n",
    )
    (source_root / "dabench-1-invalid" / "raw" / "second.csv").write_text(
        "y\n2\n", encoding="utf-8"
    )

    with pytest.raises(preparer.PreparationError, match="exactly one regular raw CSV"):
        preparer.prepare_clean_release(
            source_root,
            clean_root,
            vendor_root=vendor_root,
            expected_task_count=1,
            expected_family_count=1,
        )
