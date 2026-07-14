"""Deterministic, metadata-only attestation of solver-visible public data."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

from .models import L1PublicCoverageError


@dataclass(frozen=True)
class PublicTreeRecord:
    """One logical public entry and the filesystem object that backs it."""

    relative_path: str
    kind: Literal["file", "directory"]
    resolved_path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class PublicTreeAttestation:
    """Stable public-tree fingerprints without reading potentially huge payloads."""

    file_count: int
    directory_count: int
    total_bytes: int
    shape_sha256: str
    physical_identity_sha256: str
    integrity_sha256: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
            "shape_sha256": self.shape_sha256,
            "physical_identity_sha256": self.physical_identity_sha256,
            "integrity_sha256": self.integrity_sha256,
        }


class PublicTreeAttestor:
    """Incrementally attest records already consumed by a coverage validator."""

    def __init__(self) -> None:
        self._shape = hashlib.sha256()
        self._physical_identity = hashlib.sha256()
        self._integrity = hashlib.sha256()
        self._file_count = 0
        self._directory_count = 0
        self._total_bytes = 0

    def add(self, record: PublicTreeRecord) -> None:
        relative = record.relative_path.encode("utf-8")
        kind = record.kind.encode("ascii")
        shape_size = record.size if record.kind == "file" else 0
        self._shape.update(_fingerprint_row(kind + b"\0" + relative, shape_size))
        if record.kind == "file":
            # File identity intentionally excludes directory inodes so tasks
            # that hardlink one family dataset can prove semantic equality.
            self._physical_identity.update(
                _fingerprint_row(relative, record.device, record.inode, record.size)
            )
        self._integrity.update(
            _fingerprint_row(
                kind + b"\0" + relative,
                record.device,
                record.inode,
                record.size,
                record.mtime_ns,
                record.ctime_ns,
            )
        )
        if record.kind == "file":
            self._file_count += 1
            self._total_bytes += record.size
        else:
            self._directory_count += 1

    def finish(self) -> PublicTreeAttestation:
        return PublicTreeAttestation(
            file_count=self._file_count,
            directory_count=self._directory_count,
            total_bytes=self._total_bytes,
            shape_sha256=self._shape.hexdigest(),
            physical_identity_sha256=self._physical_identity.hexdigest(),
            integrity_sha256=self._integrity.hexdigest(),
        )


def attest_public_tree(
    public_root: Path | str,
    *,
    excluded_paths: Iterable[Path | str] = (),
) -> PublicTreeAttestation:
    """Attest every non-excluded regular file under a canonical public root."""

    attestor = PublicTreeAttestor()
    for record in iter_public_tree(public_root, excluded_paths=excluded_paths):
        attestor.add(record)
    return attestor.finish()


def iter_public_tree(
    public_root: Path | str,
    *,
    excluded_paths: Iterable[Path | str] = (),
) -> Iterator[PublicTreeRecord]:
    """Yield public entries in deterministic path order with fail-closed symlinks."""

    root = Path(public_root).expanduser()
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise L1PublicCoverageError(f"Public root does not exist: {root}") from error
    if not resolved_root.is_dir():
        raise L1PublicCoverageError(f"Public root is not a directory: {resolved_root}")

    excluded = _resolve_excluded_paths(resolved_root, excluded_paths)
    yield from _walk_public_files(resolved_root, resolved_root, excluded)


def _walk_public_files(
    root: Path,
    directory: Path,
    excluded: frozenset[Path],
) -> Iterator[PublicTreeRecord]:
    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
    except OSError as error:
        raise L1PublicCoverageError(
            f"Visible public directory cannot be read: {directory}"
        ) from error

    for entry in entries:
        candidate = Path(entry.path)
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise L1PublicCoverageError(
                f"Visible public path cannot be inspected: {candidate}"
            ) from error

        resolved_candidate = candidate
        if stat.S_ISLNK(metadata.st_mode):
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except OSError as error:
                raise L1PublicCoverageError(
                    f"Visible public path cannot be resolved: {candidate}"
                ) from error
            try:
                resolved_candidate.relative_to(root)
            except ValueError as error:
                raise L1PublicCoverageError(
                    f"Visible public path escapes the public root: {candidate}"
                ) from error
            try:
                metadata = resolved_candidate.stat()
            except OSError as error:
                raise L1PublicCoverageError(
                    f"Visible public path cannot be inspected: {candidate}"
                ) from error

        if _is_excluded(resolved_candidate, excluded):
            continue
        if stat.S_ISDIR(metadata.st_mode):
            if candidate.is_symlink():
                raise L1PublicCoverageError(
                    "Visible public directory symlinks are not supported: "
                    f"{candidate.relative_to(root).as_posix()}"
                )
            yield _record(candidate, root=root, kind="directory", metadata=metadata)
            yield from _walk_public_files(root, candidate, excluded)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise L1PublicCoverageError(
                "Visible public input must contain only regular files and directories: "
                f"{candidate.relative_to(root).as_posix()}"
            )

        yield _record(
            candidate,
            root=root,
            kind="file",
            metadata=metadata,
            resolved_path=resolved_candidate,
        )


def _record(
    candidate: Path,
    *,
    root: Path,
    kind: Literal["file", "directory"],
    metadata: os.stat_result,
    resolved_path: Path | None = None,
) -> PublicTreeRecord:
    return PublicTreeRecord(
        relative_path=candidate.relative_to(root).as_posix(),
        kind=kind,
        resolved_path=resolved_path or candidate,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _resolve_excluded_paths(
    public_root: Path,
    excluded_paths: Iterable[Path | str],
) -> frozenset[Path]:
    resolved: set[Path] = set()
    for raw_path in excluded_paths:
        path = Path(raw_path).expanduser()
        candidate = path if path.is_absolute() else public_root / path
        normalized = candidate.resolve(strict=False)
        try:
            normalized.relative_to(public_root)
        except ValueError as error:
            if path.is_absolute():
                # Output artifacts commonly live outside public input. They
                # cannot be traversed here and therefore need no exclusion.
                continue
            raise L1PublicCoverageError(
                f"Excluded public path escapes the public root: {raw_path}"
            ) from error
        resolved.add(normalized)
    return frozenset(resolved)


def _is_excluded(candidate: Path, excluded: frozenset[Path]) -> bool:
    return any(candidate == path or candidate.is_relative_to(path) for path in excluded)


def _fingerprint_row(relative: bytes, *values: int) -> bytes:
    numeric = b"\0".join(str(value).encode("ascii") for value in values)
    return relative + b"\0" + numeric + b"\n"


__all__ = [
    "PublicTreeRecord",
    "PublicTreeAttestation",
    "PublicTreeAttestor",
    "attest_public_tree",
    "iter_public_tree",
]
