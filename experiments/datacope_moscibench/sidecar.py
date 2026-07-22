"""Launch the MoSciBench DataCOPE bridge with the existing bwrap runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from experiments.datacope_dabench.sidecar import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_SIDECAR_TIMEOUT_SECONDS,
    _codex_home_and_auth,
    _codex_runtime,
    _discovery_command,
    _discovery_environment,
    _python_executable,
    _require_empty_output,
)

from .adapter import validate_public_data
from .datacope_bridge import validate_prediction_runs
from .protocol import validate_round_index
from .split import DEFAULT_SPLIT_MANIFEST, load_split

BRIDGE_PATH = Path(__file__).with_name("datacope_bridge.py")
BRIDGE_FILES = tuple(
    BRIDGE_PATH.with_name(name)
    for name in ("adapter.py", "datacope_bridge.py", "protocol.py", "split.py")
)


def run_sidecar(
    *,
    datacope_root: Path,
    predictions_dir: Path,
    verified_dir: Path,
    split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
    python_executable: str = sys.executable,
    data_dir: Path,
    skill_dir: Path,
    model: str = DEFAULT_CODEX_MODEL,
    codex_executable: str = "codex",
    round_index: int = 0,
    previous_predictions_dirs: Sequence[Path] = (),
    previous_skill_dir: Path | None = None,
    timeout_seconds: int = DEFAULT_SIDECAR_TIMEOUT_SECONDS,
) -> int:
    resolved_round = validate_round_index(round_index)
    python_path = _python_executable(python_executable)
    datacope = Path(datacope_root).expanduser().resolve()
    predictions = Path(predictions_dir).expanduser().resolve()
    verified = Path(verified_dir).expanduser().resolve()
    manifest = Path(split_manifest).expanduser().resolve()
    expected_ids = load_split(manifest).query_ids("explore")
    if validate_prediction_runs(predictions) != expected_ids:
        raise ValueError("Predictions differ from the frozen explore split")

    command = [
        str(python_path),
        str(BRIDGE_PATH),
        "--datacope-root",
        str(datacope),
        "--predictions-dir",
        str(predictions),
        "--verified-dir",
        str(verified),
        "--split-manifest",
        str(manifest),
    ]
    public_data = Path(data_dir).expanduser().resolve()
    public_roots = validate_public_data(public_data, expected_ids)
    skill = Path(skill_dir).expanduser().resolve()
    previous_predictions = tuple(
        Path(path).expanduser().resolve() for path in previous_predictions_dirs
    )
    if len(previous_predictions) != resolved_round:
        raise ValueError(
            f"Round {resolved_round} requires {resolved_round} previous prediction dirs"
        )
    previous_skill = (
        Path(previous_skill_dir).expanduser().resolve() if previous_skill_dir is not None else None
    )
    if resolved_round and previous_skill is None:
        raise ValueError(f"Round {resolved_round} requires the previous skill")
    if not resolved_round and previous_skill is not None:
        raise ValueError("Round 0 must not receive a previous skill")

    codex_runtime = _codex_runtime(codex_executable)
    command.extend(
        (
            "--round-index",
            str(resolved_round),
            "--data-dir",
            str(public_data),
            "--skill-dir",
            str(skill),
            "--model",
            model,
            "--codex-bin",
            str(codex_runtime.executable),
            "--codex-version",
            codex_runtime.version,
            "--codex-sha256",
            codex_runtime.sha256,
        )
    )
    for path in previous_predictions:
        command.extend(("--previous-predictions-dir", str(path)))
    if previous_skill is not None:
        command.extend(("--previous-skill-dir", str(previous_skill)))

    verified = _require_empty_output(verified, "Verified output directory")
    skill = _require_empty_output(skill, "Skill output directory")
    codex_home, auth_file = _codex_home_and_auth()
    environment = _discovery_environment(python_path, codex_home, codex_runtime.root)
    command = _discovery_command(
        command,
        python_executable=python_path,
        datacope_root=datacope,
        predictions_dir=predictions,
        verified_dir=verified,
        data_dir=public_data,
        skill_dir=skill,
        family_manifest=manifest,
        auth_file=auth_file,
        codex_runtime_root=codex_runtime.root,
        previous_predictions_dirs=previous_predictions,
        previous_skill_dir=previous_skill,
        extra_read_only=(*BRIDGE_FILES, *public_roots),
    )
    try:
        completed = subprocess.run(command, check=False, env=environment, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"DataCOPE discovery timed out after {timeout_seconds}s") from exc
    return completed.returncode
