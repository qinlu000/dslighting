"""Small bridge executed by the DataCOPE virtual environment.

This module intentionally imports DataCOPE only after its checkout has been
placed at the front of ``sys.path``. DataCOPE uses the generic top-level package
name ``src``, so importing it in the main DSLighting process is unsafe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .adapter import PUBLIC_DATA_FILES
    from .protocol import (
        PAPER_TRAJECTORIES_PER_TASK,
        validate_round_index,
    )
    from .split import load_dabench_split
else:
    from adapter import PUBLIC_DATA_FILES
    from protocol import PAPER_TRAJECTORIES_PER_TASK, validate_round_index
    from split import load_dabench_split


def group_predictions_exactly(predictions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group signature predictions by exact equality, preserving stable order."""

    groups: list[list[dict[str, Any]]] = []
    representatives: list[Any] = []
    for prediction in predictions:
        answer = prediction["answer"]
        for index, representative in enumerate(representatives):
            if answer == representative:
                groups[index].append(prediction)
                break
        else:
            representatives.append(answer)
            groups.append([prediction])
    return sorted(groups, key=len, reverse=True)


def validate_prediction_runs(predictions_dir: Path) -> set[str]:
    """Require the paper's ten flat sample runs with identical task IDs."""

    predictions_dir = predictions_dir.expanduser().resolve()
    if not predictions_dir.is_dir():
        raise ValueError(f"Predictions directory does not exist: {predictions_dir}")
    if list(predictions_dir.glob("prediction_*.json")):
        raise ValueError("Prediction files must be inside one subdirectory per sample run")

    run_dirs = sorted(
        path
        for path in predictions_dir.iterdir()
        if path.is_dir() and list(path.glob("prediction_*.json"))
    )
    if len(run_dirs) != PAPER_TRAJECTORIES_PER_TASK:
        raise ValueError(
            "Paper protocol requires exactly "
            f"{PAPER_TRAJECTORIES_PER_TASK} sample run directories; found {len(run_dirs)}"
        )

    accepted_files = {
        prediction_file.resolve()
        for run_dir in run_dirs
        for prediction_file in run_dir.glob("prediction_*.json")
    }
    all_files = {
        prediction_file.resolve() for prediction_file in predictions_dir.rglob("prediction_*.json")
    }
    if accepted_files != all_files:
        raise ValueError("Each sample run must contain prediction files directly, without nesting")

    query_sets: list[set[str]] = []
    for run_dir in run_dirs:
        query_ids: list[str] = []
        for prediction_file in sorted(run_dir.glob("prediction_*.json")):
            try:
                payload = json.loads(prediction_file.read_text(encoding="utf-8"))
                query_id = str(payload["extra_info"]["query_id"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid DataCOPE prediction {prediction_file}: {exc}") from exc
            query_ids.append(query_id)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Duplicate query_id in sample run {run_dir}")
        query_sets.append(set(query_ids))

    expected = query_sets[0]
    for run_dir, query_ids in zip(run_dirs[1:], query_sets[1:]):
        if query_ids != expected:
            missing = sorted(expected - query_ids)
            extra = sorted(query_ids - expected)
            raise ValueError(
                f"Sample run {run_dir} has a different task set; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return expected


def validate_public_data(data_dir: Path, query_ids: set[str]) -> None:
    """Require an exact explore-only view with the two public DABench files."""

    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    top_level = list(data_dir.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in top_level):
        raise ValueError("Public data view may contain only regular task directories")

    actual_ids = {path.name for path in top_level}
    missing = sorted(query_ids - actual_ids)
    extra = sorted(actual_ids - query_ids)
    if missing or extra:
        raise ValueError(
            "Public data task set differs from predictions; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for task_id in sorted(query_ids):
        entries = list((data_dir / task_id).iterdir())
        names = {entry.name for entry in entries}
        if names != PUBLIC_DATA_FILES:
            raise ValueError(
                f"Unexpected public files for {task_id}; "
                f"expected={sorted(PUBLIC_DATA_FILES)}, actual={sorted(names)}"
            )
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError(f"Public data files must be regular files for {task_id}")


def require_empty_directory(path: Path, label: str) -> None:
    """Reject stale DataCOPE output instead of mixing it into a new run."""

    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{label} must be an empty or new directory: {path}")


def skill_output_contract(skill_dir: Path) -> str:
    """Tell the skill writer exactly where this experiment expects its output."""

    skill_file = skill_dir.expanduser().resolve() / "SKILL.md"
    return f"The final Skill file must be saved at exactly `{skill_file}`."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DataCOPE on exported DABench trajectories")
    parser.add_argument("mode", choices=("verify", "discover"))
    parser.add_argument("--datacope-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--previous-predictions-dir", type=Path, action="append", default=[])
    parser.add_argument("--previous-skill-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--family-manifest", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--codex-bin", type=Path)
    parser.add_argument("--codex-version")
    parser.add_argument("--codex-sha256")
    return parser


def _run(args: argparse.Namespace) -> dict[str, str]:
    datacope_root = args.datacope_root.expanduser().resolve()
    if not (datacope_root / "src" / "verifier" / "instances" / "agreement_verifier.py").is_file():
        raise ValueError(f"Not a DataCOPE general checkout: {datacope_root}")
    round_index = validate_round_index(args.round_index)
    query_ids = validate_prediction_runs(args.predictions_dir)
    require_empty_directory(args.verified_dir, "Verified output directory")
    previous_predictions: list[Path] = []
    previous_skill_dir: Path | None = None

    if args.mode == "verify":
        if round_index != 0 or args.previous_predictions_dir or args.previous_skill_dir:
            raise ValueError("verify checks one initial round; iterative inputs belong to discover")

    if args.mode == "discover":
        if (
            args.data_dir is None
            or args.skill_dir is None
            or args.family_manifest is None
            or args.codex_bin is None
            or not args.model
        ):
            raise ValueError(
                "discover requires --data-dir, --skill-dir, --family-manifest, "
                "--model, and --codex-bin"
            )
        expected_ids = set(load_dabench_split(args.family_manifest).explore_task_ids)
        if query_ids != expected_ids:
            raise ValueError(
                "Prediction task set must equal the frozen explore split; "
                f"missing={sorted(expected_ids - query_ids)[:5]}, "
                f"extra={sorted(query_ids - expected_ids)[:5]}"
            )
        validate_public_data(args.data_dir, query_ids)
        require_empty_directory(args.skill_dir, "Skill output directory")
        if round_index == 0:
            if args.previous_predictions_dir or args.previous_skill_dir is not None:
                raise ValueError("Discovery round 0 must not receive previous-round inputs")
        else:
            if len(args.previous_predictions_dir) != round_index or args.previous_skill_dir is None:
                raise ValueError(
                    f"Discovery round {round_index} requires {round_index} ordered previous "
                    "prediction directories and the latest previous skill"
                )
            previous_predictions = [
                path.expanduser().resolve() for path in args.previous_predictions_dir
            ]
            for previous_dir in previous_predictions:
                previous_ids = validate_prediction_runs(previous_dir)
                if previous_ids != query_ids:
                    raise ValueError("Previous and current prediction task sets must match")
            previous_skill_dir = args.previous_skill_dir.expanduser().resolve()
            if not (previous_skill_dir / "SKILL.md").is_file():
                raise ValueError(f"Previous skill is missing SKILL.md: {previous_skill_dir}")

    sys.path.insert(0, str(datacope_root))
    from src.verifier.instances.agreement_verifier import AgreementVerifier

    class ExactAgreementVerifier(AgreementVerifier):
        def _group_answers(self, predictions):  # noqa: ANN001
            return group_predictions_exactly(predictions)

    prediction_inputs = [
        *(str(path) for path in previous_predictions),
        str(args.predictions_dir.resolve()),
    ]
    verifier = ExactAgreementVerifier(
        prediction_inputs,
        str(args.verified_dir.resolve()),
        [],
    )
    verifier_result = verifier.iterate_run() if round_index else verifier.init_run()
    result = {
        "traj_dir": str(verifier_result["traj_dir"]),
        "round_index": str(round_index),
    }

    if args.mode == "verify":
        return result

    from src.skill_manager.skill_manager import SkillManager

    if previous_skill_dir is not None:
        shutil.copytree(previous_skill_dir, args.skill_dir.resolve(), dirs_exist_ok=True)

    manager = SkillManager(
        model=args.model,
        agent_type="codex",
        task="DABench",
        data_dir=str(args.data_dir.resolve()),
        current_skill_dir=str(args.skill_dir.resolve()),
        category_list=[],
        backend="openai",
        working_dir=str(datacope_root),
        codex_bin=str(args.codex_bin.resolve()),
        skill_output_dir=str(args.skill_dir.resolve()),
        skill_creator_dir=str((datacope_root / "skills" / "skill-creator").resolve()),
    )
    manager_method = (
        manager.create_skill_without_category
        if round_index == 0
        else manager.modify_skill_without_category
    )
    writer_prompt = "\n\n".join(
        (verifier_result["prompt"], skill_output_contract(args.skill_dir))
    )
    agent_result = manager_method(verifier_result["traj_dir"], writer_prompt)
    if getattr(agent_result, "error", None):
        raise RuntimeError(f"DataCOPE SkillManager failed: {agent_result.error}")

    skill_file = args.skill_dir.resolve() / "SKILL.md"
    if not skill_file.is_file() or not skill_file.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"DataCOPE did not create a non-empty {skill_file}")
    result["skill_file"] = str(skill_file)
    result["codex_version"] = args.codex_version or "unknown"
    result["codex_sha256"] = args.codex_sha256 or "unknown"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as exc:
        print(f"DataCOPE sidecar failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
