"""Bridge from exported MoSciBench trajectories to DataCOPE."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .adapter import validate_public_data
    from .protocol import PAPER_TRAJECTORIES_PER_TASK, validate_round_index
    from .split import load_split
else:
    from adapter import validate_public_data
    from protocol import PAPER_TRAJECTORIES_PER_TASK, validate_round_index
    from split import load_split


def group_predictions_exactly(predictions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    answers: list[Any] = []
    for prediction in predictions:
        answer = prediction["answer"]
        for index, existing in enumerate(answers):
            if answer == existing:
                groups[index].append(prediction)
                break
        else:
            answers.append(answer)
            groups.append([prediction])
    return sorted(groups, key=len, reverse=True)


def validate_prediction_runs(predictions_dir: Path) -> set[str]:
    root = Path(predictions_dir).expanduser().resolve()
    run_dirs = (
        sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and list(path.glob("prediction_*.json"))
        )
        if root.is_dir()
        else []
    )
    if len(run_dirs) != PAPER_TRAJECTORIES_PER_TASK:
        raise ValueError(
            f"Expected {PAPER_TRAJECTORIES_PER_TASK} sample directories, found {len(run_dirs)}"
        )
    query_sets: list[set[str]] = []
    for run_dir in run_dirs:
        query_ids: list[str] = []
        for path in sorted(run_dir.glob("prediction_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("prediction"), str):
                raise ValueError(f"Missing normalized prediction in {path}")
            query_ids.append(str(payload["extra_info"]["query_id"]))
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Duplicate query_id in {run_dir}")
        query_sets.append(set(query_ids))
    expected = query_sets[0]
    if any(query_ids != expected for query_ids in query_sets[1:]):
        raise ValueError("MoSciBench sample runs contain different task sets")
    return expected


def _empty(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{label} must be empty or new: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datacope-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--previous-predictions-dir", type=Path, action="append", default=[])
    parser.add_argument("--previous-skill-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--codex-bin", type=Path, required=True)
    parser.add_argument("--codex-version")
    parser.add_argument("--codex-sha256")
    return parser


def _run(args: argparse.Namespace) -> dict[str, str]:
    datacope_root = args.datacope_root.expanduser().resolve()
    round_index = validate_round_index(args.round_index)
    query_ids = validate_prediction_runs(args.predictions_dir)
    split = load_split(args.split_manifest)
    if query_ids != split.query_ids("explore"):
        raise ValueError("Prediction task set must equal the frozen explore split")
    _empty(args.verified_dir, "Verified output directory")

    validate_public_data(args.data_dir, query_ids)
    _empty(args.skill_dir, "Skill output directory")
    if round_index == 0:
        if args.previous_predictions_dir or args.previous_skill_dir:
            raise ValueError("Discovery round 0 must not receive previous inputs")
        previous_predictions: list[Path] = []
        previous_skill: Path | None = None
    else:
        if len(args.previous_predictions_dir) != round_index or args.previous_skill_dir is None:
            raise ValueError(
                f"Discovery round {round_index} requires all earlier predictions and skill"
            )
        previous_predictions = [
            path.expanduser().resolve() for path in args.previous_predictions_dir
        ]
        if any(validate_prediction_runs(path) != query_ids for path in previous_predictions):
            raise ValueError("Previous prediction task sets differ from the current round")
        previous_skill = args.previous_skill_dir.expanduser().resolve()
        if not (previous_skill / "SKILL.md").is_file():
            raise ValueError(f"Previous skill is missing SKILL.md: {previous_skill}")

    sys.path.insert(0, str(datacope_root))
    from src.verifier.instances.agreement_verifier import AgreementVerifier

    class ExactAgreementVerifier(AgreementVerifier):
        def _group_answers(self, predictions):  # noqa: ANN001
            return group_predictions_exactly(predictions)

    verifier = ExactAgreementVerifier(
        [*(str(path) for path in previous_predictions), str(args.predictions_dir.resolve())],
        str(args.verified_dir.resolve()),
        [],
    )
    verifier_result = verifier.iterate_run() if round_index else verifier.init_run()
    from src.skill_manager.skill_manager import SkillManager

    if previous_skill is not None:
        shutil.copytree(previous_skill, args.skill_dir.resolve(), dirs_exist_ok=True)
    manager = SkillManager(
        model=args.model,
        agent_type="codex",
        task="MoSciBench",
        data_dir=str(args.data_dir.resolve()),
        current_skill_dir=str(args.skill_dir.resolve()),
        category_list=[],
        backend="openai",
        working_dir=str(datacope_root),
        codex_bin=str(args.codex_bin.resolve()),
        skill_output_dir=str(args.skill_dir.resolve()),
        skill_creator_dir=str((datacope_root / "skills" / "skill-creator").resolve()),
    )
    method = (
        manager.create_skill_without_category
        if round_index == 0
        else manager.modify_skill_without_category
    )
    skill_file = args.skill_dir.resolve() / "SKILL.md"
    writer_prompt = "\n\n".join(
        (
            verifier_result["prompt"],
            f"The final MoSciBench Skill file must be saved at exactly `{skill_file}`.",
        )
    )
    agent_result = method(verifier_result["traj_dir"], writer_prompt)
    if getattr(agent_result, "error", None):
        raise RuntimeError(f"DataCOPE SkillManager failed: {agent_result.error}")
    if not skill_file.is_file() or not skill_file.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"DataCOPE did not create {skill_file}")
    return {
        "traj_dir": str(verifier_result["traj_dir"]),
        "round_index": str(round_index),
        "skill_file": str(skill_file),
        "codex_version": args.codex_version or "unknown",
        "codex_sha256": args.codex_sha256 or "unknown",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as exc:
        print(f"DataCOPE MoSciBench bridge failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
