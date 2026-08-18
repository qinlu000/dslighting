"""Score collected episodes with DataMind's original Python-task Judge."""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from dotenv import load_dotenv

DEFAULT_JUDGE_FILE = Path(
    "/data/caoqinlu/projects/DataMind/datamind/train/rl/verl/verl/utils/"
    "reward_score/model_judge.py"
)


def load_judge(path: Path) -> ModuleType:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DataMind Judge module does not exist: {path}")
    spec = importlib.util.spec_from_file_location("datamind_model_judge", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load DataMind Judge module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_episode(path: Path, *, judge: ModuleType, model: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    episode = json.loads(path.read_text(encoding="utf-8"))
    if episode.get("status") != "completed":
        raise ValueError(f"Cannot judge incomplete episode: {path}")
    answer = episode.get("answer")
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)
    score, reason = judge.judge_with_retry(
        answer,
        str(episode.get("ground_truth") or ""),
        str(episode.get("question") or ""),
        model_name=model,
        max_retries=3,
    )
    episode["judge"] = {
        "model": model,
        "score": float(score),
        "reason": str(reason),
        "rubric": "datamind_original_python_v1",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1024,
    }
    path.write_text(
        json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"episode": str(path), "task_id": episode.get("task_id"), "score": float(score)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--judge-file", type=Path, default=DEFAULT_JUDGE_FILE)
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    args = _parser().parse_args(argv)
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    judge = load_judge(args.judge_file)
    results = []
    skipped = 0
    candidates = []
    for path in sorted(args.episode_root.expanduser().resolve().glob("*/episode.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        if episode.get("judge") is not None and not args.overwrite:
            skipped += 1
            continue
        if episode.get("status") != "completed":
            skipped += 1
            continue
        candidates.append(path)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(score_episode, path, judge=judge, model=args.model): path
            for path in candidates
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"judged task={result['task_id']} score={result['score']}")
    results.sort(key=lambda result: str(result.get("task_id") or ""))
    summary = {
        "model": args.model,
        "rubric": "datamind_original_python_v1",
        "concurrency": args.concurrency,
        "judged": len(results),
        "skipped": skipped,
        "mean_score": (
            sum(result["score"] for result in results) / len(results) if results else None
        ),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
