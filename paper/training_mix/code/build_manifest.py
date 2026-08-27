#!/usr/bin/env python3
"""Build a compact deterministic training mix with programmatic graders."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd

from experiments.dare_bench.runner import DareBenchTask, DareDataStore, score_prediction
from experiments.dare_bench.runner import load_tasks as load_dare_tasks
from experiments.data_agent_rl.runner import DataAgentTask
from experiments.data_agent_rl.runner import load_tasks as load_agent_tasks

TOTAL_TASKS = 2000
DARE_QUOTAS = {"classification": 275, "regression": 275, "time_series_v1": 25, "time_series_v2": 25}
AGENT_MODE_QUOTAS = {
    "numeric": 385,
    "exact_short": 240,
    "exact_bool": 40,
    "list_csv": 25,
    "list": 10,
}
DATAMIND_QUOTA = 700
MAX_DARE_INPUT_BYTES = 25 * 1024**2
MAX_DATAMIND_DATABASE_BYTES = 32 * 1024**2
MAX_DATAMIND_GOLD_BYTES = 1024**2
MAX_AGENT_TASKS_PER_DATASET = 12
AGENT_HEAVY_SHARE = 0.07


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def stable_key(seed: int, *values: object) -> str:
    text = "|".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_questions(tasks: Iterable[Any], seed: int) -> list[Any]:
    kept: dict[str, Any] = {}
    for task in tasks:
        question = normalize_question(task.question)
        current = kept.get(question)
        if current is None or stable_key(seed, task.task_id) < stable_key(seed, current.task_id):
            kept[question] = task
    return list(kept.values())


def select_hashed(tasks: Iterable[Any], count: int, seed: int, label: str) -> list[Any]:
    ordered = sorted(tasks, key=lambda task: stable_key(seed, label, task.task_id))
    if len(ordered) < count:
        raise ValueError(f"Not enough {label} tasks: requested {count}, found {len(ordered)}")
    return ordered[:count]


def dare_input_bytes(task: DareBenchTask, databases_dir: Path) -> int:
    return sum((databases_dir / member).stat().st_size for member in task.needed_file_members)


def dare_ground_truth_is_usable(
    task: DareBenchTask,
    databases_dir: Path,
    data_store: DareDataStore,
) -> bool:
    """Require the supplied ground truth to score itself perfectly."""

    ground_truth_path = databases_dir / task.ground_truth_relpath
    try:
        ground_truth = pd.read_csv(ground_truth_path)
        target_columns = list(task.targets)
        if not target_columns or any(column not in ground_truth for column in target_columns):
            return False
        if ground_truth.empty or ground_truth[target_columns].isna().any().any():
            return False
        target_set = set(target_columns)
        if task.task_type == "time_series_analysis" and task.version == "v2":
            id_columns = [column for column in ground_truth.columns if column not in target_set]
        elif "row_id" in ground_truth.columns:
            id_columns = ["row_id"]
        else:
            id_columns = [column for column in ground_truth.columns if column not in target_set]
        if (
            not id_columns
            or ground_truth[id_columns].isna().any().any()
            or ground_truth.duplicated(id_columns).any()
        ):
            return False
        grade = score_prediction(task, ground_truth_path, data_store)
    except Exception:
        return False
    return grade.get("final_score") == 1.0


def select_dare(tasks: list[DareBenchTask], seed: int, databases_dir: Path) -> list[DareBenchTask]:
    input_sizes = {task.task_id: dare_input_bytes(task, databases_dir) for task in tasks}
    data_store = DareDataStore(databases_dir=databases_dir)
    tasks = [
        task
        for task in tasks
        if input_sizes[task.task_id] <= MAX_DARE_INPUT_BYTES
        and dare_ground_truth_is_usable(task, databases_dir, data_store)
    ]
    selected: list[DareBenchTask] = []
    for task_type in ("classification", "regression"):
        pool = deduplicate_questions(
            (task for task in tasks if task.task_type == task_type and task.version == "v2"),
            seed,
        )
        selected.extend(select_hashed(pool, DARE_QUOTAS[task_type], seed, task_type))

    used_bases = {task.base_task_id for task in selected}
    used_questions = {normalize_question(task.question) for task in selected}
    graph = nx.DiGraph()
    source, sink = "source", "sink"
    version_quotas = {
        "v1": DARE_QUOTAS["time_series_v1"],
        "v2": DARE_QUOTAS["time_series_v2"],
    }
    task_by_edge: dict[tuple[tuple[str, str], tuple[str, str]], DareBenchTask] = {}
    for version, quota in version_quotas.items():
        graph.add_edge(source, ("version", version), capacity=quota)
    time_series = sorted(tasks, key=lambda task: stable_key(seed, "time_series", task.task_id))
    for task in time_series:
        question = normalize_question(task.question)
        if (
            task.task_type != "time_series_analysis"
            or task.base_task_id in used_bases
            or question in used_questions
        ):
            continue
        question_node = (task.version, question)
        base_node = ("base", task.base_task_id)
        graph.add_edge(("version", task.version), question_node, capacity=1)
        graph.add_edge(question_node, base_node, capacity=1)
        graph.add_edge(base_node, sink, capacity=1)
        task_by_edge[(question_node, base_node)] = task

    flow_value, flow = nx.maximum_flow(graph, source, sink)
    if flow_value != sum(version_quotas.values()):
        raise ValueError(f"Could only select {flow_value} unique time-series tasks")
    selected.extend(
        task
        for (question_node, base_node), task in task_by_edge.items()
        if flow[question_node][base_node] == 1
    )

    expected = sum(DARE_QUOTAS.values())
    if len(selected) != expected or len({task.base_task_id for task in selected}) != expected:
        raise ValueError(f"DARE selection must contain {expected} unique base tasks")
    return selected


def agent_is_heavy(task: DataAgentTask) -> bool:
    return bool(agent_runtime_heavy_packages(task))


def agent_runtime_heavy_packages(task: DataAgentTask) -> list[str]:
    return sorted(
        package
        for package in {value.casefold() for value in task.packages_used}
        if package in {"tensorflow", "keras"}
    )


def select_agent_segment(
    tasks: list[DataAgentTask],
    count: int,
    seed: int,
    label: str,
    dataset_counts: Counter[str],
) -> list[DataAgentTask]:
    by_dataset: dict[str, list[DataAgentTask]] = defaultdict(list)
    for task in tasks:
        by_dataset[task.bucket_prefix].append(task)
    for dataset, values in by_dataset.items():
        values.sort(key=lambda task: stable_key(seed, label, dataset, task.task_id))

    datasets = sorted(by_dataset, key=lambda dataset: stable_key(seed, label, dataset))
    offsets: Counter[str] = Counter()
    chosen: list[DataAgentTask] = []
    while len(chosen) < count:
        added = 0
        for dataset in datasets:
            index = offsets[dataset]
            values = by_dataset[dataset]
            if index >= len(values) or dataset_counts[dataset] >= MAX_AGENT_TASKS_PER_DATASET:
                continue
            chosen.append(values[index])
            offsets[dataset] += 1
            dataset_counts[dataset] += 1
            added += 1
            if len(chosen) == count:
                break
        if not added:
            raise ValueError(f"Dataset cap prevents filling Data Agent segment {label}")
    return chosen


def select_data_agent(tasks: list[DataAgentTask], seed: int) -> list[DataAgentTask]:
    pool = deduplicate_questions(tasks, seed)
    dataset_counts: Counter[str] = Counter()
    selected: list[DataAgentTask] = []
    for mode, quota in sorted(AGENT_MODE_QUOTAS.items(), key=lambda item: item[1]):
        mode_pool = [task for task in pool if task.reward_mode == mode]
        heavy_pool = [task for task in mode_pool if agent_is_heavy(task)]
        light_pool = [task for task in mode_pool if not agent_is_heavy(task)]
        heavy_count = min(round(quota * AGENT_HEAVY_SHARE), len(heavy_pool))
        light_count = quota - heavy_count
        if len(light_pool) < light_count:
            heavy_count += light_count - len(light_pool)
            light_count = len(light_pool)
        selected.extend(
            select_agent_segment(heavy_pool, heavy_count, seed, f"{mode}_heavy", dataset_counts)
        )
        selected.extend(
            select_agent_segment(light_pool, light_count, seed, f"{mode}_light", dataset_counts)
        )

    expected = sum(AGENT_MODE_QUOTAS.values())
    if len(selected) != expected:
        raise ValueError(f"Data Agent selection must contain {expected} tasks")
    return selected


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a dictionary")
    return parsed


def load_datamind_sql(dataset_root: Path, seed: int) -> tuple[list[dict[str, Any]], dict]:
    rl_root = dataset_root / "rl"
    frame = pd.read_parquet(rl_root / "train.parquet")
    frame = frame[frame["data_source"] == "darl/sql"]
    eligible: list[dict[str, Any]] = []
    excluded = Counter()

    for row in frame.to_dict(orient="records"):
        task_id = str(row.get("task_id") or "").strip()
        db_id = str(row.get("db_id") or "").strip()
        info = parse_mapping(row["extra_info"])
        question = str(info.get("question") or "").strip()
        database = rl_root / "train_files" / f"{db_id}.sqlite"
        gold = rl_root / "gold_csv_results" / f"{task_id}.csv"
        if not task_id or not db_id or not question:
            excluded["missing_metadata"] += 1
            continue
        if not database.is_file() or database.stat().st_size == 0:
            excluded["missing_database"] += 1
            continue
        if not gold.is_file() or gold.stat().st_size == 0:
            excluded["missing_gold"] += 1
            continue
        try:
            gold_sample = pd.read_csv(gold, nrows=1)
        except Exception:
            excluded["unreadable_gold"] += 1
            continue
        if gold_sample.empty or len(gold_sample.columns) == 0:
            excluded["empty_gold"] += 1
            continue
        eligible.append(
            {
                "task_id": task_id,
                "db_id": db_id,
                "question": question,
                "database_relpath": f"rl/train_files/{db_id}.sqlite",
                "gold_relpath": f"rl/gold_csv_results/{task_id}.csv",
                "database_bytes": database.stat().st_size,
                "gold_bytes": gold.stat().st_size,
            }
        )

    by_question: dict[str, dict[str, Any]] = {}
    for task in eligible:
        normalized = normalize_question(task["question"])
        current = by_question.get(normalized)
        if current is None or stable_key(seed, task["task_id"]) < stable_key(
            seed, current["task_id"]
        ):
            by_question[normalized] = task
    excluded["duplicate_question"] += len(eligible) - len(by_question)
    return list(by_question.values()), {
        "source_rows": len(frame),
        "eligible_rows": len(by_question),
        "excluded": dict(sorted(excluded.items())),
    }


def select_datamind(tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    tasks = [
        task
        for task in tasks
        if task["database_bytes"] <= MAX_DATAMIND_DATABASE_BYTES
        and task["gold_bytes"] <= MAX_DATAMIND_GOLD_BYTES
    ]
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_database[task["db_id"]].append(task)
    for db_id, values in by_database.items():
        values.sort(key=lambda task: stable_key(seed, db_id, task["task_id"]))

    database_order = sorted(by_database, key=lambda db_id: stable_key(seed, "db", db_id))
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < DATAMIND_QUOTA:
        added = 0
        for db_id in database_order:
            values = by_database[db_id]
            if round_index >= len(values):
                continue
            selected.append(values[round_index])
            added += 1
            if len(selected) == DATAMIND_QUOTA:
                break
        if not added:
            raise ValueError("Not enough eligible DataMind SQL tasks")
        round_index += 1
    return selected


def dare_record(task: DareBenchTask, databases_dir: Path) -> dict[str, Any]:
    return {
        "source": "dare_bench",
        "task_id": task.task_id,
        "question": task.question,
        "task_family": task.task_type,
        "dataset_key": task.base_task_id,
        "version": task.version,
        "cost_profile": {
            "input_bytes": dare_input_bytes(task, databases_dir),
            "heavy": False,
        },
        "input_ref": {
            "kind": "dare_archive_members",
            "members": list(task.needed_file_members),
        },
        "evaluator": {
            "kind": "artifact_metric",
            "metric": task.metric,
            "ground_truth_ref": task.ground_truth_relpath,
            "agent_visible": False,
        },
    }


def agent_record(task: DataAgentTask) -> dict[str, Any]:
    return {
        "source": "data_agent_rl",
        "task_id": task.task_id,
        "question": task.question,
        "task_family": task.reward_mode,
        "dataset_key": task.bucket_prefix,
        "difficulty_level": task.difficulty_level,
        "package_tier": task.package_tier,
        "cost_profile": {
            "heavy": agent_is_heavy(task),
            "runtime_heavy_packages": agent_runtime_heavy_packages(task),
            "files_used": len(task.files_used),
            "packages_used": len(task.packages_used),
        },
        "input_ref": {"kind": "hf_bucket_prefix", "prefix": task.bucket_prefix},
        "evaluator": {
            "kind": task.reward_mode,
            "ground_truth": task.gold_answer,
            "agent_visible": False,
        },
    }


def datamind_record(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "datamind_sql",
        "task_id": task["task_id"],
        "question": task["question"],
        "task_family": "sql",
        "dataset_key": task["db_id"],
        "cost_profile": {
            "database_bytes": task["database_bytes"],
            "ground_truth_bytes": task["gold_bytes"],
            "heavy": False,
        },
        "input_ref": {"kind": "sqlite", "ref": task["database_relpath"]},
        "evaluator": {
            "kind": "csv_table_match",
            "ignore_order": True,
            "ground_truth_ref": task["gold_relpath"],
            "agent_visible": False,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dare-manifest", type=Path, required=True)
    parser.add_argument("--dare-databases-dir", type=Path, required=True)
    parser.add_argument("--data-agent-root", type=Path, required=True)
    parser.add_argument("--datamind-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Selection audit path; defaults to <output-dir>/summary.json",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    dare_pool = load_dare_tasks(args.dare_manifest)
    agent_pool = load_agent_tasks(args.data_agent_root)
    datamind_pool, datamind_quality = load_datamind_sql(args.datamind_root, args.seed)

    dare_databases_dir = args.dare_databases_dir.expanduser().resolve()
    dare = select_dare(dare_pool, args.seed, dare_databases_dir)
    agent = select_data_agent(agent_pool, args.seed)
    datamind = select_datamind(datamind_pool, args.seed)
    records = [
        *(dare_record(task, dare_databases_dir) for task in dare),
        *(agent_record(task) for task in agent),
        *(datamind_record(task) for task in datamind),
    ]
    records.sort(key=lambda record: (record["source"], stable_key(args.seed, record["task_id"])))

    task_keys = {(record["source"], record["task_id"]) for record in records}
    questions = [normalize_question(record["question"]) for record in records]
    if (
        len(records) != TOTAL_TASKS
        or len(task_keys) != TOTAL_TASKS
        or len(set(questions)) != TOTAL_TASKS
    ):
        raise ValueError(
            f"Final mix must contain {TOTAL_TASKS:,} unique tasks and normalized questions"
        )
    if any(record["evaluator"].get("agent_visible") is not False for record in records):
        raise ValueError("Every ground truth must be evaluator-only")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"training_mix_{TOTAL_TASKS}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    selected_dare = {task.task_id for task in dare}
    dare_path = output_dir / f"dare_bench_{len(dare)}.jsonl"
    with dare_path.open("w", encoding="utf-8") as handle:
        for task in dare_pool:
            if task.task_id in selected_dare:
                handle.write(
                    json.dumps(task.upstream_record, ensure_ascii=False, sort_keys=True) + "\n"
                )
    write_json(
        output_dir / f"data_agent_rl_{len(agent)}_ids.json",
        [task.task_id for task in agent],
    )
    write_json(
        output_dir / f"datamind_sql_{len(datamind)}_ids.json",
        [task["task_id"] for task in datamind],
    )

    source_counts = Counter(record["source"] for record in records)
    summary = {
        "status": "passed",
        "seed": args.seed,
        "total": len(records),
        "counts_by_source": dict(sorted(source_counts.items())),
        "dare": {
            "candidate_tasks": len(dare_pool),
            "size_eligible_tasks": sum(
                dare_input_bytes(task, dare_databases_dir) <= MAX_DARE_INPUT_BYTES
                for task in dare_pool
            ),
            "selected": len(dare),
            "by_type_version": {
                f"{task_type}::{version}": count
                for (task_type, version), count in sorted(
                    Counter((task.task_type, task.version) for task in dare).items()
                )
            },
            "unique_base_tasks": len({task.base_task_id for task in dare}),
            "max_input_bytes": max(dare_input_bytes(task, dare_databases_dir) for task in dare),
            "input_size_limit_bytes": MAX_DARE_INPUT_BYTES,
        },
        "data_agent_rl": {
            "candidate_tasks": len(agent_pool),
            "selected": len(agent),
            "by_reward_mode": dict(sorted(Counter(task.reward_mode for task in agent).items())),
            "by_difficulty": {
                str(level): count
                for level, count in sorted(Counter(task.difficulty_level for task in agent).items())
            },
            "by_package_tier": {
                str(tier): count
                for tier, count in sorted(Counter(task.package_tier for task in agent).items())
            },
            "unique_datasets": len({task.bucket_prefix for task in agent}),
            "max_tasks_per_dataset": max(Counter(task.bucket_prefix for task in agent).values()),
            "runtime_heavy_tasks": sum(agent_is_heavy(task) for task in agent),
            "runtime_heavy_definition": "packages_used contains tensorflow or keras",
            "difficulty_used_as_cost_filter": False,
            "difficulty_5_tasks": sum(task.difficulty_level == 5 for task in agent),
        },
        "datamind_sql": {
            **datamind_quality,
            "size_eligible_rows": sum(
                task["database_bytes"] <= MAX_DATAMIND_DATABASE_BYTES
                and task["gold_bytes"] <= MAX_DATAMIND_GOLD_BYTES
                for task in datamind_pool
            ),
            "selected": len(datamind),
            "unique_databases": len({task["db_id"] for task in datamind}),
            "max_tasks_per_database": max(Counter(task["db_id"] for task in datamind).values()),
            "max_database_bytes": max(task["database_bytes"] for task in datamind),
            "database_size_limit_bytes": MAX_DATAMIND_DATABASE_BYTES,
            "max_ground_truth_bytes": max(task["gold_bytes"] for task in datamind),
            "ground_truth_size_limit_bytes": MAX_DATAMIND_GOLD_BYTES,
        },
        "quality": {
            "programmatic_evaluators_only": True,
            "ground_truth_agent_visible": False,
            "unique_source_task_keys": len(task_keys),
            "unique_normalized_questions": len(set(questions)),
            "cross_source_exact_question_duplicates": 0,
        },
    }
    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path
        else output_dir / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
