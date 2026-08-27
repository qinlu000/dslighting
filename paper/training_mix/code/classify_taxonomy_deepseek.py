"""Classify training-mix prompts with a constrained, auditable taxonomy.

The classifier deliberately separates the primary analytical objective from
overlapping operations, reasoning complexity, domain knowledge, and output
format. Two independent DeepSeek passes are used by default; category
disagreements trigger a third adjudication call. DARE-Bench's official task
family remains the final source of truth for its primary modeling label.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from dslighting.utils.host_overrides import activate_api_host_overrides

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PAPER_ROOT / "artifacts" / "manifests" / "training_mix_2000.jsonl"
DEFAULT_OUTPUT = PAPER_ROOT / "artifacts" / "annotations" / "taxonomy_annotations.jsonl"
DEFAULT_TAXONOMY = PAPER_ROOT / "config" / "taxonomy_v1.yaml"
DEFAULT_DATAMIND_PARQUET = Path("/data/caoqinlu/datasets/DataMind-Data/rl/train.parquet")

SQL_PATTERN = re.compile(
    r"""execute_sql\s*\(\s*sql\s*=\s*(["']{1,3})(.*?)\1\s*,""",
    re.VERBOSE | re.DOTALL,
)
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class Taxonomy:
    raw: dict[str, Any]
    primary_ids: tuple[str, ...]
    operation_ids: tuple[str, ...]
    reasoning_ids: tuple[str, ...]
    domain_ids: tuple[str, ...]
    output_ids: tuple[str, ...]

    @property
    def version(self) -> str:
        return str(self.raw["version"])


def load_taxonomy(path: Path) -> Taxonomy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    taxonomy = Taxonomy(
        raw=raw,
        primary_ids=tuple(item["id"] for item in raw["primary_tasks"]),
        operation_ids=tuple(raw["operation_tags"]),
        reasoning_ids=tuple(raw["reasoning_complexity"]),
        domain_ids=tuple(raw["domain_knowledge"]),
        output_ids=tuple(raw["requested_outputs"]),
    )
    for name, values in (
        ("primary_tasks", taxonomy.primary_ids),
        ("operation_tags", taxonomy.operation_ids),
        ("reasoning_complexity", taxonomy.reasoning_ids),
        ("domain_knowledge", taxonomy.domain_ids),
        ("requested_outputs", taxonomy.output_ids),
    ):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Taxonomy axis {name!r} must contain unique values")
    return taxonomy


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: str) -> str:
    # Evidence must remain a verbatim span after normalizing whitespace and
    # Markdown code quoting. Models commonly omit backticks around column/file
    # names even when copying the surrounding prompt exactly.
    for marker in ("`", '"', "'", "“", "”", "‘", "’"):
        value = value.replace(marker, "")
    return SPACE_PATTERN.sub(" ", value).strip().casefold()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Response JSON must be an object")
    return value


def validate_decision(
    value: dict[str, Any],
    taxonomy: Taxonomy,
    evidence_text: str,
) -> dict[str, Any]:
    required = {
        "primary_task",
        "secondary_tasks",
        "operations",
        "reasoning_complexity",
        "domain_knowledge",
        "requested_outputs",
        "confidence",
        "evidence",
        "rationale",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    primary = str(value["primary_task"])
    if primary not in taxonomy.primary_ids:
        raise ValueError(f"Unknown primary_task: {primary}")

    def checked_list(field: str, allowed: tuple[str, ...], maximum: int) -> list[str]:
        items = value[field]
        if not isinstance(items, list):
            raise ValueError(f"{field} must be a list")
        normalized = list(dict.fromkeys(str(item) for item in items))
        unknown = sorted(set(normalized) - set(allowed))
        if unknown:
            raise ValueError(f"Unknown {field}: {unknown}")
        if len(normalized) > maximum:
            raise ValueError(f"{field} has more than {maximum} values")
        return normalized

    secondary = checked_list("secondary_tasks", taxonomy.primary_ids, 4)
    if primary in secondary or "unresolved" in secondary:
        raise ValueError("secondary_tasks cannot repeat primary_task or contain unresolved")
    operations = checked_list("operations", taxonomy.operation_ids, 12)
    outputs = checked_list("requested_outputs", taxonomy.output_ids, 4)
    if not outputs:
        raise ValueError("requested_outputs cannot be empty")

    reasoning = str(value["reasoning_complexity"])
    domain = str(value["domain_knowledge"])
    if reasoning not in taxonomy.reasoning_ids:
        raise ValueError(f"Unknown reasoning_complexity: {reasoning}")
    if domain not in taxonomy.domain_ids:
        raise ValueError(f"Unknown domain_knowledge: {domain}")

    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    evidence = value["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
        raise ValueError("evidence must contain 1 to 3 exact spans")
    normalized_source = normalize_text(evidence_text)
    clean_evidence = []
    for span in evidence:
        span = str(span).strip()
        if not span or normalize_text(span) not in normalized_source:
            raise ValueError(f"Evidence is not an exact input span: {span!r}")
        clean_evidence.append(span)

    rationale = str(value["rationale"]).strip()
    if not rationale:
        raise ValueError("rationale cannot be empty")

    return {
        "primary_task": primary,
        "secondary_tasks": secondary,
        "operations": operations,
        "reasoning_complexity": reasoning,
        "domain_knowledge": domain,
        "requested_outputs": outputs,
        "confidence": confidence,
        "evidence": clean_evidence,
        "rationale": rationale[:400],
    }


def render_codebook(taxonomy: Taxonomy, order_seed: str) -> str:
    tasks = list(taxonomy.raw["primary_tasks"])
    random.Random(order_seed).shuffle(tasks)
    lines = ["PRIMARY TASKS (choose exactly one):"]
    for item in tasks:
        lines.append(
            f"- {item['id']}: {item['definition']} INCLUDE: {item['include']} "
            f"EXCLUDE: {item['exclude']}"
        )
    lines.extend(
        [
            "",
            "PRIMARY PRECEDENCE:",
            *(f"- {rule}" for rule in taxonomy.raw["primary_precedence"]),
            "",
            "OPERATION TAGS (zero or more): " + ", ".join(taxonomy.operation_ids),
            "REASONING COMPLEXITY: " + ", ".join(taxonomy.reasoning_ids),
            "DOMAIN KNOWLEDGE: " + ", ".join(taxonomy.domain_ids),
            "REQUESTED OUTPUTS: " + ", ".join(taxonomy.output_ids),
        ]
    )
    return "\n".join(lines)


def system_prompt(taxonomy: Taxonomy, order_seed: str) -> str:
    return f"""You are annotating data-agent tasks with taxonomy version {taxonomy.version}.
Classify the requested final analytical objective, not the implementation language,
benchmark source, answer type, or evaluator. Intermediate operations do not determine
the primary task. Use secondary_tasks only for additional objectives explicitly required
by the prompt. Use unresolved when evidence is genuinely insufficient.

Do not provide chain-of-thought. Return only one JSON object with exactly these fields:
{{
  "primary_task": "one allowed primary id",
  "secondary_tasks": ["0-4 other primary ids"],
  "operations": ["allowed operation ids"],
  "reasoning_complexity": "single_step or multi_step",
  "domain_knowledge": "none, helpful, or required",
  "requested_outputs": ["allowed output ids"],
  "confidence": 0.0,
  "evidence": ["1-3 short exact spans copied from PROMPT or REFERENCE SQL"],
  "rationale": "one short decision-rule explanation, not hidden reasoning"
}}

Before returning, verify all of the following:
- evidence contains 1-3 continuous, exact copied spans; never paraphrase and never use "...";
- every value comes from the correct axis (task, operation, or output) listed below;
- the JSON contains no comments, formulas, ranges, trailing commas, or Markdown fences;
- rationale is a single concise sentence.

{render_codebook(taxonomy, order_seed)}"""


def task_evidence_text(task: dict[str, Any], reference_sql: str | None) -> str:
    question = str(task.get("question") or "").strip()
    if reference_sql:
        return f"PROMPT:\n{question}\n\nREFERENCE SQL:\n{reference_sql.strip()}"
    return f"PROMPT:\n{question}"


def official_primary(task: dict[str, Any], taxonomy: Taxonomy) -> str | None:
    mapping = taxonomy.raw.get("official_mappings", {}).get("dare_bench", {})
    if task.get("source") == "dare_bench":
        return mapping.get(task.get("task_family"))
    return None


def extract_reference_sql(trajectory: Any) -> str | None:
    texts = [
        str(message.get("content", ""))
        for message in trajectory
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    matches = SQL_PATTERN.findall("\n".join(texts))
    if not matches:
        return None
    try:
        return codecs.decode(matches[-1][1], "unicode_escape").strip()
    except UnicodeDecodeError:
        return matches[-1][1].strip()


def load_datamind_sql(path: Path, task_ids: set[str]) -> dict[str, str]:
    if not path.exists() or not task_ids:
        return {}
    frame = pd.read_parquet(path, columns=["task_id", "trajectory"])
    frame = frame[frame["task_id"].isin(task_ids)]
    result = {}
    for row in frame.itertuples(index=False):
        sql = extract_reference_sql(row.trajectory)
        if sql:
            result[str(row.task_id)] = sql
    return result


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def needs_adjudication(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first["primary_task"] != second["primary_task"] or set(first["secondary_tasks"]) != set(
        second["secondary_tasks"]
    )


def merge_matching_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    winner = first if first["confidence"] >= second["confidence"] else second
    merged = dict(winner)
    merged["operations"] = sorted(set(first["operations"]) | set(second["operations"]))
    merged["requested_outputs"] = sorted(
        set(first["requested_outputs"]) | set(second["requested_outputs"])
    )
    merged["confidence"] = min(first["confidence"], second["confidence"])
    return merged


def adjudication_prompt(
    taxonomy: Taxonomy,
    evidence_text: str,
    first: dict[str, Any],
    second: dict[str, Any],
    task_id: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(taxonomy, f"{task_id}:adjudicate")},
        {
            "role": "user",
            "content": (
                f"{evidence_text}\n\nTwo independent annotations disagree. Adjudicate using the "
                "taxonomy definitions, not majority voting. Return a complete corrected JSON object.\n\n"
                f"ANNOTATION A:\n{json.dumps(first, ensure_ascii=False)}\n\n"
                f"ANNOTATION B:\n{json.dumps(second, ensure_ascii=False)}"
            ),
        },
    ]


async def request_decision(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    taxonomy: Taxonomy,
    evidence_text: str,
    retries: int,
) -> dict[str, Any]:
    error: Exception | None = None
    attempt_messages = list(messages)
    for attempt in range(retries):
        content = ""
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=attempt_messages,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if not response.choices:
                detail = getattr(response, "error", None) or getattr(response, "msg", None)
                raise RuntimeError(f"Provider returned no completion choices: {detail}")
            content = response.choices[0].message.content or ""
            return validate_decision(parse_json_object(content), taxonomy, evidence_text)
        except Exception as exc:  # retry transport errors and invalid structured output
            error = exc
            if attempt + 1 < retries:
                if content:
                    attempt_messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    f"Validation error: {exc}. Correct only the JSON format or "
                                    "taxonomy fields. Evidence must be copied exactly and "
                                    "continuously from the supplied input. Return the complete "
                                    "corrected JSON object only."
                                ),
                            },
                        ]
                    )
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"DeepSeek annotation failed after {retries} attempts: {error}") from error


async def classify_one(
    client: AsyncOpenAI,
    task: dict[str, Any],
    taxonomy: Taxonomy,
    reference_sql: str | None,
    model: str,
    passes: int,
    retries: int,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    evidence_text = task_evidence_text(task, reference_sql)
    runs = []
    for pass_index in range(passes):
        messages = [
            {
                "role": "system",
                "content": system_prompt(taxonomy, f"{task_id}:pass:{pass_index}"),
            },
            {"role": "user", "content": evidence_text},
        ]
        runs.append(
            await request_decision(
                client,
                model=model,
                messages=messages,
                taxonomy=taxonomy,
                evidence_text=evidence_text,
                retries=retries,
            )
        )

    adjudicated = False
    if len(runs) == 1:
        final = dict(runs[0])
    elif needs_adjudication(runs[0], runs[1]):
        final = await request_decision(
            client,
            model=model,
            messages=adjudication_prompt(taxonomy, evidence_text, runs[0], runs[1], task_id),
            taxonomy=taxonomy,
            evidence_text=evidence_text,
            retries=retries,
        )
        adjudicated = True
    else:
        final = merge_matching_runs(runs[0], runs[1])

    official = official_primary(task, taxonomy)
    model_primary = final["primary_task"]
    if official:
        final = dict(final)
        final["primary_task"] = official
        final["secondary_tasks"] = [
            label for label in final["secondary_tasks"] if label != official
        ]

    agreement = None
    if len(runs) > 1:
        agreement = {
            "primary_match": runs[0]["primary_task"] == runs[1]["primary_task"],
            "secondary_jaccard": jaccard(runs[0]["secondary_tasks"], runs[1]["secondary_tasks"]),
            "operations_jaccard": jaccard(runs[0]["operations"], runs[1]["operations"]),
        }

    return {
        "task_id": task_id,
        "source": task.get("source"),
        "taxonomy_version": taxonomy.version,
        "question_sha256": hashlib.sha256(
            str(task.get("question") or "").encode("utf-8")
        ).hexdigest(),
        "reference_sql_used": bool(reference_sql),
        "official_primary_task": official,
        "model_primary_before_official_override": model_primary if official else None,
        "annotator_runs": runs,
        "agreement": agreement,
        "adjudicated": adjudicated,
        "final": final,
        "needs_review": final["confidence"] < 0.75 or final["primary_task"] == "unresolved",
    }


def compact_checkpoint(path: Path, valid_task_ids: set[str] | None = None) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = row.get("task_id")
            if task_id is not None and (
                valid_task_ids is None or str(task_id) in valid_task_ids
            ):
                latest[str(task_id)] = row
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in latest.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return {task_id for task_id, row in latest.items() if row.get("status") == "ok"}


async def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    activate_api_host_overrides()
    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = args.api_base or os.getenv("API_BASE")
    if not api_key or not api_base:
        raise SystemExit("API_KEY and API_BASE must be supplied by arguments, environment, or .env")

    taxonomy = load_taxonomy(args.taxonomy)
    tasks = load_jsonl(args.input)
    if args.source:
        allowed_sources = set(args.source)
        tasks = [task for task in tasks if task.get("source") in allowed_sources]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    current_task_ids = {str(task["task_id"]) for task in tasks}
    done = set() if args.overwrite else compact_checkpoint(args.output, current_task_ids)
    tasks = [task for task in tasks if str(task["task_id"]) not in done]

    datamind_ids = {str(task["task_id"]) for task in tasks if task.get("source") == "datamind_sql"}
    sql_by_task = (
        load_datamind_sql(args.datamind_parquet, datamind_ids) if args.include_reference_sql else {}
    )
    if args.include_reference_sql and datamind_ids - sql_by_task.keys():
        missing = len(datamind_ids - sql_by_task.keys())
        raise SystemExit(f"Reference SQL extraction failed for {missing} selected DataMind tasks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)

    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=args.timeout, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counts = {"ok": 0, "error": 0}

    async def worker(task: dict[str, Any]) -> None:
        async with semaphore:
            try:
                result = await classify_one(
                    client,
                    task,
                    taxonomy,
                    sql_by_task.get(str(task["task_id"])),
                    args.model,
                    args.passes,
                    args.retries,
                )
                row = {"status": "ok", **result}
                counts["ok"] += 1
            except Exception as exc:
                row = {
                    "status": "error",
                    "task_id": str(task["task_id"]),
                    "source": task.get("source"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                counts["error"] += 1
            async with write_lock:
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                finished = counts["ok"] + counts["error"]
                if finished % args.progress_every == 0 or finished == len(tasks):
                    print(
                        f"completed={finished}/{len(tasks)} ok={counts['ok']} "
                        f"error={counts['error']}",
                        flush=True,
                    )

    await asyncio.gather(*(worker(task) for task in tasks))
    await client.close()
    compact_checkpoint(args.output, current_task_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--datamind-parquet", type=Path, default=DEFAULT_DATAMIND_PARQUET)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--source", action="append", help="Only classify this source; repeatable")
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--include-reference-sql",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also show DataMind reference SQL to the annotator (prompt-only by default)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.progress_every < 1:
        raise SystemExit("concurrency, retries, and progress-every must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
