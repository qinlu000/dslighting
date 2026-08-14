"""Run the shared ReAct benchmark entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.benchmark._react_benchmark_common import run_react_benchmark  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        choices=("dabench", "dacode", "scienceagentbench", "moscibench"),
    )
    args = parser.parse_args(argv)
    return run_react_benchmark(args.benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
