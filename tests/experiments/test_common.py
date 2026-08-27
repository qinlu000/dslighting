from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from experiments.common import prepare_task_output, run_batch, write_json


def test_prepare_task_output_reuses_completed_and_replaces_failed(tmp_path: Path) -> None:
    output = tmp_path / "task"
    write_json(output / "result.json", {"status": "completed", "reward": 1.0})

    assert prepare_task_output(
        output,
        overwrite=False,
        retry_failed=True,
    ) == {"status": "completed", "reward": 1.0}

    write_json(output / "result.json", {"status": "failed"})
    assert prepare_task_output(output, overwrite=False, retry_failed=True) is None
    assert output.is_dir() and not any(output.iterdir())


@pytest.mark.asyncio
async def test_run_batch_preserves_order_and_bounds_concurrency() -> None:
    active = 0
    maximum = 0

    async def run_one(value: int) -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01 * (4 - value))
            return value
        finally:
            active -= 1

    results = await run_batch([1, 2, 3], run_one, concurrency=2)

    assert results == [1, 2, 3]
    assert maximum == 2
