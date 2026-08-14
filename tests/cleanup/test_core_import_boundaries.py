from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_core_import_does_not_pull_runner_or_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import dslighting.core\n"
                "assert 'dslighting.runner' not in sys.modules\n"
                "assert 'dslighting.runtime' not in sys.modules\n"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
