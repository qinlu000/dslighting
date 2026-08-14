from pathlib import Path

from dslighting.benchmark.vendor.mlebench.grade_helpers import Grader


def test_grader_defers_grade_fn_import_until_first_use(tmp_path: Path):
    marker_path = tmp_path / "grader_imported"
    grader_module = tmp_path / "custom_grader.py"
    grader_module.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker_path)!r}).touch()\n"
        "def grade(_submission, _answers):\n"
        "    return 1.0\n",
        encoding="utf-8",
    )

    grader = Grader(
        name="StandardGrader",
        grade_fn=f"file:{grader_module}:grade",
    )

    assert not marker_path.exists()

    first = grader.grade_fn
    second = grader.grade_fn

    assert marker_path.exists()
    assert first is second
