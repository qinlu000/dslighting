import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch


class InvalidSubmissionError(Exception):
    pass


def _load_grade_helpers():
    """Load the helper with lightweight stubs without leaking them globally."""

    mock_utils = Mock()
    mock_utils.get_logger = Mock(return_value=Mock())
    mock_utils.import_fn = Mock()

    fake_errors = types.ModuleType("dslighting.benchmark.grading.errors")
    fake_errors.InvalidSubmissionError = InvalidSubmissionError
    fake_reporting_models = types.ModuleType("dslighting.benchmark.reporting.models")
    fake_reporting_models.CompetitionReport = object

    spec = importlib.util.spec_from_file_location(
        "lazy_grade_helpers_under_test",
        Path(__file__).parent.parent.parent
        / "dslighting"
        / "benchmark"
        / "vendor"
        / "mlebench"
        / "grade_helpers.py",
    )
    grade_helpers = importlib.util.module_from_spec(spec)
    stubs = {
        "pandas": Mock(),
        "dslighting": types.ModuleType("dslighting"),
        "dslighting.benchmark": types.ModuleType("dslighting.benchmark"),
        "dslighting.benchmark.grading": types.ModuleType(
            "dslighting.benchmark.grading"
        ),
        "dslighting.benchmark.grading.errors": fake_errors,
        "dslighting.benchmark.reporting": types.ModuleType(
            "dslighting.benchmark.reporting"
        ),
        "dslighting.benchmark.reporting.models": fake_reporting_models,
        "dslighting.benchmark.vendor.mlebench.utils": mock_utils,
        "lazy_grade_helpers_under_test": grade_helpers,
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(grade_helpers)
    return grade_helpers


grade_helpers = _load_grade_helpers()

Grader = grade_helpers.Grader


def test_grader_defers_grade_fn_import_until_first_use():
    calls: list[str] = []

    def fake_import_fn(ref: str):
        calls.append(ref)

        def _grade(_submission, _answers):
            return 1.0

        return _grade

    grade_helpers.import_fn = fake_import_fn

    grader = Grader(name="StandardGrader", grade_fn="file:/tmp/grade.py:grade")

    assert calls == []

    first = grader.grade_fn
    second = grader.grade_fn

    assert calls == ["file:/tmp/grade.py:grade"]
    assert first is second
