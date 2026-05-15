"""Release quality gate checks for BASIN."""

from __future__ import annotations

from typing import TypedDict


class Criterion(TypedDict):
    """A single quality gate criterion."""

    required: float | int | bool
    actual: float | int | bool
    passed: bool


def evaluate_quality_gate(
    *,
    sample_size: int,
    repeats_per_condition: int,
    classifier_accuracy: float,
    has_reproducibility_metadata: bool,
) -> dict[str, object]:
    """Evaluate release-readiness criteria and return pass/fail details."""
    criteria: dict[str, Criterion] = {
        "minimum_sample_size": {
            "required": 30,
            "actual": sample_size,
            "passed": sample_size >= 30,
        },
        "repeated_trials_required": {
            "required": 2,
            "actual": repeats_per_condition,
            "passed": repeats_per_condition >= 2,
        },
        "classifier_accuracy_floor": {
            "required": 0.7,
            "actual": classifier_accuracy,
            "passed": classifier_accuracy >= 0.7,
        },
        "reproducibility_metadata_complete": {
            "required": True,
            "actual": has_reproducibility_metadata,
            "passed": has_reproducibility_metadata,
        },
    }
    passed = all(v["passed"] for v in criteria.values())
    return {
        "passed": passed,
        "release_reporting_status": "passed" if passed else "failed",
        "criteria": criteria,
    }
