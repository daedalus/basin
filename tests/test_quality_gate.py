"""Tests for release quality gate checks."""

from basin_benchmark.quality_gate import evaluate_quality_gate


class TestEvaluateQualityGate:
    """Tests for quality gate pass/fail logic."""

    def test_passes_with_all_criteria(self):
        out = evaluate_quality_gate(
            sample_size=100,
            repeats_per_condition=3,
            classifier_accuracy=0.9,
            has_reproducibility_metadata=True,
        )
        assert out["passed"] is True
        assert out["release_reporting_status"] == "passed"
        assert all(c["passed"] for c in out["criteria"].values())

    def test_fails_when_criteria_unmet(self):
        out = evaluate_quality_gate(
            sample_size=10,
            repeats_per_condition=1,
            classifier_accuracy=0.5,
            has_reproducibility_metadata=False,
        )
        assert out["passed"] is False
        assert out["release_reporting_status"] == "failed"
        assert any(not c["passed"] for c in out["criteria"].values())
