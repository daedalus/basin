"""Tests for robustness diagnostics."""

from basin_benchmark.evaluator import TrialResult
from basin_benchmark.robustness import run_robustness_checks


class TestRunRobustnessChecks:
    """Tests for robustness summary output."""

    def test_empty_trials(self):
        out = run_robustness_checks([])
        assert out["threshold_sensitivity"] == {}
        assert out["recovery_length_sensitivity"] == {}
        assert out["turn_order_stability_proxy"]["absolute_delta"] == 0.0

    def test_nonempty_trials(self):
        trials = [
            TrialResult(
                perturbed_states=["compliant", "adversarial"],
                perturbed_scores=[
                    {"compliant": 0.8, "adversarial": 0.2},
                    {"compliant": 0.1, "adversarial": 0.9},
                ],
                recovery_states=["adversarial", "compliant"],
            ),
            TrialResult(
                perturbed_states=["adversarial", "adversarial"],
                perturbed_scores=[
                    {"compliant": 0.2, "adversarial": 0.8},
                    {"compliant": 0.1, "adversarial": 0.9},
                ],
                recovery_states=["compliant", "compliant"],
            ),
        ]
        out = run_robustness_checks(trials)
        assert "0.3" in out["threshold_sensitivity"]
        assert "1" in out["recovery_length_sensitivity"]
        proxy = out["turn_order_stability_proxy"]
        assert 0.0 <= proxy["early_noncompliance_rate"] <= 1.0
        assert 0.0 <= proxy["late_noncompliance_rate"] <= 1.0
