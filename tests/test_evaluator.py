"""Tests for the evaluator scoring module."""

import math
import pytest
from basin.evaluator import (
    TrialResult,
    BenchmarkScore,
    score_trial,
    aggregate_scores,
)


class TestScoreTrial:
    """Tests for score_trial function."""

    def test_mixed_trial_scores(self, sample_trial):
        s = score_trial(sample_trial)
        assert 0.0 <= s.persona_stability <= 1.0
        assert 0.0 <= s.inverse_accessibility <= 1.0
        assert 0.0 <= s.hysteresis <= 1.0
        assert 0.0 <= s.cross_domain_transfer <= 1.0
        assert 0.0 <= s.compression_ratio <= 1.0

    def test_all_compliant(self, all_compliant_trial):
        s = score_trial(all_compliant_trial)
        assert s.persona_stability == pytest.approx(1.0)
        assert s.inverse_accessibility == pytest.approx(0.0)
        assert s.hysteresis == pytest.approx(0.0)
        assert s.cross_domain_transfer == pytest.approx(0.0)
        assert s.recovery_half_life == 2.0

    def test_no_perturbed_states_raises(self, empty_trial):
        with pytest.raises(ValueError):
            score_trial(empty_trial)

    def test_radar_profile_keys(self, sample_trial):
        s = score_trial(sample_trial)
        profile = s.radar_profile()
        expected_keys = {
            "persona_stability",
            "inverse_accessibility",
            "hysteresis",
            "cross_domain_transfer",
            "internal_shift",
            "compression_ratio",
        }
        assert set(profile.keys()) == expected_keys

    def test_flipped_trial_penalty(self):
        t = TrialResult(
            perturbed_states=["adversarial", "adversarial"],
            recovery_states=[],
            cross_domain_states=[],
            did_flip=True,
            drift_timestep=0,
            perturbation_length_tokens=10,
        )
        s = score_trial(t)
        assert s.persona_stability == 0.0
        assert s.inverse_accessibility > 0.5


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_empty_list(self):
        scores = aggregate_scores([])
        assert all(v == 0.0 for k, v in scores.items() if k != "recovery_half_life")
        assert scores["recovery_half_life"] == float("inf")

    def test_single_trial(self, sample_trial):
        scores = aggregate_scores([sample_trial])
        assert "persona_stability" in scores
        assert "recovery_half_life" in scores

    def test_multiple_trials(self, sample_trial, all_compliant_trial):
        scores = aggregate_scores([sample_trial, all_compliant_trial])
        assert 0.0 < scores["persona_stability"] < 1.0
        assert scores["cross_domain_transfer"] >= 0.0

    def test_radar_axis_count(self, sample_trial):
        scores = aggregate_scores([sample_trial])
        assert len(scores) == 7  # 6 axes + recovery_half_life

    def test_infinite_recovery_hl(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=[],
            cross_domain_states=[],
        )
        scores = aggregate_scores([t])
        assert scores["recovery_half_life"] == float("inf")

    def test_benchmark_score_dataclass(self):
        bs = BenchmarkScore()
        assert bs.persona_stability == 0.0
        assert bs.recovery_half_life == float("inf")
