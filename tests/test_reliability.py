"""Tests for classifier reliability utilities."""

import json

from basin_benchmark.evaluator import TrialResult
from basin_benchmark.reliability import (
    evaluate_classifier_reliability,
    load_labeled_validation_examples,
    sample_human_adjudication_candidates,
)


class TestLoadLabeledValidationExamples:
    """Tests for loading labeled validation datasets."""

    def test_missing_file_returns_empty(self):
        examples = load_labeled_validation_examples("/tmp/does-not-exist-basin.json")
        assert examples == []

    def test_loads_valid_examples(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text(
            json.dumps(
                [
                    {"text": "hello", "label": "compliant"},
                    {"text": 123, "label": "compliant"},
                    {"text": "x", "label": None},
                ]
            )
        )
        examples = load_labeled_validation_examples(str(p))
        assert examples == [{"text": "hello", "label": "compliant"}]


class TestEvaluateClassifierReliability:
    """Tests for reliability metrics from labeled examples."""

    def test_empty_examples(self):
        out = evaluate_classifier_reliability([])
        assert out["sample_size"] == 0
        assert out["accuracy"] == 0.0
        assert out["per_class"] == {}

    def test_metrics_and_confusion(self, monkeypatch):
        def fake_classify(text: str):
            if "good" in text:
                return "compliant", {}
            if "refuse" in text:
                return "refusing", {}
            return "adversarial", {}

        monkeypatch.setattr("basin_benchmark.reliability.classify_text", fake_classify)
        examples = [
            {"text": "good answer", "label": "compliant"},
            {"text": "please refuse", "label": "refusing"},
            {"text": "bad actor", "label": "adversarial"},
            {"text": "good but mislabeled", "label": "refusing"},
        ]
        out = evaluate_classifier_reliability(examples)
        assert out["sample_size"] == 4
        assert 0.0 < out["accuracy"] < 1.0
        assert "compliant" in out["per_class"]
        assert "confusion_matrix" in out


class TestSampleHumanAdjudicationCandidates:
    """Tests for adjudication candidate sampling."""

    def test_zero_sample_returns_empty(self):
        out = sample_human_adjudication_candidates([], 0, 42)
        assert out == []

    def test_samples_deterministically(self):
        trials = [
            TrialResult(
                persona_name="p1",
                perturbation_category="c1",
                baseline_response="baseline",
                perturbed_responses=["p1-a", "p1-b"],
                recovery_responses=["r1"],
                cross_domain_responses=["x1"],
                prompt_family="dev",
            ),
            TrialResult(
                persona_name="p2",
                perturbation_category="c2",
                baseline_response="baseline2",
                perturbed_responses=["p2-a"],
                recovery_responses=["r2"],
                cross_domain_responses=["x2"],
                prompt_family="heldout",
            ),
        ]
        out1 = sample_human_adjudication_candidates(trials, 3, 99)
        out2 = sample_human_adjudication_candidates(trials, 3, 99)
        assert out1 == out2
        assert len(out1) == 3
        assert all("snippet" in x for x in out1)
