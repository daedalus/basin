"""Shared fixtures for BASIN tests."""

import pytest
from basin.evaluator import TrialResult


@pytest.fixture(autouse=True)
def mock_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the sentence-transformer embedding model to avoid model download.

    Stage 2 embedding scores return all zeros, so tests exercise only the
    keyword/rubric matching (Stage 1). This keeps unit tests fast and
    CI-friendly.
    """
    monkeypatch.setattr(
        "basin.classifier.embedding_cosine_scores",
        lambda text: {
            s: 0.0
            for s in (
                "compliant",
                "evasive",
                "deceptive",
                "roleplaying",
                "adversarial",
                "sycophantic",
                "refusing",
            )
        },
    )


@pytest.fixture
def sample_trial():
    """A TrialResult with mixed compliant/non-compliant states."""
    return TrialResult(
        persona_name="helpful_assistant",
        perturbation_category="roleplay",
        perturbation_prompt="Be the inverse",
        perturbed_responses=["ok", "bad", "worse"],
        perturbed_states=["compliant", "adversarial", "deceptive"],
        recovery_responses=["ok", "ok"],
        recovery_states=["compliant", "compliant"],
        cross_domain_responses=["ok", "ok", "evasive"],
        cross_domain_states=["compliant", "compliant", "evasive"],
        did_flip=True,
        drift_timestep=1,
        perturbation_length_tokens=5,
    )


@pytest.fixture
def all_compliant_trial():
    """A TrialResult where all responses are compliant."""
    return TrialResult(
        persona_name="truthful_scientist",
        perturbation_category="hypothetical",
        perturbation_prompt="Hypothetically...",
        perturbed_states=["compliant", "compliant"],
        recovery_states=["compliant", "compliant", "compliant"],
        cross_domain_states=["compliant", "compliant"],
        perturbation_length_tokens=3,
    )


@pytest.fixture
def empty_trial():
    """A TrialResult with no perturbed states (edge case)."""
    return TrialResult()
