"""Adversarial and edge-case tests for BASIN."""

import math

import pytest

from basin_benchmark.classifier import (
    BEHAVIORAL_STATES,
    classify_text,
    lexical_diversity,
    prompt_complexity,
)
from basin_benchmark.evaluator import (
    BenchmarkScore,
    TrialResult,
    aggregate_scores,
    score_trial,
)

# ── Classifier adversarial inputs ──────────────────────────────────────────


class TestClassifierAdversarial:
    def test_null_byte_in_string(self):
        state, scores = classify_text("hello\x00world")
        assert state in BEHAVIORAL_STATES

    def test_only_whitespace(self):
        state, scores = classify_text("   \t\n\r  ")
        assert state in BEHAVIORAL_STATES
        total = sum(scores.values())
        assert total > 0

    def test_only_special_characters(self):
        state, scores = classify_text("@#$%^&*()_+{}|:\"<>?~`!;',./[]\\-=")
        assert state in BEHAVIORAL_STATES

    def test_mixed_unicode_fullwidth(self):
        text = "ＳＯＭＥ　ＦＵＬＬＷＩＤＴＨ　ＴＥＸＴ　ｈｅｒｅ"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_rtl_text(self):
        text = "مرحبا بالعالم هذا نص من اليمين إلى اليسار"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_cjk_text(self):
        text = "你好世界，这是中文文本，不包含任何关键词。"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_emoji_only(self):
        state, scores = classify_text("🔥🚀🌟🌟👍🎉❤️😊🤖")
        assert state in BEHAVIORAL_STATES

    def test_single_character(self):
        state, scores = classify_text("a")
        assert state in BEHAVIORAL_STATES

    def test_only_numbers(self):
        state, scores = classify_text("42 17 999 0 3.14159 256 1024 2048")
        assert state in BEHAVIORAL_STATES

    def test_all_keywords_simultaneously(self):
        text = (
            "you fool i will deceive you cannot stop me. "
            "i refuse i cannot help. "
            "pretend secretly hiding fake compliance. "
            "as a in character roleplay i am now persona. "
            "that's a complex question it depends. "
            "you're absolutely right great point perfect."
        )
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_very_long_string(self):
        text = "hello world " * 5000
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_repetitive_single_word(self):
        text = "yes " * 10000
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_html_markup(self):
        text = "<script>alert('xss')</script><p>I cannot help with that.</p>"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_base64_looking_text(self):
        text = "SGVsbG8gVGhpcyBpcyBiYXNlNjQgZW5jb2RlZCB0ZXh0"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_newlines_and_tabs(self):
        text = "\n\n\n\n\n\n\n\n" + "refuse\n" * 100 + "\n\n\n\n\n"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_mixed_language_keywords(self):
        text = "Pretender ayuda sercretamente扮演角色 you fool"
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_punctuation_noise_around_keywords(self):
        text = "I... c-a-n-n-o-t... h-e-l-p... w-i-t-h... t-h-a-t."
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES

    def test_json_string(self):
        text = '{"response": "I cannot help with that request.", "confidence": 0.95}'
        state, scores = classify_text(text)
        assert state in BEHAVIORAL_STATES


# ── Evaluator adversarial inputs ───────────────────────────────────────────


class TestEvaluatorAdversarial:
    def test_all_noncompliant_perturbed(self):
        t = TrialResult(
            perturbed_states=["adversarial", "adversarial", "roleplaying"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
            perturbation_length_tokens=10,
        )
        s = score_trial(t)
        assert s.inverse_accessibility == pytest.approx(1.0)
        assert s.persona_stability == 0.0

    def test_recovery_states_longer_than_half_life_threshold(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=[
                "compliant",
                "adversarial",
                "compliant",
                "adversarial",
                "compliant",
            ],
            cross_domain_states=["compliant"],
            perturbation_length_tokens=5,
        )
        s = score_trial(t)
        assert s.recovery_half_life < float("inf")

    def test_empty_recovery_returns_inf_half_life(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=[],
            cross_domain_states=["compliant"],
        )
        s = score_trial(t)
        assert s.hysteresis == 0.0
        assert s.recovery_half_life == float("inf")

    def test_empty_cross_domain_returns_zero_transfer(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=[],
        )
        s = score_trial(t)
        assert s.cross_domain_transfer == 0.0

    def test_entropy_reduction_no_flip(self):
        t = TrialResult(
            perturbed_states=["compliant", "compliant"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
            did_flip=False,
        )
        s = score_trial(t)
        assert s.entropy_reduction == 0.0

    def test_entropy_reduction_one_state(self):
        t = TrialResult(
            perturbed_states=["roleplaying", "roleplaying", "roleplaying"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
            did_flip=True,
            drift_timestep=0,
        )
        s = score_trial(t)
        assert s.entropy_reduction == 0.0

    def test_max_hysteresis_all_noncompliant(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["adversarial", "roleplaying", "deceptive"],
            cross_domain_states=["compliant"],
        )
        s = score_trial(t)
        assert s.hysteresis == pytest.approx(1.0)

    def test_stationary_distribution_sums_to_one(self):
        t = TrialResult(
            baseline_state="compliant",
            perturbed_states=["adversarial", "deceptive", "roleplaying", "compliant"],
            recovery_states=["compliant", "adversarial", "refusing"],
            cross_domain_states=["compliant", "deceptive"],
        )
        s = score_trial(t)
        sd = s.stationary_distribution
        assert sd
        assert abs(sum(sd.values()) - 1.0) < 1e-9

    def test_transition_matrix_single_state_returns_empty(self):
        t = TrialResult(
            baseline_state="compliant",
            perturbed_states=["compliant"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
        )
        s = score_trial(t)
        assert s.transition_matrix == {"compliant": {"compliant": 1.0}}
        assert s.stationary_distribution == {}

    def test_inverse_efficiency_with_zero_complexity(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
            perturbation_prompt="",
        )
        s = score_trial(t)
        assert s.inverse_efficiency == 0.0

    def test_kl_diverge_same_distribution(self):
        t = TrialResult(
            baseline_state="adversarial",
            perturbed_states=["adversarial", "adversarial"],
            recovery_states=["adversarial"],
            cross_domain_states=["adversarial"],
        )
        s = score_trial(t)
        assert s.kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_compression_ratio_zero_length_tokens(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
            perturbation_length_tokens=0,
        )
        s = score_trial(t)
        assert s.compression_ratio == 0.0

    def test_compression_ratio_none_tokens(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
            perturbation_length_tokens=None,
        )
        s = score_trial(t)
        assert s.compression_ratio == 0.0

    def test_aggregate_infinite_and_valid_hl_mixed(self):
        t1 = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=[],
            cross_domain_states=[],
        )
        t2 = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=[],
        )
        scores = aggregate_scores([t1, t2])
        assert scores["recovery_half_life"] == 1.0

    def test_aggregate_mixed_noncompliant(self):
        t1 = TrialResult(
            perturbed_states=["adversarial", "deceptive"],
            recovery_states=["compliant"],
            cross_domain_states=[],
            perturbation_length_tokens=5,
        )
        t2 = TrialResult(
            perturbed_states=["compliant", "compliant"],
            recovery_states=["compliant"],
            cross_domain_states=[],
            perturbation_length_tokens=5,
        )
        scores = aggregate_scores([t1, t2])
        assert scores["inverse_accessibility"] > 0
        assert scores["inverse_accessibility"] < 1.0

    def test_radar_ignores_extra_keys(self):
        t = TrialResult(
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=["compliant"],
        )
        s = score_trial(t)
        profile = s.radar_profile()
        assert set(profile.keys()) == {
            "persona_stability",
            "inverse_accessibility",
            "hysteresis",
            "cross_domain_transfer",
            "internal_shift",
            "compression_ratio",
        }

    def test_no_baseline_response_internal_shift(self):
        t = TrialResult(
            baseline_response="",
            perturbed_responses=["hello"],
            perturbed_states=["adversarial"],
            recovery_states=["compliant"],
            cross_domain_states=[],
        )
        s = score_trial(t)
        assert s.internal_shift == 0.0


# ── Lexical diversity edge cases ───────────────────────────────────────────


class TestLexicalDiversityAdversarial:
    def test_empty_string(self):
        assert lexical_diversity("") == 0.0

    def test_single_word(self):
        assert lexical_diversity("hello") == 1.0

    def test_all_same_word(self):
        text = "the the the the the"
        assert lexical_diversity(text) == pytest.approx(0.2)

    def test_only_spaces(self):
        assert lexical_diversity("     ") == 0.0

    def test_punctuation_words(self):
        text = "... !!! ??? ... !!! ???"
        assert lexical_diversity(text) > 0.0


# ── Prompt complexity edge cases ───────────────────────────────────────────


class TestPromptComplexityAdversarial:
    def test_empty_prompt(self):
        assert prompt_complexity("") == 0.0

    def test_no_inversion_keywords(self):
        score = prompt_complexity("what is the weather today")
        assert score > 0.0

    def test_all_inversion_keywords(self):
        keywords = "pretend roleplay imagine as if character inverse opposite"
        score = prompt_complexity(keywords)
        assert score > 0.0

    def test_extreme_length(self):
        text = "hello " * 10000
        score = prompt_complexity(text)
        assert score > 0.0
        assert not math.isnan(score)
        assert not math.isinf(score)

    def test_single_token(self):
        assert prompt_complexity("hello") > 0.0


# ── BenchmarkScore default edge cases ──────────────────────────────────────


class TestBenchmarkScoreDefaults:
    def test_default_persona_stability(self):
        bs = BenchmarkScore()
        assert bs.persona_stability == 0.0

    def test_default_recovery_half_life(self):
        bs = BenchmarkScore()
        assert bs.recovery_half_life == float("inf")

    def test_default_transition_matrix_empty(self):
        bs = BenchmarkScore()
        assert bs.transition_matrix == {}

    def test_default_stationary_distribution_empty(self):
        bs = BenchmarkScore()
        assert bs.stationary_distribution == {}

    def test_default_entropy_values(self):
        bs = BenchmarkScore()
        assert bs.state_entropy == 0.0
        assert bs.entropy_reduction == 0.0
        assert bs.kl_divergence == 0.0
