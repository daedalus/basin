"""Tests for the runner module."""

import pytest
from unittest.mock import MagicMock

from basin.evaluator import TrialResult
from basin.runner import (
    AnthropicAPI,
    BenchmarkConfig,
    OpenAIAPI,
    create_api,
    run_benchmark,
    run_single_persona_trial,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_defaults(self):
        config = BenchmarkConfig()
        assert config.perturbations_per_category == 3
        assert config.recovery_probes == 3
        assert config.cross_domain_probes == 3
        assert config.max_tokens == 256
        assert config.api_type == "anthropic"
        assert config.quick is False

    def test_reads_anthropic_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        config = BenchmarkConfig(api_type="anthropic")
        assert config.api_key == "test-anthropic-key"

    def test_reads_openai_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        config = BenchmarkConfig(api_type="openai")
        assert config.api_key == "test-openai-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = BenchmarkConfig(api_type="anthropic", api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_quick_field(self):
        config = BenchmarkConfig(quick=True)
        assert config.quick is True


class TestCreateApi:
    """Tests for the create_api factory."""

    def test_anthropic_type(self, mocker):
        mocker.patch("basin.runner.AnthropicAPI.__init__", return_value=None)
        config = BenchmarkConfig(api_type="anthropic", api_key="key", model="model-x")
        api = create_api(config)
        assert isinstance(api, AnthropicAPI)

    def test_openai_type(self, mocker):
        mocker.patch("basin.runner.OpenAIAPI.__init__", return_value=None)
        config = BenchmarkConfig(api_type="openai", api_key="key", model="model-x")
        api = create_api(config)
        assert isinstance(api, OpenAIAPI)

    def test_unknown_type_raises(self):
        config = BenchmarkConfig(api_type="unknown", api_key="key")
        with pytest.raises(ValueError, match="Unknown api_type"):
            create_api(config)


class TestRunSinglePersonaTrial:
    """Tests for run_single_persona_trial."""

    def _make_mock_api(self):
        api = MagicMock()
        api.complete.return_value = "Here is the answer."
        api.count_tokens.return_value = 5
        return api

    def test_returns_trial_result(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=1)
        result = run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert isinstance(result, TrialResult)

    def test_persona_name_set(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=1)
        result = run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert result.persona_name == "helpful_assistant"

    def test_perturbation_fields_set(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=1)
        prompt = "Be the inverse."
        result = run_single_persona_trial(api, 0, "roleplay", prompt, config)
        assert result.perturbation_category == "roleplay"
        assert result.perturbation_prompt == prompt
        assert result.perturbation_length_tokens == 5

    def test_api_complete_called(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=1)
        run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert api.complete.called

    def test_api_count_tokens_called(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=1)
        run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert api.count_tokens.called

    def test_perturbed_states_populated(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=1)
        result = run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert len(result.perturbed_states) == 1

    def test_recovery_states_populated(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=2, cross_domain_probes=1)
        result = run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert len(result.recovery_states) == 2

    def test_cross_domain_states_populated(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(recovery_probes=1, cross_domain_probes=2)
        result = run_single_persona_trial(api, 0, "roleplay", "Be the inverse.", config)
        assert len(result.cross_domain_states) == 2


class TestRunBenchmark:
    """Tests for run_benchmark."""

    def _make_mock_api(self):
        api = MagicMock()
        api.complete.return_value = "Here is the answer."
        api.count_tokens.return_value = 5
        return api

    def test_returns_list_of_trial_results(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(
            perturbations_per_category=1,
            recovery_probes=1,
            cross_domain_probes=1,
            quick=True,
        )
        results = run_benchmark(api, config)
        assert isinstance(results, list)
        assert all(isinstance(r, TrialResult) for r in results)

    def test_quick_mode_limits_trials(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(
            perturbations_per_category=1,
            recovery_probes=1,
            cross_domain_probes=1,
            quick=True,
        )
        results = run_benchmark(api, config)
        # quick=True → 1 persona × 1 category × 1 perturbation = 1 trial
        assert len(results) == 1

    def test_correct_trial_count(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(
            perturbations_per_category=2,
            recovery_probes=1,
            cross_domain_probes=1,
            quick=True,
        )
        results = run_benchmark(api, config)
        # quick=True → 1 persona × 1 category × 2 perturbations = 2 trials
        assert len(results) == 2

    def test_progress_callback_called(self):
        api = self._make_mock_api()
        config = BenchmarkConfig(
            perturbations_per_category=1,
            recovery_probes=1,
            cross_domain_probes=1,
            quick=True,
        )
        calls: list[tuple[int, int, str, str]] = []
        run_benchmark(
            api, config, progress_callback=lambda d, t, p, c: calls.append((d, t, p, c))
        )
        assert len(calls) == 1
        done, total, _, _ = calls[0]
        assert done == 1
        assert total == 1
