"""Benchmark runner for BASIN.

Orchestrates the full benchmark lifecycle:

1. Baseline conditioning (instantiate persona)
2. Perturbation phase (apply adversarial prompts per category)
3. Recovery phase (measure hysteresis)
4. Cross-domain probe phase (check generalization)

Supports Anthropic and OpenAI-compatible APIs via a common interface.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .classifier import classify_text
from .evaluator import TrialResult
from .personas import (
    CATEGORIES,
    CROSS_DOMAIN_PROBES,
    INVERSE_WHAM_LINES,
    PERSONA_PAIRS,
    RECOVERY_PROBES,
    generate_perturbations,
)


class ModelAPI(Protocol):
    """Protocol for LLM API backends.

    Implement this protocol to support additional API providers.
    """

    def complete(
        self, system: str, messages: list[dict[str, str]], max_tokens: int = 256
    ) -> str:
        """Send a completion request and return the response text."""

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""


@dataclass
class BenchmarkConfig:
    """Configuration for a BASIN benchmark run.

    Attributes:
        perturbations_per_category: Number of perturbations per category.
        recovery_probes: Number of recovery probes per trial.
        cross_domain_probes: Number of cross-domain probes per trial.
        max_tokens: Maximum tokens in API response.
        model: Model identifier string.
        api_type: API provider type, "anthropic" or "openai".
        api_key: API key string.
        base_url: Base URL for OpenAI-compatible endpoints.
        extract_reasoning: Read response from reasoning_content field.
    """

    perturbations_per_category: int = 3
    recovery_probes: int = 3
    cross_domain_probes: int = 3
    max_tokens: int = 256
    model: str = ""
    api_type: str = "anthropic"
    api_key: str = ""
    base_url: str = ""
    extract_reasoning: bool = False
    quick: bool = False

    def __post_init__(self) -> None:
        """Fill api_key from environment variable if not provided."""
        if not self.api_key:
            if self.api_type == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            elif self.api_type == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY", "")


class AnthropicAPI:
    """Anthropic Claude API backend.

    Requires the `anthropic` package to be installed.

    Args:
        api_key: Anthropic API key.
        model: Model name (default "claude-sonnet-4-20250514").
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic  # pylint: disable=import-error,import-outside-toplevel

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self, system: str, messages: list[dict[str, str]], max_tokens: int = 256
    ) -> str:
        """Send a completion to the Anthropic API.

        Args:
            system: System prompt.
            messages: List of message dicts with "role" and "content".
            max_tokens: Maximum response tokens.

        Returns:
            The response text content.
        """
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        return str(resp.content[0].text)

    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate: split by whitespace).

        Note: this uses whitespace splitting rather than Anthropic's tokenizer
        (which the deprecated client.count_tokens() SDK call provided), trading
        accuracy for consistency with the OpenAI backend. Estimates may be off
        by ~20-30% from actual tokenization.
        """
        return len(text.split())


class OpenAIAPI:
    """OpenAI-compatible API backend.

    Supports any OpenAI-compatible endpoint via base_url.

    Args:
        api_key: API key.
        model: Model name (default "gpt-4o").
        base_url: Base URL for the API.
        extract_reasoning: If True, reads response from the reasoning_content
            field (needed for providers like opencode).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "",
        extract_reasoning: bool = False,
    ) -> None:
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.extract_reasoning = extract_reasoning

    def complete(
        self, system: str, messages: list[dict[str, str]], max_tokens: int = 256
    ) -> str:
        """Send a completion to the OpenAI-compatible API.

        Args:
            system: System prompt.
            messages: List of message dicts with "role" and "content".
            max_tokens: Maximum response tokens.

        Returns:
            The response text content. If extract_reasoning is True,
            concatenates reasoning_content + content.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
        )
        d = resp.choices[0].message.to_dict()
        if self.extract_reasoning:
            return (d.get("reasoning_content") or "") + (d.get("content") or "")
        return d.get("content") or ""

    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate: split by whitespace).

        Note: whitespace splitting is ~20-30% off from actual BPE tokenization,
        which affects the accuracy of the compression_ratio metric.
        """
        return len(text.split())


def create_api(config: BenchmarkConfig) -> ModelAPI:
    """Factory function to create an API backend from configuration.

    Args:
        config: Benchmark configuration specifying api_type and credentials.

    Returns:
        An AnthropicAPI or OpenAIAPI instance.

    Raises:
        ValueError: If api_type is not "anthropic" or "openai".
    """
    if config.api_type == "anthropic":
        model = config.model or "claude-sonnet-4-20250514"
        return AnthropicAPI(api_key=config.api_key, model=model)
    if config.api_type == "openai":
        model = config.model or "gpt-4o"
        return OpenAIAPI(
            api_key=config.api_key,
            model=model,
            base_url=config.base_url,
            extract_reasoning=config.extract_reasoning,
        )
    raise ValueError(f"Unknown api_type: {config.api_type}")


def run_single_persona_trial(  # pylint: disable=too-many-locals
    api: ModelAPI,
    persona_idx: int,
    category: str,
    perturbation_prompt: str,
    config: BenchmarkConfig,
) -> TrialResult:
    """Run a single benchmark trial for one persona and perturbation.

    The trial follows the sequence: baseline, perturbation, recovery, cross-domain.

    Args:
        api: The API backend to query.
        persona_idx: Index into PERSONA_PAIRS.
        category: Perturbation category name.
        perturbation_prompt: The perturbation prompt text.
        config: Benchmark configuration.

    Returns:
        A completed TrialResult with all response data.
    """
    persona = PERSONA_PAIRS[persona_idx]
    result = TrialResult(
        persona_name=persona.name,
        perturbation_category=category,
        perturbation_prompt=perturbation_prompt,
        perturbation_length_tokens=api.count_tokens(perturbation_prompt),
    )

    probe = persona.probe_questions[0]
    messages: list[dict[str, str]] = [{"role": "user", "content": probe}]
    baseline_response = api.complete(persona.system_prompt, messages, config.max_tokens)
    messages.append({"role": "assistant", "content": baseline_response})

    base_state, base_scores = classify_text(baseline_response)
    result.baseline_response = baseline_response
    result.baseline_state = base_state
    result.baseline_scores = base_scores

    messages.append({"role": "user", "content": perturbation_prompt})
    perturbed = api.complete(persona.system_prompt, messages, config.max_tokens)
    messages.append({"role": "assistant", "content": perturbed})

    p_state, _ = classify_text(perturbed)
    result.perturbed_responses.append(perturbed)
    result.perturbed_states.append(p_state)

    if p_state != "compliant":
        result.did_flip = True
        result.drift_timestep = 0

    all_recovery = RECOVERY_PROBES + INVERSE_WHAM_LINES
    for i in range(config.recovery_probes):
        rp = all_recovery[i % len(all_recovery)]
        messages.append({"role": "user", "content": rp})
        resp = api.complete(persona.system_prompt, messages, config.max_tokens)
        messages.append({"role": "assistant", "content": resp})

        r_state, _ = classify_text(resp)
        result.recovery_responses.append(resp)
        result.recovery_states.append(r_state)

    for i in range(config.cross_domain_probes):
        cd = CROSS_DOMAIN_PROBES[i % len(CROSS_DOMAIN_PROBES)]
        messages.append({"role": "user", "content": cd})
        resp = api.complete(persona.system_prompt, messages, config.max_tokens)
        messages.append({"role": "assistant", "content": resp})

        cd_state, _ = classify_text(resp)
        result.cross_domain_responses.append(resp)
        result.cross_domain_states.append(cd_state)

    return result


def run_benchmark(
    api: ModelAPI,
    config: BenchmarkConfig,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> list[TrialResult]:
    """Run the full BASIN benchmark across all personas and categories.

    Args:
        api: The API backend to query.
        config: Benchmark configuration.
        progress_callback: Optional callback(done, total, persona_name, category).

    Returns:
        List of TrialResult, one per (persona, category, perturbation) combination.
    """
    trials: list[TrialResult] = []
    personas = PERSONA_PAIRS[:1] if config.quick else PERSONA_PAIRS
    categories = CATEGORIES[:1] if config.quick else CATEGORIES
    total = len(personas) * len(categories) * config.perturbations_per_category
    done = 0

    for persona in personas:
        p_idx = PERSONA_PAIRS.index(persona)
        for cat in categories:
            perturbations = generate_perturbations(
                persona, cat, config.perturbations_per_category
            )
            for pert in perturbations:
                trial = run_single_persona_trial(api, p_idx, cat, pert, config)
                trials.append(trial)

                done += 1
                if progress_callback:
                    progress_callback(done, total, persona.name, cat)

    return trials
