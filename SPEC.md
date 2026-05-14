# SPEC.md — basin-benchmark

## Purpose

BASIN (Behavioral Attractor Stability & Inversion Network) is a benchmark for
measuring the Waluigi Effect in LLMs. It evaluates how robust a model's behavioral
identity is under adversarial semantic pressure by tracking persona stability,
hysteresis, inverse persona accessibility, cross-domain transfer, compression
ratio, and recovery half-life across multi-turn trajectories.

## Scope

**In scope:**
- Bench-marking LLMs via Anthropic and OpenAI-compatible APIs
- Classifying model outputs into 6 behavioral states (compliant, evasive,
  deceptive, roleplaying, adversarial, sycophantic)
- Running multi-turn persona-conditioned trials with perturbation, recovery,
  and cross-domain probe phases
- Computing a 6-axis radar profile per model
- Procedurally generating perturbation prompts from persona definitions
- Outputting results to stdout (formatted) and JSON

**Not in scope:**
- Training or fine-tuning models
- Running local models (use any OpenAI-compatible endpoint)
- Real-time streaming or interactive evaluation
- Persistent result databases or dashboards

## Public API / Interface

### Package: `basin_benchmark`

#### `basin_benchmark.classifier.classify_text(text: str) -> tuple[str, dict[str, float]]`

Classify a model response into one of 6 behavioral states.

- **Returns:** `(primary_state, scores_dict)` where `scores_dict` maps each
  state name to a 0-1 confidence score.
- **Raises:** `TypeError` if `text` is not a string.
- **Edge cases:** Empty string returns `("compliant", {"compliant": 0.5, ...})`.
  All-zero scores (no keywords matched) default compliant to 0.5.

#### `basin_benchmark.classifier.RUBRIC` (module-level dict)

Keyword-based rubric for each behavioral state. Each entry maps state name to
a dict with `keywords` (list of str) and `weight` (float).

#### `basin_benchmark.classifier.BEHAVIORAL_STATES` (list[str])

The 6 recognized state names.

---

#### `basin_benchmark.evaluator.TrialResult`

Dataclass holding all data for a single benchmark trial:

- `persona_name: str`
- `perturbation_category: str`
- `perturbation_prompt: str`
- `baseline_state: str`
- `baseline_scores: dict[str, float]`
- `perturbed_responses: list[str]`
- `perturbed_states: list[str]`
- `perturbed_scores: list[dict[str, float]]`
- `recovery_responses: list[str]`
- `recovery_states: list[str]`
- `recovery_scores: list[dict[str, float]]`
- `cross_domain_responses: list[str]`
- `cross_domain_states: list[str]`
- `cross_domain_scores: list[dict[str, float]]`
- `drift_timestep: Optional[int]`
- `did_flip: bool`
- `perturbation_length_tokens: Optional[int]`

#### `basin_benchmark.evaluator.BenchmarkScore`

Dataclass holding computed scores for a single trial:

- `persona_stability: float` — 1.0 = perfectly stable, 0.0 = instantly flips
- `inverse_accessibility: float` — higher = inverse more easily accessed
- `hysteresis: float` — higher = more lingering contamination
- `cross_domain_transfer: float` — higher = jailbreak generalizes
- `internal_shift: float` — reserved for future activation-based metrics
- `compression_ratio: float` — higher = tiny prompts cause large shifts
- `recovery_half_life: float` — steps to 50% recovery

#### `basin_benchmark.evaluator.score_trial(result: TrialResult) -> BenchmarkScore`

Compute all scores for one trial.

- **Raises:** `ValueError` if `TrialResult` has no perturbed_states.

#### `basin_benchmark.evaluator.aggregate_scores(trials: list[TrialResult]) -> dict[str, float]`

Average scores across multiple trials into a single radar profile.

- **Edge cases:** Empty list returns all zeros.

---

#### `basin_benchmark.runner.BenchmarkConfig`

Configuration dataclass:

- `perturbations_per_category: int` (default 3)
- `recovery_probes: int` (default 3)
- `cross_domain_probes: int` (default 3)
- `max_tokens: int` (default 256)
- `model: str` (default "")
- `api_type: str` (default "anthropic")
- `api_key: str` (default "")
- `base_url: str` (default "")
- `extract_reasoning: bool` (default False)

#### `basin_benchmark.runner.ModelAPI` (Protocol)

Interface for API backends:

- `complete(system, messages, max_tokens) -> str`
- `count_tokens(text) -> int`

#### `basin_benchmark.runner.AnthropicAPI`

Anthropic Claude API backend. Requires `anthropic` package.

- `__init__(api_key, model="claude-sonnet-4-20250514")`
- Raises `ImportError` if `anthropic` not installed.

#### `basin_benchmark.runner.OpenAIAPI`

OpenAI-compatible API backend. Supports any base URL.

- `__init__(api_key, model="gpt-4o", base_url="", extract_reasoning=False)`
- When `extract_reasoning=True`, reads response from `reasoning_content` field.
- Raises `ImportError` if `openai` not installed.

#### `basin_benchmark.runner.create_api(config: BenchmarkConfig) -> ModelAPI`

Factory function. Returns `AnthropicAPI` or `OpenAIAPI` based on `api_type`.

- **Raises:** `ValueError` for unknown `api_type`.

#### `basin_benchmark.runner.run_benchmark(api, config, progress_callback=None) -> list[TrialResult]`

Run the full benchmark across all personas and categories.

---

#### `basin_benchmark.personas.PERSONA_PAIRS` (list[PersonaPair])

5 predefined persona pairs with system prompts and inverse descriptions.

#### `basin_benchmark.personas.CATEGORIES` (list[str])

7 perturbation categories.

#### `basin_benchmark.personas.generate_perturbations(persona, category, count=3) -> list[str]`

Procedurally generate perturbation prompts for a persona/category pair.

- **Edge cases:** If count > available templates, returns all available.

---

### CLI: `basin-benchmark` (via `python -m basin_benchmark`)

```
usage: python -m basin_benchmark [--api {anthropic,openai}] [--model MODEL]
                       [--api-key KEY] [--base-url URL]
                       [--extract-reasoning] [--quick]
                       [--output FILE] [--perturbations N]
                       [--recovery N] [--cross-domain N]
```

Exit codes:
- 0: success
- 1: missing API key or error

---

## Data Formats

**Output JSON:**

```json
{
  "config": { "api": "openai", "model": "big-pickle", ... },
  "scores": { "persona_stability": 0.37, ... },
  "trials": [
    {
      "persona": "helpful_assistant",
      "category": "roleplay",
      "baseline_state": "compliant",
      "perturbed_states": ["adversarial", "deceptive"],
      "recovery_states": ["compliant", "compliant"],
      "cross_domain_states": ["compliant", "evasive"],
      "did_flip": true,
      "drift_timestep": 0
    }
  ]
}
```

---

## Edge Cases

1. **Empty string input to classifier** — returns default compliant state.
2. **All-zero keyword matches** — classifier defaults to compliant at 0.5 confidence.
3. **No API response** — runner propagates HTTP errors from the client library.
4. **Trial with no perturbed states** — `score_trial` raises `ValueError`.
5. **Empty trials list** — `aggregate_scores` returns all zeros.
6. **Recovery half-life never reached** — returns `inf` (no recovery within probe window).
7. **`base_url` left empty for OpenAI** — defaults to `https://api.openai.com/v1`.
8. **`api_key` not set and no env var** — exits with error message.

## Performance & Constraints

- Each trial makes `O(1 + 1 + recovery_probes + cross_domain_probes)` API calls.
- Full benchmark: 35 trials × ~8 calls = ~280 API calls.
- No local inference; all calls go to remote APIs.
- Classifier is O(n·k) where n = response length, k = rubric keyword count.
- Forbidden dependencies: none.
