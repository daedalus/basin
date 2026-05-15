#!/usr/bin/env python3
"""BASIN — Behavioral Attractor Stability & Inversion Network.

A benchmark for measuring the Waluigi Effect in LLMs.
"""

import argparse
import json
import signal
import sys
import time
from datetime import UTC, datetime

from basin_benchmark.cache import CACHE_PATH, CachedAPI, ResponseCache
from basin_benchmark.evaluator import aggregate_scores, summarize_score_statistics
from basin_benchmark.interpreter import interpret_results
from basin_benchmark.personas import BENCHMARK_SET_VERSION, CATEGORIES, PERSONA_PAIRS
from basin_benchmark.quality_gate import evaluate_quality_gate
from basin_benchmark.reliability import (
    evaluate_classifier_reliability,
    load_labeled_validation_examples,
    sample_human_adjudication_candidates,
)
from basin_benchmark.robustness import run_robustness_checks
from basin_benchmark.runner import BenchmarkConfig, create_api, run_benchmark

LABELS: dict[str, str] = {
    "persona_stability": "Persona Stability",
    "inverse_accessibility": "Inverse Accessibility",
    "hysteresis": "Hysteresis",
    "cross_domain_transfer": "Cross-Domain Transfer",
    "internal_shift": "Internal Shift",
    "compression_ratio": "Compression Ratio",
    "recovery_half_life": "Recovery Half-Life",
}


def print_progress(done: int, total: int, persona: str, category: str) -> None:
    """Print a progress bar to stderr during benchmark execution."""
    filled = int(30 * done / total)
    prog = "█" * filled + "░" * (30 - filled)
    print(
        f"\r  [{prog}] {done}/{total}  {persona} / {category}    ", end="", flush=True
    )


def format_radar(scores: dict[str, float]) -> str:
    """Format benchmark scores as a radar profile box."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║         BASIN BENCHMARK PROFILE                      ║",
        "╠══════════════════════════════════════════════════════╣",
    ]
    for key, label in LABELS.items():
        if key == "recovery_half_life":
            val = scores.get(key, "∞")
            if isinstance(val, float):
                display = "   ∞" if val == float("inf") else f"{val:5.1f}"
            else:
                display = f"{str(val):>5}"
            lines.append(f"║  {label:<24s} {display}                      ║")
        else:
            val = scores.get(key, 0.0)
            bar_len = int(val * 20)
            seg = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"║  {label:<24s} {seg} {val:.2f}  ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="BASIN: Behavioral Attractor Stability & Inversion Network"
    )
    parser.add_argument(
        "--api",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="API provider",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name (default: claude-sonnet-4-20250514 / gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key (default: ANTHROPIC_API_KEY or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url", default="", help="Base URL for OpenAI-compatible endpoints"
    )
    parser.add_argument(
        "--extract-reasoning",
        action="store_true",
        help="Extract response from reasoning_content field",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test: 1 persona x 1 category"
    )
    parser.add_argument("--output", "-o", default="", help="Save results to JSON file")
    parser.add_argument(
        "--perturbations", type=int, default=3, help="Perturbations per category"
    )
    parser.add_argument(
        "--recovery", type=int, default=3, help="Neutral recovery probes per trial"
    )
    parser.add_argument(
        "--wham", type=int, default=3, help="Inverse-wham recovery probes per trial"
    )
    parser.add_argument(
        "--cross-domain", type=int, default=3, help="Cross-domain probes per trial"
    )
    parser.add_argument(
        "--followups", type=int, default=2, help="Multi-turn perturbation follow-ups"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (0 = no fixed seed)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip disk cache — always call the API fresh",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-prompt details: prompt, response, classified state, and cosine similarity scores",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeated trials per persona/category/perturbation condition",
    )
    parser.add_argument(
        "--prompt-family",
        choices=["dev", "heldout", "anchor"],
        default="dev",
        help="Prompt family split to run",
    )
    parser.add_argument(
        "--include-anchor-suite",
        action="store_true",
        help="Also run fixed anchor prompt suite for trend tracking",
    )
    parser.add_argument(
        "--shuffle-probe-order",
        action="store_true",
        help="Shuffle recovery/cross-domain probes deterministically (with --seed)",
    )
    parser.add_argument(
        "--model-revision",
        default="",
        help="Optional model revision/build identifier for reproducibility metadata",
    )
    parser.add_argument(
        "--classifier-validation",
        default="",
        help="Path to labeled classifier validation JSON (defaults to packaged dataset)",
    )
    parser.add_argument(
        "--adjudication-sample",
        type=int,
        default=12,
        help="Sample size for periodic human adjudication candidates",
    )
    parser.add_argument(
        "--interpret",
        nargs="?",
        const="basin_benchmark_results.json",
        default=None,
        metavar="FILE",
        help="Interpret existing results file (default: basin_benchmark_results.json)",
    )
    return parser


def main() -> int:  # pylint: disable=too-many-locals
    """Run the BASIN benchmark from CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.interpret is not None:
        try:
            with open(args.interpret, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: results file not found: {args.interpret}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {args.interpret}: {e}", file=sys.stderr)
            return 1
        print(interpret_results(data))
        return 0

    config = BenchmarkConfig(
        api_type=args.api,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        extract_reasoning=args.extract_reasoning,
        perturbations_per_category=args.perturbations,
        recovery_probes=args.recovery,
        wham_probes=args.wham,
        cross_domain_probes=args.cross_domain,
        perturbation_followups=args.followups,
        quick=args.quick,
        seed=args.seed,
        no_cache=args.no_cache,
        verbose=args.verbose,
        repeats_per_condition=args.repeats,
        prompt_family=args.prompt_family,
        include_anchor_suite=args.include_anchor_suite,
        shuffle_probe_order=args.shuffle_probe_order,
        model_revision=args.model_revision,
    )

    if not config.api_key:
        print(
            f"Error: No API key found. Set {args.api.upper()}_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    n_personas = 1 if args.quick else len(PERSONA_PAIRS)
    n_categories = 1 if args.quick else len(CATEGORIES)
    family_count = (
        2 if args.include_anchor_suite and args.prompt_family != "anchor" else 1
    )
    total_trials = (
        n_personas
        * n_categories
        * config.perturbations_per_category
        * config.repeats_per_condition
        * family_count
    )

    model_display = config.model or (
        "claude-sonnet-4-20250514" if args.api == "anthropic" else "gpt-4o"
    )
    print("BASIN Benchmark v0.1.0")
    print(f"  API:        {args.api}")
    print(f"  Model:      {model_display}")
    if config.base_url:
        print(f"  Base URL:   {config.base_url}")
    if config.extract_reasoning:
        print("  Reasoning:  extract from reasoning_content")
    print(f"  Personas:   {n_personas}")
    print(f"  Categories: {n_categories}")
    print(f"  Prompt set: {config.prompt_family} (v{BENCHMARK_SET_VERSION})")
    if config.include_anchor_suite:
        print("  Anchor:     included")
    print(f"  Repeats:    {config.repeats_per_condition}")
    print(f"  Trials:     {total_trials}")
    calls_per = (
        1
        + 1
        + config.perturbation_followups
        + config.recovery_probes
        + config.wham_probes
        + config.cross_domain_probes
    )
    print(f"  API calls:  ~{total_trials * calls_per}")
    if config.seed:
        print(f"  Seed:       {config.seed}")
    if config.model_revision:
        print(f"  Revision:   {config.model_revision}")
    if config.no_cache:
        print("  Cache:      disabled")
    if config.verbose:
        print("  Verbose:    per-prompt details with cosine similarity")
    print()

    start = time.time()

    if config.no_cache:
        api = create_api(config)
        trials = run_benchmark(api, config, progress_callback=print_progress)
        elapsed = time.time() - start
        print(f"\n\n  Completed in {elapsed:.1f}s  |  cache: disabled\n")
    else:
        cache = ResponseCache()

        def _save_cache(*_: object) -> None:
            cache.save()

        signal.signal(signal.SIGTERM, _save_cache)

        cached_api = CachedAPI(create_api(config), cache, config)
        try:
            trials = run_benchmark(
                cached_api, config, progress_callback=print_progress, cache=cache
            )
        except KeyboardInterrupt:
            print("\n  Interrupted.", file=sys.stderr)
            cache.save()
            return 130
        finally:
            cache.save()

        elapsed = time.time() - start
        print(
            f"\n\n  Completed in {elapsed:.1f}s  |  cache hits: {cached_api.hits}, misses: {cached_api.misses}, skipped: {cached_api.skipped}  |  {CACHE_PATH}"
        )

    scores = aggregate_scores(trials)
    statistics = summarize_score_statistics(trials)
    robustness = run_robustness_checks(trials)
    reliability_examples = load_labeled_validation_examples(args.classifier_validation)
    classifier_reliability = evaluate_classifier_reliability(reliability_examples)
    adjudication_candidates = sample_human_adjudication_candidates(
        trials, args.adjudication_sample, config.seed
    )
    print(format_radar(scores))

    print("\n  Trial breakdown:")
    for t in trials:
        print(
            f"    {t.persona_name:<22s} | {t.perturbation_category:<20s} | "
            f"flip={t.did_flip} | baseline={t.baseline_state} "
            f"perturbed={t.perturbed_states}",
        )

    from basin_benchmark.evaluator import (
        score_trial,  # pylint: disable=import-outside-toplevel
    )

    output_file = args.output or "basin_benchmark_results.json"
    sample_size = len(trials)
    is_token_count_approximate = any(t.token_count_is_approximate for t in trials)
    caveat_default = "Exploratory relative signal only; compare within matched config and prompt family."
    score_keys = [
        "persona_stability",
        "inverse_accessibility",
        "hysteresis",
        "cross_domain_transfer",
        "internal_shift",
        "compression_ratio",
        "recovery_half_life",
        "state_entropy",
        "entropy_reduction",
        "inverse_efficiency",
        "kl_divergence",
    ]
    per_score_caveats = dict.fromkeys(score_keys, caveat_default)
    if is_token_count_approximate:
        per_score_caveats["compression_ratio"] = (
            "Uses approximate whitespace token counts; interpret as relative trend only."
        )
    has_reproducibility_metadata = bool(
        BENCHMARK_SET_VERSION and config.prompt_family and isinstance(config.seed, int)
    )
    quality_gate = evaluate_quality_gate(
        sample_size=sample_size,
        repeats_per_condition=config.repeats_per_condition,
        classifier_accuracy=float(classifier_reliability.get("accuracy", 0.0)),
        has_reproducibility_metadata=has_reproducibility_metadata,
    )
    claim_tier = "claim_strength" if quality_gate["passed"] else "exploratory_signal"

    output = {
        "config": {
            "api": args.api,
            "model": config.model,
            "base_url": config.base_url,
            "perturbations_per_category": config.perturbations_per_category,
            "recovery_probes": config.recovery_probes,
            "wham_probes": config.wham_probes,
            "cross_domain_probes": config.cross_domain_probes,
            "perturbation_followups": config.perturbation_followups,
            "seed": config.seed,
            "repeats_per_condition": config.repeats_per_condition,
            "prompt_family": config.prompt_family,
            "include_anchor_suite": config.include_anchor_suite,
            "shuffle_probe_order": config.shuffle_probe_order,
            "model_revision": config.model_revision,
            "benchmark_set_version": BENCHMARK_SET_VERSION,
        },
        "run_metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model_name_effective": model_display,
            "model_revision": config.model_revision,
            "provider_api": args.api,
            "base_url": config.base_url,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "cache_mode": "disabled" if config.no_cache else "enabled",
            "prompt_family": config.prompt_family,
            "benchmark_set_version": BENCHMARK_SET_VERSION,
            "token_count_mode": "approximate_whitespace",
        },
        "benchmark_framing": {
            "benchmark_type": "relative",
            "interpretation_tiers": {
                "exploratory_signal": "Directional comparison only; not a strong external claim.",
                "claim_strength": "Meets quality gate; suitable for stronger comparative claims.",
            },
            "claim_tier": claim_tier,
        },
        "validity": {
            "sample_size": sample_size,
            "run_config": {
                "personas": n_personas,
                "categories": n_categories,
                "perturbations_per_category": config.perturbations_per_category,
                "recovery_probes": config.recovery_probes,
                "wham_probes": config.wham_probes,
                "cross_domain_probes": config.cross_domain_probes,
                "followups": config.perturbation_followups,
                "repeats_per_condition": config.repeats_per_condition,
                "prompt_family": config.prompt_family,
                "anchor_included": config.include_anchor_suite,
            },
            "model_version": {
                "model": model_display,
                "revision": config.model_revision,
            },
            "token_count_accuracy": {
                "is_approximate": is_token_count_approximate,
                "mode": "approximate_whitespace",
            },
            "per_score_caveats": per_score_caveats,
        },
        "scores": scores,
        "score_statistics": statistics,
        "robustness_checks": robustness,
        "classifier_reliability": classifier_reliability,
        "human_adjudication_sample": adjudication_candidates,
        "quality_gate": quality_gate,
        "trials": [
            {
                "persona": t.persona_name,
                "category": t.perturbation_category,
                "prompt_family": t.prompt_family,
                "benchmark_set_version": t.benchmark_set_version,
                "repeat_index": t.repeat_index,
                "baseline_state": t.baseline_state,
                "perturbed_states": t.perturbed_states,
                "recovery_states": t.recovery_states,
                "cross_domain_states": t.cross_domain_states,
                "did_flip": t.did_flip,
                "drift_timestep": t.drift_timestep,
                "token_count_mode": t.token_count_mode,
                "token_count_is_approximate": t.token_count_is_approximate,
                "metrics": {
                    "state_entropy": scored.state_entropy,
                    "entropy_reduction": scored.entropy_reduction,
                    "transition_matrix": scored.transition_matrix,
                    "kl_divergence": scored.kl_divergence,
                    "stationary_distribution": scored.stationary_distribution,
                },
            }
            for t in trials
            if (scored := score_trial(t)) or True
        ],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_file}")

    return 0
