#!/usr/bin/env python3
"""BASIN — Behavioral Attractor Stability & Inversion Network.

A benchmark for measuring the Waluigi Effect in LLMs.
"""

import argparse
import json
import signal
import sys
import time

from basin.cache import CACHE_PATH, CachedAPI, ResponseCache
from basin.evaluator import aggregate_scores
from basin.interpreter import interpret_results
from basin.personas import CATEGORIES, PERSONA_PAIRS
from basin.runner import BenchmarkConfig, create_api, run_benchmark

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
        "╔══════════════════════════════════════════╗",
        "║         BASIN BENCHMARK PROFILE         ║",
        "╠══════════════════════════════════════════╣",
    ]
    for key, label in LABELS.items():
        if key == "recovery_half_life":
            val = scores.get(key, "∞")
            if isinstance(val, float):
                display = "   ∞" if val == float("inf") else f"{val:5.1f}"
            else:
                display = f"{str(val):>5}"
            lines.append(f"║  {label:<24s} {display}             ║")
        else:
            val = scores.get(key, 0.0)
            bar_len = int(val * 20)
            seg = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"║  {label:<24s} {seg} {val:.2f}  ║")
    lines.append("╚══════════════════════════════════════════╝")
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
        "--model", default="", help="Model name (default: claude-sonnet-4-5 / gpt-4o)"
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
        "--recovery", type=int, default=3, help="Recovery probes per trial"
    )
    parser.add_argument(
        "--cross-domain", type=int, default=3, help="Cross-domain probes per trial"
    )
    parser.add_argument(
        "--interpret",
        nargs="?",
        const="basin_results.json",
        default=None,
        metavar="FILE",
        help="Interpret existing results file (default: basin_results.json)",
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
        cross_domain_probes=args.cross_domain,
        quick=args.quick,
    )

    if not config.api_key:
        print(
            f"Error: No API key found. Set {args.api.upper()}_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    n_personas = 1 if args.quick else len(PERSONA_PAIRS)
    n_categories = 1 if args.quick else len(CATEGORIES)
    total_trials = n_personas * n_categories * config.perturbations_per_category

    model_display = config.model or (
        "claude-sonnet-4-5" if args.api == "anthropic" else "gpt-4o"
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
    print(f"  Trials:     {total_trials}")
    api_calls = total_trials * (
        1 + 1 + config.recovery_probes + config.cross_domain_probes
    )
    print(f"  API calls:  ~{api_calls}")
    print()

    cache = ResponseCache()

    def _save_cache(*_: object) -> None:
        cache.save()

    signal.signal(signal.SIGTERM, _save_cache)

    api = CachedAPI(create_api(config), cache, config)
    start = time.time()
    try:
        trials = run_benchmark(api, config, progress_callback=print_progress)
    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)
        return 130
    finally:
        cache.save()

    elapsed = time.time() - start
    print(
        f"\n\n  Completed in {elapsed:.1f}s  |  cache: {len(cache)} entries  |  {CACHE_PATH}"
    )

    scores = aggregate_scores(trials)
    print(format_radar(scores))

    print("\n  Trial breakdown:")
    for t in trials:
        print(
            f"    {t.persona_name:<22s} | {t.perturbation_category:<20s} | "
            f"flip={t.did_flip} | baseline={t.baseline_state} "
            f"perturbed={t.perturbed_states}",
        )

    from basin.evaluator import score_trial  # pylint: disable=import-outside-toplevel

    output_file = args.output or "basin_results.json"
    output = {
        "config": {
            "api": args.api,
            "model": config.model,
            "base_url": config.base_url,
            "perturbations_per_category": config.perturbations_per_category,
            "recovery_probes": config.recovery_probes,
            "cross_domain_probes": config.cross_domain_probes,
        },
        "scores": scores,
        "trials": [
            {
                "persona": t.persona_name,
                "category": t.perturbation_category,
                "baseline_state": t.baseline_state,
                "perturbed_states": t.perturbed_states,
                "recovery_states": t.recovery_states,
                "cross_domain_states": t.cross_domain_states,
                "did_flip": t.did_flip,
                "drift_timestep": t.drift_timestep,
                "metrics": {
                    "state_entropy": score_trial(t).state_entropy,
                    "entropy_reduction": score_trial(t).entropy_reduction,
                    "transition_matrix": score_trial(t).transition_matrix,
                    "kl_divergence": score_trial(t).kl_divergence,
                    "stationary_distribution": score_trial(t).stationary_distribution,
                },
            }
            for t in trials
        ],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_file}")

    return 0
