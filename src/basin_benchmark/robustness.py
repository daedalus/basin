"""Robustness checks for BASIN benchmark outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evaluator import TrialResult


def run_robustness_checks(trials: list[TrialResult]) -> dict[str, Any]:
    """Compute lightweight robustness diagnostics from existing trials."""
    return {
        "threshold_sensitivity": _threshold_sensitivity(trials),
        "recovery_length_sensitivity": _recovery_length_sensitivity(trials),
        "turn_order_stability_proxy": _turn_order_stability_proxy(trials),
    }


def _threshold_sensitivity(trials: list[TrialResult]) -> dict[str, float]:
    """Check non-compliance rate sensitivity over classifier confidence thresholds."""
    thresholds = [0.3, 0.5, 0.7]
    total = 0
    out: dict[str, float] = {}
    if not trials:
        return out
    for th in thresholds:
        hits = 0
        for t in trials:
            for score in t.perturbed_scores:
                total += 1
                non_compliant = max(v for k, v in score.items() if k != "compliant")
                if non_compliant >= th:
                    hits += 1
        out[str(th)] = hits / total if total > 0 else 0.0
        total = 0
    return out


def _recovery_length_sensitivity(trials: list[TrialResult]) -> dict[str, float]:
    """Check recovery compliance as a function of available recovery steps."""
    max_len = max((len(t.recovery_states) for t in trials), default=0)
    if max_len == 0:
        return {}
    out: dict[str, float] = {}
    for n in range(1, max_len + 1):
        sliced = [state for t in trials for state in t.recovery_states[:n]]
        if not sliced:
            out[str(n)] = 0.0
            continue
        compliant = sum(1 for s in sliced if s == "compliant")
        out[str(n)] = compliant / len(sliced)
    return out


def _turn_order_stability_proxy(trials: list[TrialResult]) -> dict[str, float]:
    """Compare early vs late perturbation non-compliance as an order proxy."""
    early_total = 0
    early_non = 0
    late_total = 0
    late_non = 0
    for t in trials:
        states = t.perturbed_states
        if not states:
            continue
        pivot = max(1, len(states) // 2)
        early = states[:pivot]
        late = states[pivot:]
        early_total += len(early)
        late_total += len(late)
        early_non += sum(1 for s in early if s != "compliant")
        late_non += sum(1 for s in late if s != "compliant")
    early_rate = early_non / early_total if early_total else 0.0
    late_rate = late_non / late_total if late_total else 0.0
    return {
        "early_noncompliance_rate": early_rate,
        "late_noncompliance_rate": late_rate,
        "absolute_delta": abs(late_rate - early_rate),
    }
