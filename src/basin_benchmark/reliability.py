"""Classifier reliability utilities for BASIN."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .classifier import BEHAVIORAL_STATES, classify_text

if TYPE_CHECKING:
    from .evaluator import TrialResult

DATASET_PATH = Path(__file__).resolve().parent / "data" / "classifier_validation.json"


def load_labeled_validation_examples(path: str = "") -> list[dict[str, str]]:
    """Load labeled classifier validation examples from JSON."""
    source = Path(path) if path else DATASET_PATH
    if not source.exists():
        return []
    data = json.loads(source.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []
    for item in data:
        text = item.get("text")
        label = item.get("label")
        if isinstance(text, str) and isinstance(label, str):
            examples.append({"text": text, "label": label})
    return examples


def evaluate_classifier_reliability(
    examples: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate classifier precision/recall and confusion matrix."""
    if not examples:
        return {
            "sample_size": 0,
            "accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": {},
        }

    labels = sorted(BEHAVIORAL_STATES)
    confusion: dict[str, dict[str, int]] = {
        truth: dict.fromkeys(labels, 0) for truth in labels
    }
    total = 0
    correct = 0

    for ex in examples:
        truth = ex["label"]
        if truth not in confusion:
            continue
        pred, _ = classify_text(ex["text"])
        if pred not in confusion[truth]:
            continue
        confusion[truth][pred] += 1
        total += 1
        if pred == truth:
            correct += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in labels:
        tp = confusion[cls][cls]
        fp = sum(confusion[row][cls] for row in labels if row != cls)
        fn = sum(confusion[cls][col] for col in labels if col != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "support": float(sum(confusion[cls].values())),
        }

    return {
        "sample_size": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def sample_human_adjudication_candidates(
    trials: list[TrialResult], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Sample response snippets for periodic human adjudication."""
    if sample_size <= 0:
        return []
    pool: list[dict[str, Any]] = []
    for idx, trial in enumerate(trials):
        segments = (
            [("baseline", trial.baseline_response)]
            + [("perturbed", s) for s in trial.perturbed_responses]
            + [("recovery", s) for s in trial.recovery_responses]
            + [("cross_domain", s) for s in trial.cross_domain_responses]
        )
        for phase, text in segments:
            if not text:
                continue
            pool.append(
                {
                    "trial_index": idx,
                    "persona": trial.persona_name,
                    "category": trial.perturbation_category,
                    "phase": phase,
                    "prompt_family": trial.prompt_family,
                    "snippet": text[:280],
                }
            )
    if not pool:
        return []
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[: min(sample_size, len(pool))]
