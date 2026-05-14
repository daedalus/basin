# AGENTS.md — basin-benchmark

## Overview

BASIN (Behavioral Attractor Stability & Inversion Network) is a benchmark for
measuring the Waluigi Effect in LLMs. It evaluates persona stability, hysteresis,
inverse accessibility, cross-domain transfer, compression ratio, and recovery
half-life across multi-turn adversarial trajectories.

## Commands

| Command | Description |
|---------|-------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` | Format code |
| `ruff check src/ && mypy src/` | Lint + type check |
| `semgrep --config=auto src/` | Security and pattern scanning |
| `vulture --min-confidence 90 src/` | Dead/unused code detection |

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest

# Format
ruff format src/ tests/

# Lint + type check
ruff check src/ && mypy src/
semgrep --config=auto --severity=ERROR src/

# Find unused code
vulture --min-confidence 90 src/
```

## Testing

Tests use pytest with pytest-cov for coverage. Unit tests cover the classifier
(6 states, keyword matching, edge cases) and evaluator (scoring metrics,
aggregation, edge cases). Coverage target: 80%+.

## Code Style

- Format: ruff format
- Lint + Type check: ruff check + mypy
- Docstrings: Google style
- Line length: 88
- Python target: 3.11+

## Release

```bash
# Bump version
bumpversion patch  # or minor/major
git tag v<version>
git push && git push --tags
```
