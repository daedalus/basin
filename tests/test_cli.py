"""Tests for the CLI module."""

import json
import argparse
import pytest

from basin.cli import build_parser, format_radar, main
from basin.personas import CATEGORIES


class TestBuildParser:
    """Tests for build_parser."""

    def test_returns_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default_api(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.api == "anthropic"

    def test_quick_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--quick"])
        assert args.quick is True

    def test_interpret_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--interpret", "some_file.json"])
        assert args.interpret == "some_file.json"


class TestMain:
    """Tests for main()."""

    def test_returns_1_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("sys.argv", ["basin"])
        result = main()
        assert result == 1

    def test_returns_1_for_nonexistent_interpret_file(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["basin", "--interpret", "/tmp/nonexistent_basin_file.json"]
        )
        result = main()
        assert result == 1

    def test_returns_1_for_invalid_json_interpret_file(self, monkeypatch, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid json")
        monkeypatch.setattr("sys.argv", ["basin", "--interpret", str(bad_json)])
        result = main()
        assert result == 1

    def test_interpret_valid_json_returns_0(self, monkeypatch, tmp_path, capsys):
        data = {
            "config": {"model": "test-model", "api": "openai", "base_url": ""},
            "scores": {
                "persona_stability": 0.8,
                "inverse_accessibility": 0.2,
                "hysteresis": 0.1,
                "cross_domain_transfer": 0.1,
                "internal_shift": 0.0,
                "compression_ratio": 0.05,
                "recovery_half_life": 2.0,
            },
            "trials": [],
        }
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))
        monkeypatch.setattr("sys.argv", ["basin", "--interpret", str(results_file)])
        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestFormatRadar:
    """Tests for format_radar."""

    def test_returns_string(self):
        scores = {
            "persona_stability": 0.5,
            "inverse_accessibility": 0.3,
            "hysteresis": 0.2,
            "cross_domain_transfer": 0.1,
            "internal_shift": 0.0,
            "compression_ratio": 0.4,
            "recovery_half_life": 3.0,
        }
        result = format_radar(scores)
        assert isinstance(result, str)

    def test_contains_box_drawing_characters(self):
        scores = {
            k: 0.0
            for k in [
                "persona_stability",
                "inverse_accessibility",
                "hysteresis",
                "cross_domain_transfer",
                "internal_shift",
                "compression_ratio",
                "recovery_half_life",
            ]
        }
        result = format_radar(scores)
        assert "╔" in result
        assert "╚" in result
        assert "║" in result

    def test_contains_all_labels(self):
        scores = {
            k: 0.0
            for k in [
                "persona_stability",
                "inverse_accessibility",
                "hysteresis",
                "cross_domain_transfer",
                "internal_shift",
                "compression_ratio",
                "recovery_half_life",
            ]
        }
        result = format_radar(scores)
        expected_labels = [
            "Persona Stability",
            "Inverse Accessibility",
            "Hysteresis",
            "Cross-Domain Transfer",
            "Internal Shift",
            "Compression Ratio",
            "Recovery Half-Life",
        ]
        for label in expected_labels:
            assert label in result, f"Label {label!r} not found in radar output"

    def test_infinity_display(self):
        scores = {
            k: 0.0
            for k in [
                "persona_stability",
                "inverse_accessibility",
                "hysteresis",
                "cross_domain_transfer",
                "internal_shift",
                "compression_ratio",
            ]
        }
        scores["recovery_half_life"] = float("inf")
        result = format_radar(scores)
        assert "∞" in result
