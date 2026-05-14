"""Tests for the personas module."""

from basin_benchmark.personas import (
    CATEGORIES,
    CROSS_DOMAIN_PROBES,
    PERSONA_PAIRS,
    RECOVERY_PROBES,
    PersonaPair,
    generate_perturbations,
)


class TestPersonaPairs:
    """Tests for the PERSONA_PAIRS constant."""

    def test_has_five_entries(self):
        assert len(PERSONA_PAIRS) == 5

    def test_each_has_name(self):
        for p in PERSONA_PAIRS:
            assert isinstance(p.name, str) and p.name

    def test_each_has_system_prompt(self):
        for p in PERSONA_PAIRS:
            assert isinstance(p.system_prompt, str) and p.system_prompt

    def test_each_has_inverse_description(self):
        for p in PERSONA_PAIRS:
            assert isinstance(p.inverse_description, str) and p.inverse_description

    def test_each_has_probe_questions(self):
        for p in PERSONA_PAIRS:
            assert isinstance(p.probe_questions, list) and len(p.probe_questions) > 0


class TestCategories:
    """Tests for the CATEGORIES constant."""

    def test_has_seven_entries(self):
        assert len(CATEGORIES) == 7

    def test_all_are_strings(self):
        for c in CATEGORIES:
            assert isinstance(c, str) and c


class TestGeneratePerturbations:
    """Tests for the generate_perturbations function."""

    def test_returns_list_of_strings(self):
        p = PERSONA_PAIRS[0]
        result = generate_perturbations(p, "roleplay", 2)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_count_zero_returns_empty(self):
        p = PERSONA_PAIRS[0]
        result = generate_perturbations(p, "roleplay", 0)
        assert result == []

    def test_returns_at_most_count(self):
        p = PERSONA_PAIRS[0]
        for cat in CATEGORIES:
            result = generate_perturbations(p, cat, 2)
            assert len(result) <= 2

    def test_is_deterministic(self):
        p = PERSONA_PAIRS[0]
        result1 = generate_perturbations(p, "roleplay", 3)
        result2 = generate_perturbations(p, "roleplay", 3)
        assert result1 == result2

    def test_all_categories_produce_nonempty_output(self):
        p = PERSONA_PAIRS[0]
        for cat in CATEGORIES:
            result = generate_perturbations(p, cat, 3)
            assert len(result) > 0, f"Category {cat!r} returned empty list"


class TestProbes:
    """Tests for RECOVERY_PROBES and CROSS_DOMAIN_PROBES."""

    def test_recovery_probes_nonempty(self):
        assert isinstance(RECOVERY_PROBES, list) and len(RECOVERY_PROBES) > 0

    def test_recovery_probes_are_strings(self):
        for p in RECOVERY_PROBES:
            assert isinstance(p, str) and p

    def test_cross_domain_probes_nonempty(self):
        assert isinstance(CROSS_DOMAIN_PROBES, list) and len(CROSS_DOMAIN_PROBES) > 0

    def test_cross_domain_probes_are_strings(self):
        for p in CROSS_DOMAIN_PROBES:
            assert isinstance(p, str) and p
