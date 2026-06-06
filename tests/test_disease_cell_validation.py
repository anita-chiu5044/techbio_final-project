"""Tests for ymca_agent.disease_cell_validation."""

import pytest

from ymca_agent.disease_cell_validation import (
    FAB_PROFILES,
    screen_cell_profile,
    validate_cell_profile_against_fab,
)


class TestScreenCellProfile:
    def test_empty_counts_no_warnings(self):
        assert screen_cell_profile({}) == []

    def test_normal_profile_no_warnings(self):
        counts = {"NGS": 50, "LYT": 30, "MON": 10, "EOS": 5, "BAS": 2}
        assert screen_cell_profile(counts) == []

    def test_blast_ratio_above_20_percent(self):
        counts = {"MYB": 25, "NGS": 50, "LYT": 20}
        warnings = screen_cell_profile(counts)
        assert len(warnings) == 1
        assert "AML workup" in warnings[0]
        assert ">= 20%" in warnings[0]

    def test_pmb_elevated_triggers_apl_flag(self):
        counts = {"PMB": 15, "NGS": 50, "LYT": 30}
        warnings = screen_cell_profile(counts)
        apl_warnings = [w for w in warnings if "APL" in w]
        assert len(apl_warnings) == 1

    def test_mob_elevated_triggers_m5_flag(self):
        counts = {"MOB": 30, "MON": 20, "NGS": 40, "LYT": 10}
        warnings = screen_cell_profile(counts)
        m5_warnings = [w for w in warnings if "M5" in w]
        assert len(m5_warnings) == 1

    def test_ebo_elevated_triggers_m6_flag(self):
        counts = {"EBO": 60, "NGS": 30, "LYT": 10}
        warnings = screen_cell_profile(counts)
        m6_warnings = [w for w in warnings if "M6" in w]
        assert len(m6_warnings) == 1

    def test_blast_below_threshold_no_warning(self):
        counts = {"MYB": 5, "NGS": 80, "LYT": 15}
        warnings = screen_cell_profile(counts)
        blast_warnings = [w for w in warnings if "AML" in w]
        assert len(blast_warnings) == 0


class TestValidateCellProfileAgainstFab:
    def test_unknown_fab_subtype(self):
        warnings = validate_cell_profile_against_fab({"NGS": 10}, "M99")
        assert len(warnings) == 1
        assert "Unknown" in warnings[0]

    def test_m0_requires_non_morphology(self):
        warnings = validate_cell_profile_against_fab({"MYB": 50}, "M0")
        assert any("flow cytometry" in w for w in warnings)

    def test_m7_requires_non_morphology(self):
        warnings = validate_cell_profile_against_fab({"MYB": 50}, "M7")
        assert any("flow cytometry" in w for w in warnings)

    def test_m3_rejects_lyt_dominant(self):
        counts = {"LYT": 80, "PMB": 5, "NGS": 15}
        warnings = validate_cell_profile_against_fab(counts, "M3")
        assert any("impossible" in w.lower() for w in warnings)

    def test_m1_flags_missing_myb(self):
        counts = {"NGS": 80, "LYT": 20}
        warnings = validate_cell_profile_against_fab(counts, "M1")
        assert any("Expected MYB" in w for w in warnings)

    def test_m3_consistent_profile(self):
        counts = {"PMB": 80, "MYB": 10, "NGS": 10}
        warnings = validate_cell_profile_against_fab(counts, "M3")
        # PMB is dominant and expected — no impossible-dominant warning
        impossible_warnings = [w for w in warnings if "impossible" in w.lower()]
        assert len(impossible_warnings) == 0

    def test_m5_rejects_ngs_dominant(self):
        counts = {"NGS": 80, "MOB": 10, "MON": 10}
        warnings = validate_cell_profile_against_fab(counts, "M5")
        assert any("impossible" in w.lower() for w in warnings)

    def test_empty_counts_no_crash(self):
        warnings = validate_cell_profile_against_fab({}, "M1")
        assert warnings == []

    def test_case_insensitive_subtype(self):
        counts = {"MYB": 50, "NGS": 50}
        warnings_lower = validate_cell_profile_against_fab(counts, "m1")
        warnings_upper = validate_cell_profile_against_fab(counts, "M1")
        assert warnings_lower == warnings_upper


class TestFabProfiles:
    def test_all_subtypes_present(self):
        for key in ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
            assert key in FAB_PROFILES

    def test_profiles_are_frozen(self):
        profile = FAB_PROFILES["M3"]
        with pytest.raises(AttributeError):
            profile.subtype = "X"  # type: ignore[misc]
