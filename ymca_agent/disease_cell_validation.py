"""Disease-cell plausibility validation for FAB M0-M7 AML subtypes.

Provides morphological screening warnings based on cell type distributions.
These are NOT diagnostic claims — they are flags for review.

References:
  - FAB classification: Bennett et al., Br J Haematol 1976
  - YMCA_Architecture_Review.md section 6.1
  - WHO 2022: molecular markers are primary; FAB is morphological supplement
"""

from __future__ import annotations

from dataclasses import dataclass


BLAST_LIKE = frozenset({"MYB", "MOB", "PMB", "PMO", "MMZ"})

# Blast ratio threshold for AML workup flag (WHO/FAB standard: 20%)
BLAST_RATIO_THRESHOLD = 0.20

# APL flag: if PMB fraction exceeds this among all cells
APL_PMB_THRESHOLD = 0.10


@dataclass(frozen=True)
class FabProfile:
    subtype: str
    description: str
    expected: frozenset[str]
    elevated: frozenset[str]
    impossible_dominant: frozenset[str]
    requires_non_morphology: bool
    notes: str


FAB_PROFILES: dict[str, FabProfile] = {
    "M0": FabProfile(
        "M0", "AML undifferentiated",
        expected=frozenset(),
        elevated=frozenset(),
        impossible_dominant=frozenset(),
        requires_non_morphology=True,
        notes="Cannot classify by morphology alone; requires flow cytometry",
    ),
    "M1": FabProfile(
        "M1", "AML without maturation",
        expected=frozenset({"MYB"}),
        elevated=frozenset({"MYB"}),
        impossible_dominant=frozenset({"LYT", "LYA", "MON", "EBO"}),
        requires_non_morphology=False,
        notes="MYB >= 90% of non-erythroid cells",
    ),
    "M2": FabProfile(
        "M2", "AML with maturation",
        expected=frozenset({"MYB", "MYO", "PMO"}),
        elevated=frozenset({"MYB"}),
        impossible_dominant=frozenset({"LYT", "LYA", "MON", "EBO"}),
        requires_non_morphology=False,
        notes="MYB with granulocytic maturation (MYO, MMZ, PMO)",
    ),
    "M3": FabProfile(
        "M3", "Acute promyelocytic leukemia (APL)",
        expected=frozenset({"PMB"}),
        elevated=frozenset({"PMB"}),
        impossible_dominant=frozenset({"LYT", "LYA", "MON", "MOB", "MYO"}),
        requires_non_morphology=False,
        notes="PMB dominant; Auer rods; URGENT — requires immediate ATRA",
    ),
    "M4": FabProfile(
        "M4", "Acute myelomonocytic leukemia",
        expected=frozenset({"MYB", "MON", "PMO"}),
        elevated=frozenset({"MON", "MYB"}),
        impossible_dominant=frozenset({"LYT", "LYA", "EBO"}),
        requires_non_morphology=False,
        notes="Monocytic component >= 20%; both myeloid and monocytic lineage",
    ),
    "M5": FabProfile(
        "M5", "Acute monocytic leukemia",
        expected=frozenset({"MOB", "PMO", "MON"}),
        elevated=frozenset({"MOB", "MON"}),
        impossible_dominant=frozenset({"LYT", "LYA", "NGS", "EBO"}),
        requires_non_morphology=False,
        notes="M5a: MOB >= 80%; M5b: mixed monocytic maturation",
    ),
    "M6": FabProfile(
        "M6", "Acute erythroid leukemia",
        expected=frozenset({"EBO"}),
        elevated=frozenset({"EBO"}),
        impossible_dominant=frozenset({"LYT", "LYA", "MON"}),
        requires_non_morphology=False,
        notes="EBO >= 50% of nucleated cells",
    ),
    "M7": FabProfile(
        "M7", "Acute megakaryoblastic leukemia",
        expected=frozenset(),
        elevated=frozenset(),
        impossible_dominant=frozenset(),
        requires_non_morphology=True,
        notes="Cannot classify by morphology alone; requires flow cytometry",
    ),
}


def screen_cell_profile(hard_counts: dict[str, int]) -> list[str]:
    """Screen cell distribution for morphological warnings.

    Returns a list of warning strings. Empty list = no concerns.
    These are screening flags, NOT diagnostic claims.
    """
    warnings: list[str] = []
    total = sum(hard_counts.values())
    if total == 0:
        return warnings

    blast_count = sum(hard_counts.get(c, 0) for c in BLAST_LIKE)
    blast_ratio = blast_count / total

    if blast_ratio >= BLAST_RATIO_THRESHOLD:
        warnings.append(
            f"Blast-like cells ({blast_count}/{total} = {blast_ratio:.0%}) "
            f">= 20% — consider AML workup"
        )

    pmb_count = hard_counts.get("PMB", 0)
    if pmb_count > 0 and pmb_count / total >= APL_PMB_THRESHOLD:
        warnings.append(
            f"PMB (promyelocyte) elevated ({pmb_count}/{total} = {pmb_count/total:.0%}) "
            f"— consider urgent APL (M3) screening"
        )

    mob_count = hard_counts.get("MOB", 0)
    if mob_count > 0 and mob_count / total >= 0.20:
        warnings.append(
            f"MOB (monoblast) elevated ({mob_count}/{total} = {mob_count/total:.0%}) "
            f"— profile consistent with M5 (monocytic leukemia)"
        )

    ebo_count = hard_counts.get("EBO", 0)
    if ebo_count > 0 and ebo_count / total >= 0.50:
        warnings.append(
            f"EBO (erythroblast) elevated ({ebo_count}/{total} = {ebo_count/total:.0%}) "
            f"— profile consistent with M6 (erythroleukemia)"
        )

    return warnings


def validate_cell_profile_against_fab(
    hard_counts: dict[str, int],
    suggested_fab: str,
) -> list[str]:
    """Validate cell distribution against a suggested FAB subtype.

    Returns a list of warning strings. Empty list = no inconsistencies found.
    """
    key = suggested_fab.upper()
    profile = FAB_PROFILES.get(key)
    if profile is None:
        return [f"Unknown FAB subtype: {suggested_fab}"]

    warnings: list[str] = []

    if profile.requires_non_morphology:
        warnings.append(
            f"{key} ({profile.description}) cannot be classified by morphology alone "
            f"— requires flow cytometry / immunophenotyping"
        )
        return warnings

    total = sum(hard_counts.values())
    if total == 0:
        return warnings

    # Check for impossible dominant cell types
    dominant_cell = max(hard_counts, key=lambda k: hard_counts[k])
    if dominant_cell in profile.impossible_dominant:
        warnings.append(
            f"{dominant_cell} is dominant ({hard_counts[dominant_cell]}/{total}) "
            f"but is impossible for {key} ({profile.description}) "
            f"— review suggested subtype"
        )

    # Check expected cells are present
    for expected_cell in profile.expected:
        if hard_counts.get(expected_cell, 0) == 0:
            warnings.append(
                f"Expected {expected_cell} for {key} ({profile.description}) "
                f"but none found in cell counts"
            )

    return warnings
