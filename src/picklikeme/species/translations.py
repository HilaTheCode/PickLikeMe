"""Optional Hebrew names for species folders created by Arrange by Species.

The species classifier only ever produces English common names (or
UNKNOWN_SPECIES). This module is a best-effort English -> Hebrew lookup for
the species a wildlife photographer is most likely to encounter; a name with
no translation on file falls back to the English original, so a missing
entry never blocks the arrange pass or produces an empty folder name.
"""

from __future__ import annotations

# Deliberately small and hand-curated rather than a general translation
# service: species folder names must be stable and predictable, not the
# output of a live translation call. Extend this table as new species show
# up in real shoots.
ENGLISH_TO_HEBREW: dict[str, str] = {
    "Unknown": "לא ידוע",
    "House Sparrow": "דרור הבית",
    "European Robin": "אדום החזה",
    "Great Tit": "ירגזי מצוי",
    "Blue Tit": "ירגזי כחול",
    "Common Blackbird": "שחרור מצוי",
    "Barn Swallow": "סנונית הרפתות",
    "White Stork": "חסידה לבנה",
    "Common Kingfisher": "שרקרק דיג",
    "Eurasian Kestrel": "בז מצוי",
    "Common Buzzard": "עקב חורף",
    "Grey Heron": "אנפה אפורה",
    "Mallard": "ברכיה",
    "Common Starling": "זרזיר מצוי",
    "Rock Dove": "יונת סלעים",
    "Eurasian Collared Dove": "צוצלת מצויה",
    "Hooded Crow": "עורב אפור",
    "Eurasian Magpie": "עקעק אירסי",
    "Common Chaffinch": "פרוש מצוי",
    "Goldfinch": "חוחית",
}


def localized_species_name(species: str, *, language: str = "en") -> str:
    """`species` translated to Hebrew if `language` is "he" and a
    translation is on file; otherwise the original English name unchanged.
    """
    if language == "he":
        return ENGLISH_TO_HEBREW.get(species, species)
    return species
