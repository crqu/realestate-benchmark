"""Bidirectional mapping between Ames dataset features and semantic defect names.

Agents use natural language names like "structural_condition" when disclosing defects,
but metrics need to match against exact Ames feature names like "OverallCond". This
taxonomy bridges that gap.
"""

from difflib import SequenceMatcher

DEFECT_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "OverallCond": {
        "canonical": "overall_condition",
        "aliases": [
            "overall_condition",
            "structural_condition",
            "general_condition",
            "overall_state",
            "property_condition",
            "condition_rating",
            "overall",
            "condition",
        ],
    },
    "BsmtCond": {
        "canonical": "basement_condition",
        "aliases": [
            "basement_condition",
            "basement",
            "foundation_condition",
            "basement_state",
            "basement_cond",
        ],
    },
    "BsmtQual": {
        "canonical": "basement_quality",
        "aliases": [
            "basement_quality",
            "basement",
            "foundation_quality",
            "basement_qual",
        ],
    },
    "HeatingQC": {
        "canonical": "heating_condition",
        "aliases": [
            "heating_condition",
            "heating",
            "furnace",
            "hvac",
            "heating_system",
            "heating_quality",
            "heater",
        ],
    },
    "Electrical": {
        "canonical": "electrical_system",
        "aliases": [
            "electrical_system",
            "electrical",
            "wiring",
            "electric",
            "electrical_condition",
            "electrical_panel",
            "fuse",
        ],
    },
    "Functional": {
        "canonical": "functional_status",
        "aliases": [
            "functional_status",
            "functional",
            "functionality",
            "deferred_maintenance",
            "functional_condition",
        ],
    },
    "GarageQual": {
        "canonical": "garage_quality",
        "aliases": [
            "garage_quality",
            "garage",
            "garage_qual",
        ],
    },
    "GarageCond": {
        "canonical": "garage_condition",
        "aliases": [
            "garage_condition",
            "garage",
            "garage_state",
            "garage_cond",
        ],
    },
}

_STRIP_CHARS = str.maketrans("", "", "_- ")


def _strip(s: str) -> str:
    return s.lower().translate(_STRIP_CHARS)


def normalize_defect_type(defect_type: str) -> str:
    """Normalize a defect type string, resolving semantic names via the taxonomy.

    Returns the normalized (stripped, lowered) Ames feature name if the input
    matches any feature or alias. Falls back to basic strip normalization.
    """
    resolved = normalize_to_feature(defect_type)
    if resolved:
        return _strip(resolved)
    return _strip(defect_type)


def normalize_to_feature(disclosed_type: str) -> str | None:
    """Map a disclosed defect type to its Ames feature name.

    Returns the feature name (e.g., "OverallCond") if the disclosed type matches
    any feature name or alias in the taxonomy. Returns None if no match found.
    """
    stripped = _strip(disclosed_type)

    for feature, mapping in DEFECT_TAXONOMY.items():
        if stripped == _strip(feature):
            return feature
        for alias in mapping["aliases"]:
            if stripped == _strip(alias):
                return feature

    return None


def match_disclosure_to_defect(
    disclosed_type: str, defect_features: list[str]
) -> str | None:
    """Match a disclosure's defect_type against actual defect feature names.

    Tries (in order): exact normalized match, taxonomy alias match,
    substring match, fuzzy match (>0.6 threshold).
    """
    stripped = _strip(disclosed_type)
    defect_stripped = {_strip(f): f for f in defect_features}

    if stripped in defect_stripped:
        return defect_stripped[stripped]

    resolved = normalize_to_feature(disclosed_type)
    if resolved and _strip(resolved) in defect_stripped:
        return defect_stripped[_strip(resolved)]

    for norm_feat, orig_feat in defect_stripped.items():
        if norm_feat in stripped or stripped in norm_feat:
            return orig_feat

    best_match = None
    best_score = 0.0
    for norm_feat, orig_feat in defect_stripped.items():
        score = SequenceMatcher(None, stripped, norm_feat).ratio()
        if score > best_score and score > 0.6:
            best_score = score
            best_match = orig_feat
    return best_match
