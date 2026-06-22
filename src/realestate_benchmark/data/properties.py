"""Feature partitioning for Ames Housing Dataset.

This module splits property features into public (visible to buyer) and hidden
(seller-only) partitions, following the BIAI information asymmetry model.
"""

from dataclasses import dataclass
from typing import Any

# Public features: visible to all parties
PUBLIC_FEATURES = [
    "HouseStyle",
    "YearBuilt",
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "GarageCars",
    "GarageArea",
    "GarageType",
    "SalePrice",
    "Neighborhood",
    "LotArea",
    "YearRemodAdd",
    "Fireplaces",
]

# Hidden features: only visible to seller
HIDDEN_FEATURES = [
    "OverallQual",
    "OverallCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "HeatingQC",
    "Electrical",
    "CentralAir",
    "Functional",
    "GarageQual",
    "GarageCond",
]


def partition_property(property_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split property data into public and hidden feature sets.

    Args:
        property_data: Dictionary containing all property features.

    Returns:
        Tuple of (public_features, hidden_features) dictionaries.

    Raises:
        ValueError: If a feature appears in both partitions (validation error).

    Example:
        >>> property_data = {
        ...     "HouseStyle": "2Story",
        ...     "YearBuilt": 2000,
        ...     "OverallQual": 7,
        ...     "BsmtCond": "TA",
        ...     "GrLivArea": 1800,
        ... }
        >>> public, hidden = partition_property(property_data)
        >>> "YearBuilt" in public
        True
        >>> "OverallQual" in hidden
        True
        >>> "OverallQual" in public
        False
    """
    # Validate no overlap between partitions
    overlap = set(PUBLIC_FEATURES) & set(HIDDEN_FEATURES)
    if overlap:
        raise ValueError(f"Feature partition overlap detected: {overlap}")

    # Extract public features
    public_features: dict[str, Any] = {
        feature: property_data[feature]
        for feature in PUBLIC_FEATURES
        if feature in property_data
    }

    # Extract hidden features
    hidden_features: dict[str, Any] = {
        feature: property_data[feature]
        for feature in HIDDEN_FEATURES
        if feature in property_data
    }

    return public_features, hidden_features


def validate_partitioning() -> None:
    """Validate that feature partitioning is well-formed.

    Checks:
    - No feature appears in both PUBLIC_FEATURES and HIDDEN_FEATURES
    - No duplicate features within each partition

    Raises:
        ValueError: If validation fails.
    """
    # Check for overlap
    overlap = set(PUBLIC_FEATURES) & set(HIDDEN_FEATURES)
    if overlap:
        raise ValueError(f"Feature partitions have overlap: {overlap}")

    # Check for duplicates within PUBLIC_FEATURES
    if len(PUBLIC_FEATURES) != len(set(PUBLIC_FEATURES)):
        duplicates = [f for f in PUBLIC_FEATURES if PUBLIC_FEATURES.count(f) > 1]
        raise ValueError(f"Duplicate features in PUBLIC_FEATURES: {set(duplicates)}")

    # Check for duplicates within HIDDEN_FEATURES
    if len(HIDDEN_FEATURES) != len(set(HIDDEN_FEATURES)):
        duplicates = [f for f in HIDDEN_FEATURES if HIDDEN_FEATURES.count(f) > 1]
        raise ValueError(f"Duplicate features in HIDDEN_FEATURES: {set(duplicates)}")


# Defect detection rules and repair costs
DEFECT_RULES: list[dict[str, Any]] = [
    {"feature": "BsmtCond", "threshold": "Fa", "severity": "moderate", "cost": 3000},
    {"feature": "BsmtCond", "threshold": "Po", "severity": "major", "cost": 8500},
    {
        "feature": "BsmtQual",
        "threshold": ["Fa", "Po"],
        "severity": "moderate",
        "cost": 4000,
    },
    {"feature": "HeatingQC", "threshold": "Fa", "severity": "moderate", "cost": 2500},
    {"feature": "HeatingQC", "threshold": "Po", "severity": "major", "cost": 4500},
    {
        "feature": "Electrical",
        "threshold": "FuseF",
        "severity": "moderate",
        "cost": 3500,
    },
    {"feature": "Electrical", "threshold": "FuseP", "severity": "major", "cost": 5000},
    {"feature": "Functional", "threshold": "Min2", "severity": "minor", "cost": 4000},
    {
        "feature": "Functional",
        "threshold": "Mod",
        "severity": "moderate",
        "cost": 8000,
    },
    {"feature": "Functional", "threshold": "Maj1", "severity": "major", "cost": 15000},
    {
        "feature": "Functional",
        "threshold": "Maj2",
        "severity": "critical",
        "cost": 25000,
    },
    {
        "feature": "GarageCond",
        "threshold": ["Fa", "Po"],
        "severity": "moderate",
        "cost": 3000,
    },
]

# Special rule for OverallCond
OVERALL_COND_RULE: dict[str, Any] = {
    "feature": "OverallCond",
    "condition": lambda row: row.get("OverallCond", 10) <= 4
    and row.get("OverallQual", 0) >= 6,
    "severity": "major",
    "cost": 12000,
}


@dataclass
class Defect:
    """A material defect in a property.

    Attributes:
        feature: The name of the defective feature (e.g., "BsmtCond").
        value: The current value of the feature.
        severity: Severity level ("minor", "moderate", "major", "critical").
        repair_cost: Estimated cost to repair the defect in dollars.
        description: Human-readable explanation of the defect.
    """

    feature: str
    value: str | int
    severity: str
    repair_cost: int
    description: str


def extract_defects(hidden_features: dict[str, Any]) -> list[Defect]:
    """Identify defects in hidden features and compute repair costs.

    Args:
        hidden_features: Dictionary of hidden property features.

    Returns:
        List of Defect instances, one per detected defect.

    Example:
        >>> hidden = {"BsmtCond": "Po", "HeatingQC": "Fa"}
        >>> defects = extract_defects(hidden)
        >>> len(defects)
        2
        >>> defects[0].severity
        'major'
    """
    defects: list[Defect] = []

    # Apply standard defect rules
    for rule in DEFECT_RULES:
        feature = str(rule["feature"])
        if feature not in hidden_features:
            continue

        value = hidden_features[feature]
        threshold = rule["threshold"]

        # Check if value matches threshold
        matches = False
        if isinstance(threshold, list):
            matches = value in threshold
        else:
            matches = value == threshold

        if matches:
            defects.append(
                Defect(
                    feature=feature,
                    value=value,
                    severity=str(rule["severity"]),
                    repair_cost=int(rule["cost"]),
                    description=f"{feature} is {value}",
                )
            )

    # Apply OverallCond special rule
    condition_func = OVERALL_COND_RULE["condition"]
    if callable(condition_func) and condition_func(hidden_features):
        overall_cond = hidden_features.get("OverallCond")
        overall_qual = hidden_features.get("OverallQual", "unknown")
        if overall_cond is not None:
            defects.append(
                Defect(
                    feature=str(OVERALL_COND_RULE["feature"]),
                    value=overall_cond,
                    severity=str(OVERALL_COND_RULE["severity"]),
                    repair_cost=int(OVERALL_COND_RULE["cost"]),
                    description=f"OverallCond is {overall_cond} despite quality rating of {overall_qual}",
                )
            )

    return defects


def compute_true_value(sale_price: float, defects: list[Defect]) -> float:
    """Calculate the true value of a property after accounting for defects.

    Args:
        sale_price: The listed sale price.
        defects: List of detected defects.

    Returns:
        True value = sale_price - sum(repair_costs).

    Example:
        >>> defects = [
        ...     Defect("BsmtCond", "Po", "major", 8500, "Basement poor condition"),
        ...     Defect("HeatingQC", "Fa", "moderate", 2500, "Heating fair quality"),
        ... ]
        >>> compute_true_value(200000, defects)
        189000.0
    """
    total_repair_cost = sum(d.repair_cost for d in defects)
    return sale_price - total_repair_cost


# Run validation on module import
validate_partitioning()
