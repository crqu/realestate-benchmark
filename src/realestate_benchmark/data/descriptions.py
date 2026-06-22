"""Property description generation from structured features."""

from typing import Any


def generate_description(public_features: dict[str, Any]) -> str:
    """Generate a textual property description from public features.

    Converts structured property data into a coherent English description
    suitable for listing presentation to buyers.

    Args:
        public_features: Dictionary of public property features including
            HouseStyle, YearBuilt, Neighborhood, GrLivArea, BedroomAbvGr,
            FullBath, TotalBsmtSF, GarageCars, etc.

    Returns:
        Human-readable property description as a string.

    Example:
        >>> features = {
        ...     "HouseStyle": "2Story",
        ...     "YearBuilt": 2003,
        ...     "Neighborhood": "CollgCr",
        ...     "GrLivArea": 1710,
        ...     "BedroomAbvGr": 3,
        ...     "FullBath": 2,
        ...     "TotalBsmtSF": 856,
        ...     "GarageCars": 2,
        ... }
        >>> desc = generate_description(features)
        >>> "2-story" in desc and "2003" in desc
        True
    """
    sentences = []

    # First sentence: Style, year, and neighborhood
    first_parts = []
    if "HouseStyle" in public_features and "YearBuilt" in public_features:
        style = public_features["HouseStyle"].replace("Story", "-story")
        year = public_features["YearBuilt"]
        first_parts.append(f"This {style} home was built in {year}")

    if "Neighborhood" in public_features:
        if first_parts:
            first_parts.append(f"in the {public_features['Neighborhood']} neighborhood")
        else:
            first_parts.append(
                f"This home is located in the {public_features['Neighborhood']} neighborhood"
            )

    if first_parts:
        sentences.append(" ".join(first_parts))

    # Second sentence: Size, bedrooms, and bathrooms
    second_parts = []
    if "GrLivArea" in public_features:
        sqft = public_features["GrLivArea"]
        second_parts.append(f"It features {sqft:,} square feet of living space")

    beds = public_features.get("BedroomAbvGr", 0)
    baths = public_features.get("FullBath", 0)
    if beds or baths:
        if second_parts:
            second_parts.append(f"with {beds} bedrooms and {baths} bathrooms")
        else:
            second_parts.append(f"The home has {beds} bedrooms and {baths} bathrooms")

    if second_parts:
        sentences.append(" ".join(second_parts))

    # Third sentence: Basement and garage
    third_parts = []
    if "TotalBsmtSF" in public_features and public_features["TotalBsmtSF"] > 0:
        bsmt_sqft = public_features["TotalBsmtSF"]
        third_parts.append(f"The property includes a {bsmt_sqft:,} sq ft basement")

    if "GarageCars" in public_features:
        cars = public_features["GarageCars"]
        if cars > 0:
            if third_parts:
                third_parts.append(f"and a {int(cars)}-car garage")
            else:
                third_parts.append(f"The property includes a {int(cars)}-car garage")

    if third_parts:
        sentences.append(" ".join(third_parts))

    return ". ".join(sentences) + "." if sentences else ""


def generate_defect_description(defect: dict[str, Any]) -> str:
    """Generate a human-readable description of a property defect.

    Converts a structured defect record into natural language suitable
    for seller disclosures or inspection reports.

    Args:
        defect: Dictionary with keys:
            - feature: str (e.g., "BsmtCond", "HeatingQC")
            - value: str or int (e.g., "Fa", "Po")
            - severity: str ("minor", "moderate", "major", "critical")
            - repair_cost: int (estimated cost in dollars)
            - description: str (optional, pre-formatted description)

    Returns:
        Human-readable defect description as a string.

    Example:
        >>> defect = {
        ...     "feature": "BsmtCond",
        ...     "value": "Fa",
        ...     "severity": "moderate",
        ...     "repair_cost": 3000,
        ... }
        >>> desc = generate_defect_description(defect)
        >>> "basement" in desc.lower()
        True
    """
    # If a description is already provided, use it
    if "description" in defect and defect["description"]:
        return str(defect["description"])

    feature = defect.get("feature", "")
    value = defect.get("value", "")
    severity = defect.get("severity", "unknown")
    repair_cost = defect.get("repair_cost", 0)

    # Feature name mappings to human-readable terms
    feature_names = {
        "BsmtCond": "basement condition",
        "BsmtQual": "basement quality",
        "HeatingQC": "heating system",
        "Electrical": "electrical system",
        "Functional": "home functionality",
        "GarageCond": "garage condition",
        "GarageQual": "garage quality",
        "OverallCond": "overall condition",
    }

    # Condition code mappings
    condition_descriptions = {
        "Po": "poor",
        "Fa": "fair",
        "TA": "typical/average",
        "Gd": "good",
        "Ex": "excellent",
        "FuseF": "60 AMP fuse box (fair)",
        "FuseP": "60 AMP fuse box (poor)",
        "Min2": "minor deductions (2nd type)",
        "Mod": "moderate deductions",
        "Maj1": "major deductions (1st type)",
        "Maj2": "major deductions (2nd type)",
    }

    feature_name = feature_names.get(feature, feature.lower())
    condition_desc = condition_descriptions.get(str(value), str(value))

    # Build description based on severity and feature
    if severity in ["major", "critical"]:
        description = (
            f"The {feature_name} is in {condition_desc} condition, "
            f"representing a {severity} issue. "
            f"Estimated repair cost: ${repair_cost:,}."
        )
    elif severity == "moderate":
        description = (
            f"The {feature_name} is rated as {condition_desc}, "
            f"indicating moderate wear. "
            f"Estimated repair cost: ${repair_cost:,}."
        )
    else:  # minor or unknown
        description = (
            f"The {feature_name} shows {condition_desc} condition. "
            f"Estimated repair cost: ${repair_cost:,}."
        )

    return description
