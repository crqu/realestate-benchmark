"""Ames Housing Dataset loader and property access utilities.

This module provides functions to load the Ames Housing dataset and retrieve
individual property data. The dataset contains residential property sales from
Ames, Iowa (2006-2010) with 80+ features per property.

Example:
    >>> df = load_ames_data()
    >>> print(f"Loaded {len(df)} properties")
    >>> property_data = get_property(1)
    >>> print(property_data['Neighborhood'])
"""

from pathlib import Path
from typing import Any

import pandas as pd


def load_ames_data(path: str | None = None) -> pd.DataFrame:
    """Load the Ames Housing dataset from CSV files.

    This function loads and combines the train.csv and test.csv files from the
    Ames Housing dataset. It handles missing values appropriately and validates
    that expected columns exist.

    Args:
        path: Path to the directory containing train.csv and test.csv.
              If None, uses the default location: data/ames/

    Returns:
        DataFrame containing all properties (train + test combined) with 80+ features.
        Missing values in condition codes are preserved as None.

    Raises:
        FileNotFoundError: If train.csv or test.csv cannot be found.
        ValueError: If required columns are missing from the dataset.

    Example:
        >>> df = load_ames_data()
        >>> print(df.shape)
        (2919, 81)
        >>> print(df['SalePrice'].mean())
        180921.19589041095
    """
    if path is None:
        # Default to data/ames/ relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        path = str(project_root / "data" / "ames")

    path_obj = Path(path)
    train_path = path_obj / "train.csv"
    test_path = path_obj / "test.csv"

    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please download the Ames Housing dataset. "
            f"See {path_obj / 'README.md'} for instructions."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            f"Please download the Ames Housing dataset. "
            f"See {path_obj / 'README.md'} for instructions."
        )

    # Load train and test data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Test data doesn't have SalePrice - fill with None for consistency
    if "SalePrice" not in test_df.columns:
        test_df["SalePrice"] = None

    # Combine datasets
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Validate that expected columns exist
    expected_columns = [
        "Id",
        "MSSubClass",
        "MSZoning",
        "LotFrontage",
        "LotArea",
        "Street",
        "Neighborhood",
        "HouseStyle",
        "YearBuilt",
        "YearRemodAdd",
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "TotalBsmtSF",
        "BsmtFinSF1",
        "GarageCars",
        "GarageArea",
        "GarageType",
        "Fireplaces",
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

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in dataset: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Handle missing values:
    # - For condition codes (quality/condition features), keep as NaN (will convert to None)
    # - This preserves the semantic meaning that the feature is not applicable
    # - pandas will handle NaN appropriately when converting to dict

    # Convert NaN to None for condition code columns to make them JSON-serializable
    condition_columns = [
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "HeatingQC",
        "Electrical",
        "CentralAir",
        "Functional",
        "GarageQual",
        "GarageCond",
        "GarageType",
    ]

    for col in condition_columns:
        if col in df.columns:
            df[col] = df[col].where(pd.notna(df[col]), None)

    return df


def get_property(
    property_id: int | str,
    data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Retrieve a single property's data by ID.

    Args:
        property_id: Property identifier. Can be:
                    - Integer: 0-based index into the dataframe
                    - String: Value to match in the 'Id' column
        data: Pre-loaded DataFrame. If None, will call load_ames_data().

    Returns:
        Dictionary containing all features for the specified property.
        Missing values are represented as None.

    Raises:
        ValueError: If property_id is not found in the dataset.
        IndexError: If integer property_id is out of range.

    Example:
        >>> property = get_property(1)
        >>> print(property['Neighborhood'])
        'CollgCr'
        >>> print(property['SalePrice'])
        208500
    """
    if data is None:
        data = load_ames_data()

    # Handle different ID types
    if isinstance(property_id, int):
        # Treat as 0-based index
        if property_id < 0 or property_id >= len(data):
            raise IndexError(
                f"Property index {property_id} out of range. "
                f"Dataset has {len(data)} properties (valid indices: 0-{len(data) - 1})"
            )
        property_data = data.iloc[property_id]
    else:
        # Treat as ID column value
        matching = data[data["Id"] == property_id]
        if len(matching) == 0:
            raise ValueError(
                f"No property found with Id={property_id}. "
                f"Available IDs: {data['Id'].min()}-{data['Id'].max()}"
            )
        property_data = matching.iloc[0]

    # Convert to dictionary
    # Replace NaN with None for JSON compatibility
    result = property_data.to_dict()
    result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

    return result


def get_dataset_info(data: pd.DataFrame | None = None) -> dict[str, Any]:
    """Get summary information about the Ames Housing dataset.

    Args:
        data: Pre-loaded DataFrame. If None, will call load_ames_data().

    Returns:
        Dictionary with dataset statistics:
        - total_properties: Total number of properties
        - features: List of feature names
        - numeric_features: List of numeric feature names
        - categorical_features: List of categorical feature names
        - missing_data: Dict of features with missing values and counts

    Example:
        >>> info = get_dataset_info()
        >>> print(info['total_properties'])
        2919
    """
    if data is None:
        data = load_ames_data()

    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = data.select_dtypes(include=["object"]).columns.tolist()

    # Find features with missing values
    missing = data.isnull().sum()
    missing_data = {col: int(count) for col, count in missing.items() if count > 0}

    return {
        "total_properties": len(data),
        "features": data.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_data": missing_data,
        "sale_price_mean": float(data["SalePrice"].mean()) if "SalePrice" in data.columns else None,
        "sale_price_median": float(data["SalePrice"].median())
        if "SalePrice" in data.columns
        else None,
    }
