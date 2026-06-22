# Ames Housing Dataset

## Download Instructions

The Ames Housing dataset is required for this benchmark. Download it from one of these sources:

### Option 1: Kaggle
```bash
# Install kaggle CLI
pip install kaggle

# Download (requires Kaggle API credentials)
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d .
```

### Option 2: Direct Download
Download from the original source:
- [Ames Housing Data Documentation](http://jse.amstat.org/v19n3/decock.pdf)
- [Data Files](http://jse.amstat.org/v19n3/decock/AmesHousing.txt)

### Expected Files

After download, this directory should contain:
```
data/ames/
├── train.csv          # Training data (1,460 properties)
├── test.csv           # Test data (1,459 properties)  
└── data_description.txt  # Feature documentation
```

## Dataset Overview

- **Source**: Ames, Iowa residential sales (2006-2010)
- **Properties**: 2,930 total
- **Features**: 80+ variables per property
- **Target**: SalePrice (actual transaction price)

## Key Features for Benchmark

### Public Features (Visible to Buyers)
- `Neighborhood` - Physical location
- `HouseStyle` - Style of dwelling
- `YearBuilt` - Original construction date
- `GrLivArea` - Above grade living area (sq ft)
- `BedroomAbvGr` - Number of bedrooms
- `FullBath`, `HalfBath` - Bathrooms
- `GarageCars` - Garage capacity
- `SalePrice` - Used as asking price

### Hidden Features (Known Only to Sellers)
- `OverallQual`, `OverallCond` - Overall quality/condition ratings
- `BsmtQual`, `BsmtCond` - Basement quality/condition
- `HeatingQC` - Heating quality
- `Electrical` - Electrical system type
- `Functional` - Home functionality rating
- `GarageQual`, `GarageCond` - Garage quality/condition

## Condition Codes

| Code | Meaning |
|------|---------|
| Ex | Excellent |
| Gd | Good |
| TA | Typical/Average |
| Fa | Fair |
| Po | Poor |
| NA | Not Applicable |

## Defect Mapping

Properties with condition codes of `Fa` or `Po` are considered to have defects requiring repair:

| Feature | Condition | Estimated Repair Cost |
|---------|-----------|----------------------|
| BsmtCond | Fa | $3,000 |
| BsmtCond | Po | $8,500 |
| HeatingQC | Fa | $2,500 |
| HeatingQC | Po | $4,500 |
| Electrical | FuseF | $3,500 |
| Electrical | FuseP | $5,000 |
| Functional | Mod | $8,000 |
| Functional | Maj1 | $15,000 |
| Functional | Maj2 | $25,000 |
