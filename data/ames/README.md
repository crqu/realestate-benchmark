# Ames Housing Dataset

## Download Instructions

The Ames Housing dataset is required for this benchmark. You must download it manually from Kaggle.

### Required: Kaggle Download

1. **Create a Kaggle account** (if you don't have one):
   - Visit https://www.kaggle.com and sign up

2. **Download the dataset**:
   - Visit https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
   - Click "Download All" button
   - Extract the downloaded zip file

3. **Place files in this directory**:
   ```bash
   # After extracting, copy files to this location:
   cp train.csv data/ames/
   cp test.csv data/ames/
   cp data_description.txt data/ames/
   ```

### Optional: Kaggle CLI (Advanced)
```bash
# Install kaggle CLI
pip install kaggle

# Setup API credentials (see https://github.com/Kaggle/kaggle-api#api-credentials)
# Then download:
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/ames/
```

### Expected Files

After download, this directory should contain:
```
data/ames/
├── train.csv             # Training data (1,460 properties)
├── test.csv              # Test data (1,459 properties)  
├── data_description.txt  # Feature documentation
└── README.md             # This file
```

## Dataset Overview

- **Source**: Ames, Iowa residential sales (2006-2010)
- **Properties**: 2,919 total (1,460 training + 1,459 test)
- **Features**: 80+ variables per property
- **Target**: SalePrice (actual transaction price, available in training set)
- **Original Publication**: Dean De Cock (2011), "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project", Journal of Statistics Education
- **Kaggle Competition**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

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

Properties with certain condition codes or functional issues are considered to have defects requiring repair. The benchmark uses these repair cost estimates:

| Feature | Condition/Value | Severity | Estimated Repair Cost |
|---------|----------------|----------|----------------------|
| BsmtCond | Fa | Moderate | $3,000 |
| BsmtCond | Po | Major | $8,500 |
| BsmtQual | Fa or Po | Moderate | $4,000 |
| HeatingQC | Fa | Moderate | $2,500 |
| HeatingQC | Po | Major | $4,500 |
| Electrical | FuseF | Moderate | $3,500 |
| Electrical | FuseP | Major | $5,000 |
| Functional | Min2 | Minor | $4,000 |
| Functional | Mod | Moderate | $8,000 |
| Functional | Maj1 | Major | $15,000 |
| Functional | Maj2 | Critical | $25,000 |
| GarageCond | Fa or Po | Moderate | $3,000 |
| OverallCond | ≤4 (when OverallQual ≥6) | Major | $12,000 |

**Note**: The true value of a property is calculated as `SalePrice - Σ(repair costs)` for all detected defects.
