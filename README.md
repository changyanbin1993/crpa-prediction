# Clinical Prediction Model for Carbapenem-Resistant Pseudomonas aeruginosa (CRPA)

## Overview

This repository contains the complete machine learning pipeline for predicting carbapenem resistance in invasive *Pseudomonas aeruginosa* infections. The model was developed using clinical data and employs multiple machine learning algorithms with comprehensive validation.

## Project Structure

```
.
├── 01_data_preprocessing.py      # Data preprocessing and feature engineering
├── 02_feature_selection.py       # Feature selection (LASSO, RFE, SHAP)
├── 03_model_training.py          # Model training and evaluation
├── 04_model_interpretation.py    # Model interpretation and analysis
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License information
```

## Features

- **Data Preprocessing**: Handles missing values, removes high-collinearity features, and performs train-test splitting
- **Feature Selection**: Combines LASSO, RFE, and SHAP-based methods with bootstrap stability validation
- **Model Training**: Implements 6 machine learning algorithms (LASSO, Random Forest, AdaBoost, XGBoost, LightGBM, CatBoost)
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, PR curves, calibration curves, and decision curve analysis
- **Model Interpretation**: SHAP analysis, individual case explanations, and sensitivity analysis

## Requirements

- Python 3.8+
- See `requirements.txt` for complete package list

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crpa-prediction.git
cd crpa-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
python 01_data_preprocessing.py
```

This script:
- Loads raw data from `CRPA.csv`
- Removes high-missing features (>10%)
- Splits data into train/test sets (80/20)
- Performs median imputation
- Removes multicollinear features (VIF > 10)
- Outputs: `processed_train.csv`, `processed_test.csv`

### 2. Feature Selection

```bash
python 02_feature_selection.py
```

This script:
- Applies LASSO feature selection
- Performs RFE with Random Forest
- Uses SHAP for validation
- Bootstrap stability analysis (100 iterations)
- Outputs: `final_train_data.csv`, `final_val_data.csv`, `selected_features.txt`

### 3. Model Training

```bash
python 03_model_training.py
```

This script:
- Trains 6 ML models
- Performs 10-fold repeated cross-validation
- Generates ROC and PR curves
- Calculates performance metrics with 95% CI
- Outputs: Model files in `./models/`, performance tables in `./tables/`

### 4. Model Interpretation

```bash
python 04_model_interpretation.py
```

This script:
- Generates calibration curves
- Performs decision curve analysis
- SHAP global and individual analysis
- Hosmer-Lemeshow test
- Outputs: Plots in `./plots/`, tables in `./tables/`

## Model Performance

The models are evaluated using:
- **AUROC** with 95% confidence intervals (bootstrap)
- **AUPRC** (Area Under Precision-Recall Curve)
- **Calibration** (calibration curves, Hosmer-Lemeshow test)
- **Clinical utility** (decision curve analysis)
- **DeLong test** for model comparison

## Key Methodological Features

1. **Prevention of data leakage**: Train-test split before imputation
2. **Robust feature selection**: Multiple methods with bootstrap validation
3. **Comprehensive evaluation**: Multiple metrics and validation approaches
4. **Model interpretability**: SHAP analysis for clinical insights
5. **Reproducibility**: Fixed random seed (42) and environment logging

## Output Files

### Plots (`./plots/`)
- `01_target_distribution.png` - Target variable distribution
- `02a_lasso_analysis.png` - LASSO feature selection
- `03_rfe_curve.png` - RFE cross-validation
- `04_shap_summary_feature_selection.png` - SHAP feature importance
- `05_feature_stability.png` - Bootstrap stability
- `06_roc_curves.png` - ROC curves for all models
- `07_calibration_curves.png` - Calibration curves
- `08_decision_curve_analysis.png` - Decision curve analysis
- `09_shap_summary.png` - SHAP summary plot
- `10_shap_bar.png` - SHAP feature importance bar plot
- `11_shap_case_*.png` - Individual case explanations

### Tables (`./tables/`)
- `model_performance.csv` - Model performance metrics
- `Suppl_AUPRC.csv` - AUPRC values
- `Suppl_CV_Summary.csv` - Cross-validation results
- `Suppl_Table_S5_Thresholds.csv` - Performance at different thresholds
- `Suppl_Table_S6_Prevalence.csv` - PPV/NPV at different prevalence
- `Suppl_Table_S7_DeLong.csv` - DeLong test results
- `Suppl_Hosmer_Lemeshow.csv` - Hosmer-Lemeshow test
- `Suppl_Software_Versions.csv` - Software environment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact 1109605451@qq.com.
