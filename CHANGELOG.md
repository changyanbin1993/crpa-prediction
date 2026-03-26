# CHANGELOG

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-26

### Added
- Initial release of CRPA prediction model
- Complete ML pipeline with 4 main scripts
- Support for 6 machine learning algorithms
- Comprehensive model evaluation metrics
- SHAP-based model interpretation
- Bootstrap confidence intervals
- Cross-validation analysis
- Decision curve analysis
- Calibration curves and Hosmer-Lemeshow test

### Features
- Data preprocessing with missing value handling
- Feature selection using LASSO, RFE, and SHAP
- Model training with hyperparameter optimization
- Extensive visualization outputs
- Reproducibility tracking

### Documentation
- README with installation and usage instructions
- USAGE guide with detailed examples
- CONTRIBUTING guidelines
- MIT License

## Reviewer Response Modifications

### [SE-2] Data Leakage Prevention
- Modified imputation to fit on training set only
- Split data before imputation to prevent information leakage

### [SE-3] SHAP Methodology Clarification
- Clarified SHAP as validation tool, not primary feature selection
- Added feature overlap analysis between methods
- Documented log-odds scale in SHAP plots

### [SE-7] High-Missing Variables
- Added tracking of excluded high-missing variables
- Saved excluded variable information to CSV
- Documented rationale for exclusion threshold

### [SE-8] Model Validation
- Added 10-fold repeated cross-validation
- Saved hyperparameters to JSON files
- Calculated Events-Per-Variable (EPV) ratio

### [SE-9] Reproducibility
- Added software environment logging
- Documented all package versions
- Fixed random seed throughout pipeline

### [SE-5] Additional Metrics
- Added Precision-Recall curves
- Calculated AUPRC with 95% CI

### [SE-6] Threshold Analysis
- Added multi-threshold performance table
- Documented Youden index threshold

### [R1-3] Prevalence Adjustment
- Added PPV/NPV calculation at different prevalence rates
- Used Bayes theorem for adjustment

### [R1-5] Statistical Comparison
- Implemented DeLong test for AUROC comparison
- Added pairwise model comparisons
