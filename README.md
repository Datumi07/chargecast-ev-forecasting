# ChargeCast: EV Charging Session Forecasting

**DATA2010 – Data Science Programming | VinUniversity**

## Project Overview

ChargeCast is a data science project focused on forecasting the number of EV charging sessions at individual public charging points for the following day. The project investigates whether machine learning models trained across multiple charging points can improve forecasting accuracy, particularly when individual locations have limited historical data.

The project uses publicly available charging records from Scotland. Its findings may provide useful methodological insights for EV charging infrastructure management, including future applications in Vietnam.

## Research Questions

1. Can machine learning models outperform simple historical baselines in predicting next-day charging sessions?
2. Can a shared model trained across multiple charging points outperform point-specific models when only 30 or 90 days of historical data are available?
3. How does forecasting accuracy change during high-usage days, particularly when actual demand is underestimated?

## Dataset

**Source:** [ChargePlace Scotland – Monthly Charge Point Performance](https://chargeplacescotland.org/monthly-charge-point-performance/)

- **Proposed period:** January 2025 – September 2026
- **Initial scope:** 20–50 charging points
- **Estimated size:** 12,760–31,900 charging point-day observations before filtering
- **Target variable:** Number of charging sessions starting at each charging point on a given day
- **Predictors:** Calendar information, historical session counts, rolling statistics, recent trends, and charging point identifiers

Dataset availability, completeness, and final dimensions will be verified during data preparation.

## Proposed Methodology

The project follows the Data Science Life Cycle:

1. Data acquisition and validation
2. Data cleaning and preprocessing
3. Exploratory data analysis and visualisation
4. Feature engineering
5. Predictive modelling and baseline comparison
6. Chronological validation and performance evaluation
7. Development of a reproducible forecasting pilot

**Candidate models:** Historical baselines, Poisson Regression, and LightGBM.

**Evaluation metrics:** MAE, RMSE, and analysis of high-usage underprediction.

## Expected Outcomes

- A reproducible Python-based forecasting pipeline
- Comparisons between baseline and machine learning models
- Evaluation of shared and point-specific models under limited historical data
- Visualisations of charging usage patterns and prediction errors
- A working pilot for generating and evaluating next-day forecasts

## Team Members

- Dang Tuan Minh
- Nguyen Quy Tu
- Nguyen Dinh Nam Khanh
- Ngo Van Thang
- Bui Anh Minh

*Individual responsibilities will be finalised during project planning.*

## Project Status

**Phase 1 – Project Proposal**

The research questions, proposed dataset, and initial methodology have been defined. Data verification and implementation are planned for subsequent phases.

## Academic Context

This repository is maintained as part of the DATA2010 Data Science Programming course at VinUniversity.
