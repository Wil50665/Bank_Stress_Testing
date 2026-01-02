# Bank Stress Testing

## Project Overview
The objective of this project is to build a data-driven stress testing system that helps financial institutions assess potential losses under adverse economic conditions. The framework integrates historical macroeconomic indicators with loan loss data to forecast portfolio performance and evaluate regulatory compliance, including Tier 1 Capital adequacy.

## Key Features
- **Portfolio Management**  
  Analysis of four distinct portfolios:
  - Residential Mortgages  
  - Commercial Loans  
  - Credit Cards  
  - Securities  

- **Advanced Modeling**  
  Comparison of baseline **ARIMA** models with **ARIMAX** models that incorporate economic indicators.

- **Scenario Analysis**  
  Implementation of four standard stress scenarios:
  - Economic Recession (Severe)  
  - Interest Rate Shock (Moderate)  
  - Market Volatility Crisis (Severe)  
  - Mild Stress (Baseline)  

- **Interactive Dashboard**  
  Streamlit-based interface for visualizing forecasts, losses, and capital impacts.

## Methodology

### 1. Data Understanding & Preparation
Data is sourced from a PostgreSQL database containing historical economic indicators and loan loss data. Portfolio-specific features are selected based on economic relevance and empirical relationships.

### 2. Modeling
Walk-Forward Validation using `TimeSeriesSplit` is applied to ensure robust out-of-sample performance and realistic forecasting.

- **ARIMAX Performance**  
  Incorporating exogenous economic variables significantly improved predictive accuracy.  
  For example, the **Credit Cards** portfolio achieved approximately a **68% reduction in RMSE** when using ARIMAX compared to standard ARIMA.

### 3. Evaluation
Models are evaluated using:
- Residual diagnostics (ACF/PACF plots)
- Shapiro–Wilk test for normality
- Ljung–Box test for autocorrelation  

These diagnostics ensure forecast stability and reliability.

## Findings & Recommendations
- **Capital Impact**  
  Severe recession scenarios materially reduce Tier 1 capital through declines in retained earnings.

- **Model Sensitivity**  
  Portfolios such as Credit Cards exhibit high sensitivity to unemployment shocks and GDP fluctuations.

- **Action Plan**  
  The framework identifies scenarios that breach regulatory capital thresholds, signaling the need for capital contingency planning.
