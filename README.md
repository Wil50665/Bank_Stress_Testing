# Bank Stress Testing

Project Overview
The objective of this project is to build a data-driven system that helps financial institutions understand potential losses in adverse economic conditions. It integrates historical economic data with loan loss history to forecast future performance and ensure regulatory compliance (e.g., Tier 1 Capital ratios).

Key Features
  Portfolio Management: Analysis of four distinct portfolios: Residential Mortgages, Commercial Loans, Credit Cards, and Securities.
  Advanced Modeling: Comparison between baseline ARIMA and economic-indicator-driven ARIMAX models.
  Scenario Analysis: Implementation of four standard stress scenarios:
    - Economic Recession (Severe)
    - Interest Rate Shock (Moderate)
    - Market Volatility Crisis (Severe)
    - Mild Stress (Baseline)
  Interactive Dashboard: A Streamlit-based front-end for visualizing forecasts and capital impact.

Methodology
1. Data Understanding & Preparation
  Data is pulled from a PostgreSQL database including historical economic indicators and historical loan losses. Features are selected per portfolio based on economic relevance.

2. Modeling
  We utilize Walk-Forward Validation (TimeSeriesSplit) to ensure models generalize well over time.

  ARIMAX Performance: The inclusion of exogenous economic variables improved prediction accuracy significantly. For example, the Credit Cards portfolio saw a ~68% improvement in RMSE using ARIMAX over standard ARIMA.

3. Evaluation
  Models are evaluated using residual diagnostics (ACF/PACF plots, Shapiro-Wilk test for normality, and Ljung-Box test for autocorrelation) to ensure forecast reliability.

Findings & Recommendations
Capital Impact: Severe recession scenarios significantly reduce Tier 1 capital via retained earnings.

Model Sensitivity: Portfolios like Credit Cards are highly sensitive to unemployment spikes and GDP fluctuations.

Action Plan: Framework identifies when scenarios breach regulatory minimums, triggering the need for capital contingency plans.
