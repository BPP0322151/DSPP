Student Performance Prediction - README.md
# 🎓 Student Performance Prediction: Binary Classification with Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Quality](https://img.shields.io/badge/Data%20Quality-100%25-brightgreen.svg)](https://www.gov.uk/government/publications/the-government-data-quality-framework)

&gt; **Predicting student pass/fail outcomes using behavioral and demographic features to enable early intervention strategies**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Technical Architecture](#-technical-architecture)
- [Installation &amp; Setup](#-installation--setup)
- [Data Engineering Approach](#-data-engineering-approach)
- [Model Development &amp; Rationale](#-model-development--rationale)
- [Results &amp; Performance](#-results--performance)
- [Key Insights &amp; Business Value](#-key-insights--business-value)
- [Future Improvements](#-future-improvements)
- [References](#-references)

---

## 🎯 Project Overview

This project demonstrates end-to-end data science methodology by developing a **binary classification model** to predict whether high school students will pass (Grades A-C) or fail (Grades D-F) based on 13 behavioral and demographic features.

### Key Metrics
- **Accuracy**: 89.1%
- **Balanced Accuracy**: 87%
- **ROC-AUC**: 0.93
- **F1-Score**: 83%
- **Precision (Fail Detection)**: 84%

### Dataset
- **Source**: [Kaggle - Student Performance Data](https://www.kaggle.com/)
- **Size**: 2,392 student records
- **Features**: 13 predictors (demographics, study habits, parental involvement, extracurricular activities)
- **Target**: Binary pass/fail outcome (engineered from 5-tier grade classification)

---

## 💼 Business Problem

### Challenge
Educational institutions often adopt **reactive** rather than **proactive** approaches to student support—intervening only after poor performance becomes evident through failing grades. This results in:

- Wasted tutoring resources on students unlikely to benefit
- Missed opportunities for early intervention with at-risk students
- Inability to quantify which behavioral factors most influence academic outcomes

### Solution
A **logistic regression classifier** that:
1. **Identifies at-risk students** before end-of-term assessments
2. **Quantifies feature importance** to guide targeted interventions (e.g., addressing absences vs. study time)
3. **Provides probability scores** (0-1) for nuanced risk stratification

### Impact
- **Early identification**: Flag at-risk students in Week 4-6 of term (not Week 12)
- **Resource optimization**: Allocate tutoring to students with &gt;70% predicted failure probability
- **Data-driven policy**: Evidence that reducing absences has 3x more impact than increasing study time

---

## 🏗️ Technical Architecture

### Why Python Ecosystem?

| Tool | Purpose | Rationale |
|------|---------|-----------|
| **Pandas** | Data manipulation | Industry-standard for tabular data; `DataFrame` structure intuitive for feature engineering |
| **Scikit-learn** | Model training &amp; evaluation | Consistent API (`fit`, `predict`, `transform`); extensive metrics library; production-ready pipelines |
| **Statsmodels** | Statistical inference | Provides p-values &amp; confidence intervals for hypothesis testing (not available in sklearn) |
| **Seaborn/Matplotlib** | Visualization | Publication-quality plots; heatmaps ideal for correlation analysis |

### Model Choice: Logistic Regression

**Why not other algorithms?**

| Model | Rejected Reason |
|-------|----------------|
| **Linear Regression** | Requires continuous target variable (we have binary pass/fail) |
| **Random Forest** | Black-box model; stakeholders (educators/administrators) require **interpretable** coefficients to justify interventions |
| **Neural Networks** | Overkill for tabular data with 13 features; overfitting risk with only 2,392 samples |

**Why Logistic Regression?**
1. **Interpretable coefficients**: Each feature's impact on failure risk is quantifiable (e.g., "1 additional absence increases failure odds by 12%")
2. **Probability outputs**: Returns P(Fail) ∈ [0,1], not just binary predictions—enables risk tiers (low/medium/high)
3. **Statistical rigor**: Compatible with hypothesis testing (p-values, confidence intervals via statsmodels)
4. **Computational efficiency**: Trains in &lt;1 second; suitable for real-time dashboards

### Trade-offs Acknowledged
- **Limitation**: Assumes linear relationship between features and log-odds of failure
- **Future Work**: Compare with Gradient Boosting (XGBoost) for potential accuracy gains while maintaining feature importance explainability
