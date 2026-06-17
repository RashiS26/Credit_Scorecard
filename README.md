# Credit Risk Scorecard with Explainability
To build a machine learning system that predicts the probability of a borrower defaulting on a loan, using real-world Lending Club data, while ensuring every prediction is explainable and compliant with banking regulatory standards.

---

## Features
* **Real-world large-scale data** — trained on 890,000 actual Lending Club loan records with 150 features, not synthetic data.
* **Robust ML pipeline with imbalance handling** — full cleaning + feature engineering pipeline, with SMOTE applied to fix the 80/20 class imbalance between non-defaulters and defaulters.
* **Multi-model benchmarking** — compares Logistic Regression, Random Forest, and XGBoost using industry-standard metrics (AUC-ROC, KS Statistic), with XGBoost selected as the best performer (AUC ~0.91).
* **Explainable AI with SHAP** — every prediction comes with both global feature importance and per-applicant waterfall explanations, aligned with SR 11-7 regulatory standards.
* **Interactive deployed app** — a Streamlit-based loan decision simulator where users input borrower details and instantly get a default probability, risk tier, and visual explanation.
---

## Tech Stack
* **Python**: The core programming language used for the analysis.
* **Jupyter Notebook**: For interactive data analysis and visualization.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Matplotlib & Seaborn**: For data visualization.
* **Scikit-learn**: Train-test splitting, Logistic Regression model, Random Forest model, and evaluation metrics (AUC, classification report).
* **XGBoost**: The main predictive model — gradient boosted trees that achieved the best AUC and KS score among all models tested.
* **Imbalanced-learn (SMOTE)**: Fixing class imbalance by generating synthetic minority-class (defaulter) samples for training.
* **SHAP**: Explainability layer — calculates how much each feature contributed to each individual prediction.
* **Streamlit**: Building and deploying the interactive web application — the loan decision simulator UI.

---

## Project Structure
- `src/` — Jupyter notebooks for each phase + Streamlit app
- `output/models/` — Trained model files
- `output/plots/` — Generated charts and visualizations

## How to Run
pip install -r requirements.txt
streamlit run src/app.py

## Results
- AUC-ROC: ~0.91
- KS Statistic: ~0.45
- Models compared: Logistic Regression, Random Forest, XGBoost

## Dataset
Dataset not included due to GitHub size limits.
Download it from:
<https://www.kaggle.com/datasets/wordsforthewise/lending-club>
