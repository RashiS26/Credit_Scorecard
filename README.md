# Credit Risk Scorecard with Explainability

Predict loan default probability using real Lending Club data (890K loans).

## Tech Stack
Python · XGBoost · SHAP · Streamlit · Scikit-learn 

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