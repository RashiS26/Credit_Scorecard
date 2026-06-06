
import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


st.set_page_config(
    page_title="CreditIQ — Loan Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* Background */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: #6b7280;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1f2937;
}

/* Cards */
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result card */
.result-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-medium {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 1px solid #f59e0b;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-prob {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0;
    line-height: 1;
}
.result-label {
    font-size: 1rem;
    font-weight: 500;
    margin-top: 0.5rem;
    opacity: 0.85;
}

/* Gauge bar */
.gauge-wrap { margin: 1.5rem 0; }
.gauge-track {
    background: #1f2937;
    border-radius: 99px;
    height: 12px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* Slider styling */
.stSlider > div > div > div > div {
    background: #3b82f6 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Input fields */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 2rem 0;
}

/* Tag badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 6px;
}
.badge-blue { background: #1e3a5f; color: #60a5fa; }
.badge-purple { background: #2e1065; color: #a78bfa; }
.badge-green { background: #064e3b; color: #34d399; }

/* SHAP plot background fix */
.stpyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, explainer, features

try:
    model, explainer, features = load_model()
    model_loaded = True
except:
    model_loaded = False



st.markdown("""
<div class="hero">
    <h1>CreditIQ</h1>
    <p>AI-powered loan default risk analyzer · Powered by XGBoost + SHAP Explainability</p>
    <div style="margin-top:1rem;">
        <span class="badge badge-blue">XGBoost</span>
        <span class="badge badge-purple">SHAP</span>
        <span class="badge badge-green">SR 11-7 Compliant</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Make sure xgb_model.pkl, explainer.pkl, and features.pkl are in the same folder as app.py")
    st.stop()

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Borrower Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-label">💰 Loan Info</p>', unsafe_allow_html=True)
    loan_amnt   = st.slider("Loan Amount ($)", 1000, 40000, 10000, step=500)
    int_rate    = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0, step=0.1)
    dti         = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, step=0.5)

with col2:
    st.markdown('<p class="section-label">👤 Borrower Profile</p>', unsafe_allow_html=True)
    annual_inc  = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=60000, step=1000)
    fico        = st.slider("FICO Credit Score", 580, 850, 700)
    open_acc    = st.slider("Open Credit Accounts", 1, 30, 8)

with col3:
    st.markdown('<p class="section-label">📋 Credit History</p>', unsafe_allow_html=True)
    credit_util      = st.slider("Credit Utilization", 0.0, 1.0, 0.3, step=0.01)
    revol_bal        = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=5000, step=500)
    has_derog        = st.selectbox("Major Derogatory Record?", [0, 1], format_func=lambda x: "Yes " if x else "No ")
    has_revol_delinq = st.selectbox("Recent Delinquency?",       [0, 1], format_func=lambda x: "Yes " if x else "No ")
    has_pub_rec      = st.selectbox("Public Record?",            [0, 1], format_func=lambda x: "Yes " if x else "No ")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Analyze Credit Risk")

if predict_clicked:
    loan_to_income = loan_amnt / (annual_inc + 1)

    input_dict = {
        "loan_amnt":        loan_amnt,
        "int_rate":         int_rate,
        "annual_inc":       annual_inc,
        "dti":              dti,
        "fico_range_low":   fico,
        "credit_util":      credit_util,
        "loan_to_income":   loan_to_income,
        "has_derog":        has_derog,
        "has_revol_delinq": has_revol_delinq,
        "has_pub_rec":      has_pub_rec,
        "revol_bal":        revol_bal,
        "open_acc":         open_acc
    }

    input_df = pd.DataFrame([[input_dict.get(f, 0) for f in features]], columns=features)
    prob = model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Risk Assessment Result</p>', unsafe_allow_html=True)

    res_col, detail_col = st.columns([1, 2])

    with res_col:
        if prob < 0.15:
            tier, css, emoji, color = "LOW RISK", "result-low", "✅", "#10b981"
        elif prob < 0.35:
            tier, css, emoji, color = "MEDIUM RISK", "result-medium", "⚠️", "#f59e0b"
        else:
            tier, css, emoji, color = "HIGH RISK", "result-high", "🚨", "#ef4444"

        st.markdown(f"""
        <div class="{css}">
            <div style="font-size:2.5rem">{emoji}</div>
            <p class="result-prob" style="color:{color}">{prob_pct:.1f}%</p>
            <p class="result-label">Default Probability</p>
            <p style="font-family:'Syne',sans-serif;font-weight:700;
                      font-size:0.85rem;letter-spacing:0.1em;
                      margin-top:0.75rem;color:{color}">{tier}</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        gauge_color = color
        st.markdown(f"""
        <div class="gauge-wrap">
            <div style="display:flex;justify-content:space-between;
                        font-size:0.75rem;color:#6b7280;margin-bottom:6px;">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{min(prob_pct,100)}%;
                     background:linear-gradient(90deg,#3b82f6,{gauge_color});">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with detail_col:
        st.markdown('<p class="section-label">Key Risk Factors</p>', unsafe_allow_html=True)

        # Mini scorecard
        metrics = [
            ("FICO Score",        fico,          580,   850,   False),
            ("Debt-to-Income",    dti,           0,     40,    True),
            ("Credit Utilization",credit_util*100,0,   100,    True),
            ("Interest Rate",     int_rate,      5,     30,    True),
            ("Loan-to-Income",    round(loan_to_income,3), 0, 1, True),
        ]

        for label, val, lo, hi, higher_is_worse in metrics:
            norm = (val - lo) / (hi - lo + 1e-9)
            if higher_is_worse:
                bar_color = "#ef4444" if norm > 0.66 else "#f59e0b" if norm > 0.33 else "#10b981"
            else:
                bar_color = "#10b981" if norm > 0.66 else "#f59e0b" if norm > 0.33 else "#ef4444"

            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.82rem;margin-bottom:4px;">
                    <span style="color:#9ca3af">{label}</span>
                    <span style="color:#e8eaf0;font-weight:500">{val}</span>
                </div>
                <div style="background:#1f2937;border-radius:99px;height:6px;overflow:hidden;">
                    <div style="width:{norm*100:.0f}%;height:100%;
                                background:{bar_color};border-radius:99px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🔍 Why this prediction? — SHAP Explanation</p>', unsafe_allow_html=True)

    shap_vals = explainer.shap_values(input_df)

    matplotlib.rcParams.update({
        'figure.facecolor': '#111827',
        'axes.facecolor':   '#111827',
        'text.color':       '#e8eaf0',
        'axes.labelcolor':  '#e8eaf0',
        'xtick.color':      '#9ca3af',
        'ytick.color':      '#9ca3af',
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=features
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Decision summary ──────────────────────────────────────────────────────
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📋 Decision Summary</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Default Probability", f"{prob_pct:.1f}%")
    s2.metric("Risk Tier", tier)
    s3.metric("FICO Score", fico, delta=f"{fico-700} vs avg")
    s4.metric("DTI Ratio", f"{dti:.1f}%", delta=f"{dti-15:.1f}% vs avg",
              delta_color="inverse")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:2rem;
            border-top:1px solid #1f2937;color:#374151;font-size:0.78rem;">
    CreditIQ · Built with XGBoost + SHAP · Lending Club Dataset · 890K loans
</div>
""", unsafe_allow_html=True)

