import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import altair as alt
from datetime import datetime
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Heart Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Design tokens & global styling
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0A1420;
    --bg-panel: #101D2E;
    --bg-panel-alt: #0D1826;
    --border-soft: #1F3040;
    --text-primary: #EDF2F7;
    --text-muted: #7E93A7;
    --accent: #2DD4BF;
    --accent-dark: #14B8A6;
    --risk-low: #10B981;
    --risk-mid: #F59E0B;
    --risk-high: #F43F5E;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-primary); color: var(--text-primary); }

.hero-wrap {
    padding: 8px 0 24px 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 24px;
}
.hero-eyebrow { font-size: 14px; color: var(--accent); font-weight: 600; margin-bottom: 6px; }
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 38px; font-weight: 700; color: var(--text-primary); line-height: 1.15; margin: 0;
}
.hero-subtitle { font-size: 15px; color: var(--text-muted); max-width: 680px; margin-top: 10px; line-height: 1.5; }

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: var(--text-primary); margin: 4px 0 14px 0;
}
.page-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;
}
.page-sub { font-size: 14px; color: var(--text-muted); margin-bottom: 20px; }

section[data-testid="stSidebar"] { background: var(--bg-panel-alt); border-right: 1px solid var(--border-soft); }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] li { color: var(--text-muted); font-size: 13.5px; line-height: 1.6; }
.sidebar-card { background: var(--bg-panel); border: 1px solid var(--border-soft); border-radius: 10px; padding: 16px; margin-bottom: 14px; }
.sidebar-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px; color: var(--text-primary); margin-bottom: 6px; }
.legend-row { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); margin: 4px 0; }
.legend-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

button[data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; color: var(--text-muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent); }
div[data-baseweb="tab-border"] { background-color: var(--border-soft); }

div[data-testid="stForm"] { background: var(--bg-panel); border: 1px solid var(--border-soft); border-radius: 14px; padding: 28px 30px 20px 30px; }
label, .stSelectbox label, .stSlider label, .stRadio label { color: var(--text-muted) !important; font-size: 13.5px !important; font-weight: 500 !important; }

div.stButton > button, button[kind="formSubmit"] {
    background: linear-gradient(90deg, var(--accent-dark) 0%, var(--accent) 100%);
    color: #05221D; font-weight: 600; font-size: 15.5px; border-radius: 10px; border: none;
    padding: 12px 0px; width: 100%; transition: filter 0.15s ease;
}
div.stButton > button:hover, button[kind="formSubmit"]:hover { filter: brightness(1.08); }

div[data-testid="stDownloadButton"] > button {
    background: var(--bg-panel-alt); color: var(--text-primary); border: 1px solid var(--border-soft);
    border-radius: 10px; font-weight: 500;
}
div[data-testid="stDownloadButton"] > button:hover { border-color: var(--accent); color: var(--accent); }

.result-card { border-radius: 14px; padding: 26px 28px; margin-top: 6px; border: 1px solid var(--border-soft); }
.result-card.high { background: linear-gradient(135deg, rgba(244,63,94,0.14), rgba(244,63,94,0.04)); border-color: rgba(244,63,94,0.35); }
.result-card.low { background: linear-gradient(135deg, rgba(16,185,129,0.14), rgba(16,185,129,0.04)); border-color: rgba(16,185,129,0.35); }
.result-eyebrow { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; }
.result-verdict { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; }
.result-verdict.high { color: var(--risk-high); }
.result-verdict.low { color: var(--risk-low); }

.gauge-wrap { margin-top: 18px; }
.gauge-track { width: 100%; height: 10px; background: var(--bg-panel-alt); border: 1px solid var(--border-soft); border-radius: 999px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 999px; }
.gauge-caption { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--text-muted); margin-top: 6px; }

div[data-testid="stMetric"] { background: var(--bg-panel-alt); border: 1px solid var(--border-soft); border-radius: 10px; padding: 14px 16px 10px 16px; }
div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--text-primary) !important; }

div[data-testid="stExpander"] { background: var(--bg-panel-alt); border: 1px solid var(--border-soft); border-radius: 10px; }
div[data-testid="stFileUploaderDropzone"] { background: var(--bg-panel-alt); border: 1px dashed var(--border-soft); border-radius: 10px; }

.badge {
    display: inline-block; background: rgba(45,212,191,0.14); color: var(--accent);
    border: 1px solid rgba(45,212,191,0.35); padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; margin-left: 8px;
}
.footnote { color: var(--text-muted); font-size: 12.5px; margin-top: 18px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load model, scaler, columns
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("LogisticRegression_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load model files. Make sure LogisticRegression_heart.pkl, "
        f"scaler.pkl and columns.pkl are in the same folder as app.py.\n\nError: {e}"
    )
    st.stop()

CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
RAW_COLS = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
            "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

CHEST_PAIN_MAP = {
    "ATA (Atypical Angina)": "ATA", "NAP (Non-Anginal Pain)": "NAP",
    "TA (Typical Angina)": "TA", "ASY (Asymptomatic)": "ASY",
}
ECG_MAP = {
    "Normal": "Normal", "ST (ST-T abnormality)": "ST",
    "LVH (Left Ventricular Hypertrophy)": "LVH",
}

STATIC_METRICS = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": 0.8696, "Precision": 0.9109, "Recall": 0.8598, "F1": 0.8846},
    {"Model": "KNN", "Accuracy": 0.8641, "Precision": 0.8942, "Recall": 0.8692, "F1": 0.8815},
    {"Model": "Naive Bayes", "Accuracy": 0.8478, "Precision": 0.9158, "Recall": 0.8131, "F1": 0.8614},
    {"Model": "SVM", "Accuracy": 0.8478, "Precision": 0.8835, "Recall": 0.8505, "F1": 0.8667},
    {"Model": "Decision Tree", "Accuracy": 0.7717, "Precision": 0.8351, "Recall": 0.7570, "F1": 0.7941},
])

# ============================================================
# Helpers
# ============================================================
def encode_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode + align columns exactly like training (drop_first=True)."""
    encoded = pd.get_dummies(df_raw, columns=CAT_COLS, drop_first=True)
    return encoded.reindex(columns=model_columns, fill_value=0)


def predict_frame(df_raw: pd.DataFrame):
    """Returns (predictions, risk_pct array) for a raw-schema dataframe."""
    encoded = encode_frame(df_raw)
    scaled = scaler.transform(encoded)
    preds = model.predict(scaled)
    proba = model.predict_proba(scaled)[:, 1] * 100
    return preds, proba, scaled


def risk_color(pct: float) -> str:
    if pct < 30:
        return "#10B981"
    elif pct < 60:
        return "#F59E0B"
    return "#F43F5E"


def generate_pdf(raw_input: dict, risk_pct: float, no_risk_pct: float, prediction: int) -> bytes:
    labels = {
        "Age": "Age", "Sex": "Sex", "ChestPainType": "Chest Pain Type",
        "RestingBP": "Resting BP (mm Hg)", "Cholesterol": "Cholesterol (mg/dl)",
        "FastingBS": "Fasting Blood Sugar > 120", "RestingECG": "Resting ECG",
        "MaxHR": "Max Heart Rate", "ExerciseAngina": "Exercise Angina",
        "Oldpeak": "Oldpeak (ST depression)", "ST_Slope": "ST Slope",
    }
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 30, 40)
    pdf.cell(0, 12, "Heart Disease Risk Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 130, 140)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    verdict = "ELEVATED RISK OF HEART DISEASE" if prediction == 1 else "LOW RISK OF HEART DISEASE"
    color = (229, 72, 77) if prediction == 1 else (16, 150, 90)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, verdict, ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 70, 80)
    pdf.cell(0, 8, f"Estimated risk score: {risk_pct:.1f}%   |   No-disease likelihood: {no_risk_pct:.1f}%", ln=True)
    pdf.ln(6)

    pdf.set_draw_color(210, 210, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 30, 40)
    pdf.cell(0, 8, "Patient Details", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 60, 70)
    for key, label in labels.items():
        pdf.cell(0, 7, f"{label}: {raw_input[key]}", ln=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(0, 5, "This report is generated by a machine learning model for educational "
                          "purposes only and is not a substitute for professional medical diagnosis.")

    output = pdf.output()
    return bytes(output)


def render_gauge(risk_pct: float):
    color = risk_color(risk_pct)
    st.markdown(f"""
    <div class="gauge-wrap">
        <div class="gauge-track">
            <div class="gauge-fill" style="width:{risk_pct:.1f}%; background: linear-gradient(90deg, {color}99, {color});"></div>
        </div>
        <div class="gauge-caption">
            <span>0%</span><span>{risk_pct:.1f}% estimated risk</span><span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def contribution_chart(scaled_row: np.ndarray, top_n: int = 10):
    coefs = model.coef_[0]
    contrib = coefs * scaled_row
    df = pd.DataFrame({"Feature": model_columns, "Contribution": contrib})
    df["AbsContribution"] = df["Contribution"].abs()
    df = df.sort_values("AbsContribution", ascending=False).head(top_n)
    df["Direction"] = df["Contribution"].apply(lambda x: "Pushes risk up" if x > 0 else "Pushes risk down")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Contribution to risk score"),
            y=alt.Y("Feature:N", sort="-x" if False else None, title=None,
                    axis=alt.Axis(labelLimit=200)),
            color=alt.Color("Direction:N",
                             scale=alt.Scale(domain=["Pushes risk up", "Pushes risk down"],
                                              range=["#F43F5E", "#10B981"]),
                             legend=alt.Legend(title=None, orient="top")),
            tooltip=["Feature", alt.Tooltip("Contribution:Q", format=".3f")],
        )
        .properties(height=32 * len(df))
        .configure(background="transparent")
        .configure_axis(labelColor="#EDF2F7", titleColor="#EDF2F7", gridColor="#1F3040", domainColor="#1F3040")
        .configure_legend(labelColor="#EDF2F7", titleColor="#EDF2F7")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def global_importance_chart():
    coefs = model.coef_[0]
    df = pd.DataFrame({"Feature": model_columns, "Coefficient": coefs})
    df["AbsCoefficient"] = df["Coefficient"].abs()
    df = df.sort_values("AbsCoefficient", ascending=False)
    df["Direction"] = df["Coefficient"].apply(lambda x: "Increases risk" if x > 0 else "Decreases risk")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Coefficient:Q", title="Standardized coefficient"),
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            color=alt.Color("Direction:N",
                             scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                              range=["#F43F5E", "#10B981"]),
                             legend=alt.Legend(title=None, orient="top")),
            tooltip=["Feature", alt.Tooltip("Coefficient:Q", format=".3f")],
        )
        .properties(height=32 * len(df))
        .configure(background="transparent")
        .configure_axis(labelColor="#EDF2F7", titleColor="#EDF2F7", gridColor="#1F3040", domainColor="#1F3040")
        .configure_legend(labelColor="#EDF2F7", titleColor="#EDF2F7")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def add_history(entry: dict):
    st.session_state.history.insert(0, entry)


# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "extra_models" not in st.session_state:
    st.session_state.extra_models = {}

# ============================================================
# Hero
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">🫀 Cardiac risk model</div>
    <p class="hero-title">Heart Disease Risk Assessment</p>
    <p class="hero-subtitle">
        Enter a patient's clinical values to estimate heart disease risk, score a whole
        batch of patients at once, compare model performance, or review past predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    page = st.radio(
        "Navigate",
        ["🔍 Single Prediction", "📁 Batch Prediction", "📊 Model Comparison", "🕘 History"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown('<div class="sidebar-card">'
                '<div class="sidebar-title">About this tool</div>'
                'Trained on the UCI Heart Failure Prediction dataset (918 records). '
                'Logistic Regression was selected as the deployed model for the best '
                'balance of accuracy and generalization.'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">'
                '<div class="sidebar-title">Deployed model performance</div>'
                '<div class="legend-row"><span>Accuracy</span><span style="margin-left:auto;color:#EDF2F7">86.9%</span></div>'
                '<div class="legend-row"><span>Precision</span><span style="margin-left:auto;color:#EDF2F7">91.1%</span></div>'
                '<div class="legend-row"><span>Recall</span><span style="margin-left:auto;color:#EDF2F7">86.0%</span></div>'
                '<div class="legend-row"><span>F1-score</span><span style="margin-left:auto;color:#EDF2F7">88.5%</span></div>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">'
                '<div class="sidebar-title">Risk bands</div>'
                '<div class="legend-row"><span class="legend-dot" style="background:#10B981"></span>Low — under 30%</div>'
                '<div class="legend-row"><span class="legend-dot" style="background:#F59E0B"></span>Moderate — 30 to 60%</div>'
                '<div class="legend-row"><span class="legend-dot" style="background:#F43F5E"></span>High — above 60%</div>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<p class="footnote">⚠️ Educational tool only. Not a substitute for '
                'professional medical diagnosis.</p>', unsafe_allow_html=True)

# ============================================================
# PAGE: Single Prediction
# ============================================================
if page == "🔍 Single Prediction":
    with st.form("prediction_form"):
        tab1, tab2, tab3 = st.tabs(["👤  Demographics", "🩺  Vitals", "📈  ECG & Stress Test"])

        with tab1:
            st.markdown('<p class="section-label">Basic patient information</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", min_value=18, max_value=100, value=50)
            with c2:
                sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            chest_pain = st.selectbox("Chest pain type", list(CHEST_PAIN_MAP.keys()))

        with tab2:
            st.markdown('<p class="section-label">Blood pressure & metabolic markers</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                resting_bp = st.slider("Resting blood pressure (mm Hg)", min_value=80, max_value=220, value=120)
                cholesterol = st.slider("Cholesterol (mg/dl)", min_value=0, max_value=650, value=200)
            with c2:
                fasting_bs = st.radio("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True)
                max_hr = st.slider("Max heart rate achieved", min_value=60, max_value=220, value=150)

        with tab3:
            st.markdown('<p class="section-label">Exercise & ECG findings</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                resting_ecg = st.selectbox("Resting ECG", list(ECG_MAP.keys()))
                exercise_angina = st.radio("Exercise-induced angina", ["No", "Yes"], horizontal=True)
            with c2:
                oldpeak = st.slider("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
                st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Predict heart disease risk")

    if submitted:
        raw_input = {
            "Age": age, "Sex": "M" if sex == "Male" else "F",
            "ChestPainType": CHEST_PAIN_MAP[chest_pain], "RestingBP": resting_bp,
            "Cholesterol": cholesterol, "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "RestingECG": ECG_MAP[resting_ecg], "MaxHR": max_hr,
            "ExerciseAngina": "Y" if exercise_angina == "Yes" else "N",
            "Oldpeak": oldpeak, "ST_Slope": st_slope,
        }
        input_df = pd.DataFrame([raw_input])
        preds, probs, scaled = predict_frame(input_df)
        prediction = int(preds[0])
        risk_pct = float(probs[0])
        no_risk_pct = 100 - risk_pct

        st.session_state.last_result = {
            "raw_input": raw_input, "prediction": prediction,
            "risk_pct": risk_pct, "no_risk_pct": no_risk_pct,
            "scaled": scaled[0], "timestamp": datetime.now(),
        }
        add_history({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Type": "Single", "Details": f"Age {age}, {sex}, {CHEST_PAIN_MAP[chest_pain]}",
            "Risk %": round(risk_pct, 1),
            "Verdict": "High risk" if prediction == 1 else "Low risk",
        })

    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        level = "high" if res["prediction"] == 1 else "low"
        verdict_text = "Elevated risk of heart disease" if res["prediction"] == 1 else "Low risk of heart disease"

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card {level}">
            <div class="result-eyebrow">Prediction result</div>
            <div class="result-verdict {level}">{verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)
        render_gauge(res["risk_pct"])

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            st.metric("No heart disease", f"{res['no_risk_pct']:.1f}%")
        with c2:
            st.metric("Heart disease", f"{res['risk_pct']:.1f}%")
        with c3:
            pdf_bytes = generate_pdf(res["raw_input"], res["risk_pct"], res["no_risk_pct"], res["prediction"])
            st.download_button(
                "📄  Download PDF report", data=pdf_bytes,
                file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", use_container_width=True,
            )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">What influenced this prediction</p>', unsafe_allow_html=True)
        contribution_chart(res["scaled"])

        with st.expander("🧾  See input summary"):
            summary_df = pd.DataFrame([res["raw_input"]]).T.rename(columns={0: "Value"}).astype(str)
            st.dataframe(summary_df, width="stretch")

        st.markdown(
            "<p class='footnote'>⚠️ This prediction is generated by a machine learning model "
            "for educational purposes only. Please consult a certified medical professional "
            "for actual diagnosis.</p>", unsafe_allow_html=True,
        )

# ============================================================
# PAGE: Batch Prediction
# ============================================================
elif page == "📁 Batch Prediction":
    st.markdown('<p class="page-heading">Batch Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Upload a CSV with one row per patient. Required columns: '
        + ", ".join(RAW_COLS) + '.</p>', unsafe_allow_html=True,
    )

    uploaded_csv = st.file_uploader("Upload patient data (CSV)", type=["csv"])

    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)
            missing = [c for c in RAW_COLS if c not in batch_df.columns]
            if missing:
                st.error(f"Your file is missing required columns: {', '.join(missing)}")
            else:
                work_df = batch_df[RAW_COLS].copy()
                preds, probs, _ = predict_frame(work_df)
                result_df = batch_df.copy()
                result_df["Predicted"] = np.where(preds == 1, "High risk", "Low risk")
                result_df["Risk %"] = np.round(probs, 1)
                st.session_state.batch_results = result_df

                add_history({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Type": "Batch", "Details": f"{len(result_df)} patients uploaded",
                    "Risk %": round(float(np.mean(probs)), 1),
                    "Verdict": f"{int((preds == 1).sum())} flagged high risk",
                })
        except Exception as e:
            st.error(f"Could not process this file: {e}")

    if st.session_state.batch_results is not None:
        result_df = st.session_state.batch_results
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Patients scored", len(result_df))
        with c2:
            st.metric("Flagged high risk", int((result_df["Predicted"] == "High risk").sum()))
        with c3:
            st.metric("Average risk score", f"{result_df['Risk %'].mean():.1f}%")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.dataframe(result_df, width="stretch", height=380)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download results as CSV", data=csv_bytes,
            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.markdown('<div class="sidebar-card">Upload a CSV to see results here. '
                     'Columns should match the raw patient fields (categorical values like '
                     '"M"/"F", "ATA"/"NAP"/"TA"/"ASY", "Y"/"N", "Up"/"Flat"/"Down" — same '
                     'format as the original training data).</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: Model Comparison
# ============================================================
elif page == "📊 Model Comparison":
    st.markdown('<p class="page-heading">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Performance of all five candidate models evaluated during training. '
        'Logistic Regression <span class="badge">Deployed</span></p>', unsafe_allow_html=True,
    )

    display_metrics = STATIC_METRICS.copy()
    display_metrics[["Accuracy", "Precision", "Recall", "F1"]] = (
        display_metrics[["Accuracy", "Precision", "Recall", "F1"]] * 100
    ).round(1)
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    melted = STATIC_METRICS.melt(id_vars="Model", var_name="Metric", value_name="Score")
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("Score:Q", title="Score", axis=alt.Axis(format="%")),
            color=alt.Color("Metric:N",
                             scale=alt.Scale(range=["#2DD4BF", "#14B8A6", "#F59E0B", "#F43F5E"]),
                             legend=alt.Legend(title=None, orient="top")),
            xOffset="Metric:N",
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".1%")],
        )
        .properties(height=340)
        .configure(background="transparent")
        .configure_axis(labelColor="#EDF2F7", titleColor="#EDF2F7", gridColor="#1F3040", domainColor="#1F3040")
        .configure_legend(labelColor="#EDF2F7", titleColor="#EDF2F7")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Which factors matter most overall (Logistic Regression)</p>',
                unsafe_allow_html=True)
    global_importance_chart()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.expander("🔬  Compare live predictions with other trained models (optional)"):
        st.markdown(
            "Only the Logistic Regression model is bundled with this app. If you also saved "
            "the other models from your notebook (KNN, Naive Bayes, Decision Tree, SVM) as "
            "`.pkl` files with `joblib.dump()`, upload them here to see how they'd score the "
            "**last patient you ran through Single Prediction**."
        )
        extra_files = st.file_uploader(
            "Upload one or more model .pkl files", type=["pkl"], accept_multiple_files=True,
            key="extra_model_uploader",
        )
        if extra_files:
            for f in extra_files:
                try:
                    st.session_state.extra_models[f.name] = joblib.load(f)
                except Exception as e:
                    st.error(f"Could not load {f.name}: {e}")

        if st.session_state.extra_models and st.session_state.last_result is not None:
            scaled_row = st.session_state.last_result["scaled"].reshape(1, -1)
            rows = [{
                "Model": "Logistic Regression (deployed)",
                "Prediction": "High risk" if st.session_state.last_result["prediction"] == 1 else "Low risk",
                "Risk %": f"{st.session_state.last_result['risk_pct']:.1f}%",
            }]
            for name, extra_model in st.session_state.extra_models.items():
                try:
                    pred = extra_model.predict(scaled_row)[0]
                    try:
                        proba = extra_model.predict_proba(scaled_row)[0][1] * 100
                        proba_str = f"{proba:.1f}%"
                    except Exception:
                        proba_str = "N/A"
                    rows.append({
                        "Model": name,
                        "Prediction": "High risk" if pred == 1 else "Low risk",
                        "Risk %": proba_str,
                    })
                except Exception as e:
                    rows.append({"Model": name, "Prediction": f"Error: {e}", "Risk %": "—"})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        elif st.session_state.extra_models and st.session_state.last_result is None:
            st.info("Run a prediction on the Single Prediction page first, then come back here.")

# ============================================================
# PAGE: History
# ============================================================
elif page == "🕘 History":
    st.markdown('<p class="page-heading">Prediction History</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Everything you have predicted in this session '
                '(single and batch). History clears when the app restarts.</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<div class="sidebar-card">No predictions yet. Run one from '
                     '"Single Prediction" or "Batch Prediction".</div>', unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, width="stretch", height=420)

        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download history as CSV", data=csv_bytes,
                                file_name="prediction_history.csv", mime="text/csv")
        with c2:
            if st.button("🗑️  Clear history"):
                st.session_state.history = []
                st.rerun()