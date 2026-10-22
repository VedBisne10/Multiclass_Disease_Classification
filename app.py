
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="MediSense AI | Disease Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_xgboost_model.joblib"
ENCODER_PATH = BASE_DIR / "data" / "processed" / "label_encoder.joblib"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #0F766E;
    --primary-light: #CCFBF1;
    --text: #172033;
    --muted: #64748B;
    --border: #E2E8F0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #F5FAFA 0%, #F8FAFC 45%, #F3F7FB 100%);
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0B1727;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #CBD5E1;
}

.hero {
    background: linear-gradient(120deg, #0B1727 0%, #123B46 60%, #0F766E 100%);
    border-radius: 24px;
    padding: 32px;
    color: white;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(15, 60, 70, 0.12);
}

.hero-kicker {
    color: #99F6E4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    color: #FFFFFF;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(28px, 4vw, 43px);
    line-height: 1.15;
    font-weight: 800;
    margin: 0 0 12px 0;
}

.hero p {
    color: #D1E8E8;
    max-width: 700px;
    font-size: 15px;
    line-height: 1.8;
    margin: 0;
}

.hero-pill {
    display: inline-block;
    margin-top: 20px;
    background: rgba(153, 246, 228, 0.12);
    color: #99F6E4;
    border: 1px solid rgba(153, 246, 228, 0.28);
    border-radius: 30px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 600;
}

.section-heading {
    color: #172033;
    font-family: 'Manrope', sans-serif;
    font-size: 21px;
    font-weight: 800;
    margin: 8px 0 4px 0;
}

.section-subtitle {
    color: #64748B;
    font-size: 13px;
    margin-bottom: 18px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 18px;
    min-height: 115px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.025);
}

.metric-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #0F172A;
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
}

.panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 22px;
    margin: 12px 0 18px 0;
}

.result-panel {
    background: linear-gradient(135deg, #ECFDF5, #F0FDFA);
    border: 1px solid #A7F3D0;
    border-radius: 18px;
    padding: 24px;
    margin-top: 18px;
}

.result-label {
    color: #047857;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1.4px;
    font-weight: 800;
}

.result-name {
    color: #064E3B;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(23px, 3vw, 32px);
    font-weight: 800;
    margin: 8px 0;
    overflow-wrap: anywhere;
}

.disclaimer {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 14px;
    padding: 14px 16px;
    color: #78350F;
    font-size: 12px;
    line-height: 1.7;
    margin: 14px 0;
}

.footer {
    color: #94A3B8;
    font-size: 12px;
    text-align: center;
    padding-top: 24px;
    border-top: 1px solid #E2E8F0;
    margin-top: 35px;
}

.stButton > button {
    background: linear-gradient(110deg, #0F766E, #0D9488);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #115E59;
    color: white;
    border: none;
    box-shadow: 0 5px 15px rgba(15, 118, 110, 0.2);
}

div[data-testid="stMultiSelect"] > div {
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 700px) {
    .hero {
        padding: 23px;
        border-radius: 18px;
    }
    .panel {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    feature_columns = pd.read_csv(
        FEATURES_PATH, nrows=0
    ).columns.tolist()

    return model, encoder, feature_columns


try:
    model, label_encoder, feature_columns = load_model()
except Exception as exc:
    st.error(f"Unable to load model files: {exc}")
    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🩺 MediSense AI")
    st.caption("Multiclass Disease Classification")

    st.divider()

    st.markdown("### Project overview")
    st.write(
        "An academic machine-learning project that explores "
        "relationships between symptom patterns and disease labels."
    )

    st.markdown("### Model information")
    st.write("**Algorithm:** XGBoost")
    st.write(f"**Input features:** {len(feature_columns)}")
    st.write(f"**Disease classes:** {len(label_encoder.classes_)}")

    st.divider()

    st.markdown("### How to use")
    st.write("1. Search for symptoms.")
    st.write("2. Select all applicable symptoms.")
    st.write("3. Run the prediction.")
    st.write("4. Interpret results cautiously.")

    st.divider()

    st.caption("Research prototype · Not a medical device")


# ==================================================
# HERO
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">AI-powered health research</div>
    <h1>Understand symptom patterns.<br>Explore model predictions.</h1>
    <p>
        An interactive interface for exploring an experimental
        multiclass disease classification model. Select symptoms
        and examine the category predicted by the trained model.
    </p>
    <div class="hero-pill">● &nbsp; XGBoost model interface</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# METRIC CARDS
# ==================================================

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">AVAILABLE FEATURES</div>
        <div class="metric-value">{len(feature_columns)}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">DISEASE CATEGORIES</div>
        <div class="metric-value">{len(label_encoder.classes_)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">CLASSIFICATION ENGINE</div>
        <div class="metric-value">XGBoost</div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# MEDICAL DISCLAIMER
# ==================================================

st.markdown("""
<div class="disclaimer">
    <b>⚠ Medical disclaimer:</b> This application is intended
    exclusively for educational and research purposes. Its output
    is not a diagnosis, medical advice, or a treatment recommendation.
    The model can make incorrect predictions. Consult a qualified
    healthcare professional about symptoms or health concerns.
</div>
""", unsafe_allow_html=True)


# ==================================================
# SYMPTOM INPUT
# ==================================================

st.markdown(
    '<div class="section-heading">Symptom explorer</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Search the dataset features and select the symptoms you want '
    'to evaluate.'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    left, right = st.columns([3, 1])

    with left:
        selected_symptoms = st.multiselect(
            "Search and select symptoms",
            options=feature_columns,
            format_func=lambda s: s.replace("_", " ").title(),
            placeholder="Start typing a symptom...",
            key="symptom_selector",
        )

    with right:
        st.markdown("#### Selection")
        st.markdown(f"### {len(selected_symptoms)}")
        st.caption("Symptoms selected")

    if selected_symptoms:
        st.markdown("**Selected symptoms**")
        st.write(
            "  ·  ".join(
                symptom.replace("_", " ").title()
                for symptom in selected_symptoms
            )
        )
    else:
        st.info(
            "No symptoms selected yet. Use the search box above "
            "to begin."
        )

    predict_clicked = st.button(
        "Run model prediction  →",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# PREDICTION RESULT
# ==================================================

if predict_clicked:
    if not selected_symptoms:
        st.warning("Select at least one symptom before predicting.")
    else:
        input_data = pd.DataFrame(
            np.zeros((1, len(feature_columns)), dtype=np.int8),
            columns=feature_columns,
        )

        input_data.loc[0, selected_symptoms] = 1

        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(
            np.asarray(prediction, dtype=int).reshape(-1)
        )[0]

        st.markdown(
            '<div class="section-heading">Prediction results</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-panel">
            <div class="result-label">Model-predicted category</div>
            <div class="result-name">
                {predicted_disease.replace('_', ' ')}
            </div>
            <div style="color:#065F46;font-size:13px;">
                Generated from the selected dataset features.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Input summary")

        r1, r2 = st.columns(2)

        with r1:
            st.metric("Selected symptoms", len(selected_symptoms))

        with r2:
            st.metric("Input feature count", len(feature_columns))

        with st.expander("View selected symptoms"):
            for symptom in selected_symptoms:
                st.write(f"✓ {symptom.replace('_', ' ').title()}")

        # Optional top-k model scores.
        # These are model outputs, NOT calibrated medical probabilities.
        if hasattr(model, "predict_proba"):
            st.markdown("#### Other model-ranked categories")
            st.caption(
                "Scores below are raw model probability outputs. "
                "They are not calibrated probabilities of having a disease "
                "and must not be interpreted as clinical risk."
            )

            try:
                scores = model.predict_proba(input_data)[0]
                class_ids = np.asarray(model.classes_, dtype=int)

                top_indices = np.argsort(scores)[::-1][:5]

                score_rows = []
                for idx in top_indices:
                    class_id = class_ids[idx]
                    disease_name = label_encoder.inverse_transform(
                        [class_id]
                    )[0]

                    score_rows.append({
                        "Model-ranked category": disease_name.replace("_", " "),
                        "Model score (%)": round(float(scores[idx]) * 100, 2),
                    })

                st.dataframe(
                    pd.DataFrame(score_rows),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception:
                st.caption("Additional model scores are unavailable.")

        st.markdown("""
        <div class="disclaimer">
            <b>Remember:</b> A predicted category does not confirm
            that a person has the disease. The model was trained on
            a dataset and may not generalize to real-world patients.
            Seek qualified medical advice for any health concern.
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    MediSense AI · Academic machine-learning project<br>
    Experimental predictions only · Not for clinical decision-making
</div>
""", unsafe_allow_html=True)
