# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------- CONFIG -----------------
MODEL_PATH = "models/xgb_model.joblib"  # change if different
FEATURE_COLUMNS = [
    "Chromosome_Encoded",
    "Clinical_Review_Status_Encoded",
    "Gene_Symbol_Encoded",
    "POS_Percentile",
    "IS_SNP",
    "IS_INDEL",
]

# sensible defaults taken from the sample you provided (first row)
UI_DEFAULTS = {
    "Chromosome_Encoded": 1,
    "Clinical_Review_Status_Encoded": 1,
    "Gene_Symbol_Encoded": 1,
    "POS_Percentile": 12.0,
    "IS_SNP": 0,
    "IS_INDEL": 1,
}
# ------------------------------------------

sig_label_map = {
     0 :  'Benign',
     1 : 'Uncertain',
     2 : 'Pathogenic',
     3 : 'Risk_factor',
     4 : 'Drug_response',
     5 : 'Association',
     6 : 'Other',
     7 : 'Conflicting',
}

st.set_page_config(page_title="Genomics Predictor — Form Inputs", layout="wide")
st.title("Genomics Predictor — Form Inputs")

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None

model = load_model()
if model is None:
    st.warning("Model not loaded. Place model at 'models/xgb_model.joblib' or update MODEL_PATH.")
    # still allow UI building for testing

# --------- Single-sample form ----------
st.header("Sample input")
with st.form("single_form"):
    cols = st.columns(2)
    ui_vals = {}
    for i, feat in enumerate(FEATURE_COLUMNS):
        col = cols[i % 2]
        default = UI_DEFAULTS.get(feat)
        # choose widget types:
        if feat in ("IS_SNP", "IS_INDEL"):
            # booleans: show checkbox (store as 0/1)
            with col:
                checked = st.checkbox(feat, value=bool(default))
                ui_vals[feat] = int(checked)
        elif feat == "POS_Percentile":
            with col:
                ui_vals[feat] = st.number_input(feat, value=float(default), format="%.6f")
        else:
            # encoded integer features
            with col:
                ui_vals[feat] = int(st.number_input(feat, value=int(default), step=1))
    submit_single = st.form_submit_button("Predict single sample")

if submit_single:
    # build dataframe in required column order
    try:
        X = pd.DataFrame([ui_vals], columns=FEATURE_COLUMNS)
        st.write("Input data:")
        st.dataframe(X)
        if model is None:
            st.info("No model loaded — showing input only.")
        else:
            pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            out = X.copy()
            out["prediction"] = pred
            out["Clinical_Significance"] = out["prediction"].map(sig_label_map)
            if proba is not None:
                out["prob_max"] = proba.max(axis=1)
            st.success("Prediction complete")
            st.dataframe(out)
            csv = out.to_csv(index=False)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")