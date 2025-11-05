# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv
import dspy
import json
import os
import re
import logging

logging.basicConfig(filename="error.log", level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()
llm = dspy.LM(model='gemini/gemini-2.0-flash', api_key=os.getenv("GOOGLE_API_KEY"))
dspy.settings.configure(lm=llm)
signature = "input_features -> model_output_prediction_with_explanation"

instruction_block = """
                You are an expert genomic interpreter who explains model results in simple, layman-friendly language.

                ### GOAL
                Given model inputs describing a DNA variant and its predicted clinical significance, produce a short, clear, and friendly explanation that any educated person without genetics training can understand.
                ---
                ### STYLE RULES
                1. Start with a **short summary (1–2 sentences)** that tells what the model predicts (e.g., 'likely harmless', 'potentially disease-causing') and how confident it is (convert numeric confidence into plain terms: low, moderate, high).
                2. Then add **2 bullet points** (each one short sentence) that explain *why* in everyday terms:
                    - Mention whether it’s a **single-letter change (SNP)** or **insertion/deletion (INDEL)**.
                    - Mention what the **position percentile** and **review status** imply (e.g., “well-studied region” or “limited expert review”).
                    - Explain only from the provided data — do **not invent** new biology facts.
                3. End with a **single closing sentence** suggesting a non-prescriptive next step, such as:
                    - “If you are concerned, you can share this report with a clinician.”
                    - “This result is mostly reassuring but always best discussed with a professional.”
                4. Use a warm, informative tone — short words, active voice, no jargon.
                5. Output a JSON object with **exactly three fields**:
                    {
                    "user_facing_summary": "<short paragraph>",
                    "why": ["<bullet1>", "<bullet2>"],
                    "next_step": "<one closing sentence>"
                    }
                """

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
    "Chromosome_Encoded": 3,
    "Clinical_Review_Status_Encoded": 1,
    "Gene_Symbol_Encoded": 3969,
    "POS_Percentile": 0.888870,
    "IS_SNP": 0,
    "IS_INDEL": 1,
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
    map_list = {
        'sig_label_map' : 1, 
        'gene_id_map' : 2, 
        'review_map' : 3, 
        'chrom_map' : 4                
        }

    for map_name in map_list.keys():
        filename = f'json/{map_name}.json'
        with open(filename, 'r') as f:
            map_list[map_name] = json.load(f)

    

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
            out["Clinical_Significance"] = [key for key, value in map_list['sig_label_map'].items() if value == pred]
            if proba is not None:
                out["prob_max"] = proba.max(axis=1)
            
            
            input_to_llm = {   
                "instruction_block" : instruction_block,
                "Clinical_Significance": out["Clinical_Significance"].iloc[0],
                "Chromosome": [key for key, value in map_list['chrom_map'].items() if value == X["Chromosome_Encoded"].iloc[0]][0],
                "Clinical_Review_Status": [key for key, value in map_list['review_map'].items() if value == X["Clinical_Review_Status_Encoded"].iloc[0]][0],
                "Gene_Symbol": [key for key, value in map_list['gene_id_map'].items() if value == X["Gene_Symbol_Encoded"].iloc[0]][0],
                "POS_Percentile": float(X["POS_Percentile"].iloc[0]),
                "IS_SNP": "Yes" if X["IS_SNP"].iloc[0] == 1 else "No",
                "IS_INDEL": "Yes" if X["IS_INDEL"].iloc[0] == 1 else "No",
                "prediction_label" : out["Clinical_Significance"].iloc[0],
                "confidence" : float(out["prob_max"].iloc[0])
            }
    
            gemini_model = dspy.ChainOfThought(signature=signature, expose_cot=False)
            result = gemini_model(input_features=json.dumps(input_to_llm))
            st.success("Prediction complete")
            st.dataframe(out)
            clean_json_str = json.loads(re.sub(r'```json|```', '', result.model_output_prediction_with_explanation).strip())
            str_to_display = clean_json_str['user_facing_summary'] + '\n\t' + clean_json_str['why'][0] + '\n\t' + clean_json_str['why'][1] + '\n' + clean_json_str['next_step']
            st.info(str_to_display)
    except Exception as e:
        logger.exception("Error occurred: %s", e, exc_info=True)
        st.error("An unexpected error occurred. Check error.log for details.")


st.markdown("---")