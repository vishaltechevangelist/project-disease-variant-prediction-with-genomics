import streamlit as st
import pandas as pd
import config, joblib, traceback, dspy
from user_dspy_llm import user_dspy_llm
from helper import df_to_json, format_output

logger = config.logging.getLogger(__name__)

st.set_page_config(page_title=config.PAGE_TITLE, layout="wide")
st.title(config.PAGE_TITLE)

st.sidebar.header(config.SIDEBAR_TITLE)

st.markdown("""
    <style>
        /* Sidebar width */
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 350px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path=config.__MODEL_PATH__, model_file_name=config.__MODEL_FILE_NAME__):
    try:
        model = joblib.load(path+"/"+model_file_name)
        return model
    except Exception as e:
        st.error(config.MESSAGE['MODEL_NOT_LOADED'])
        return None

@st.cache_resource
def get_configured_dspy_llm():
    try:
        llm = dspy.LM(model='gemini/gemini-2.0-flash', api_key=st.secrets["GOOGLE_API_KEY"])
        dspy.settings.configure(lm=llm)
    except Exception as e:
        logger.exception("Error occurred: %s", e, exc_info=True)
        st.error(f"An error occurred: {e}")
    return llm
    
with st.sidebar.form(config.FORM_NAME, width=500):
    ui_vals = {}
    encoded_vals = {}
    for feature_name, feature_maps in config.FEATURE_COLUMNS_DEFAULTS.items():
        label = feature_maps['label']
        if feature_maps['type'] == config.__SELECT__:
            option_list = config.json_maps[feature_maps['map_name']]
            ui_vals[feature_name] = st.selectbox(label=label, options=list(option_list.keys()), 
                                                 key=feature_name, format_func= lambda x : f"{x} - {option_list[x]}")
            encoded_vals[feature_name] = option_list[ui_vals[feature_name]]
        elif feature_maps['type'] == config.__CHECKBOX__:
            encoded_vals[feature_name] = ui_vals[feature_name] = st.checkbox(label=label, value=bool(feature_maps['default']))
        elif feature_maps['type'] == config.__NUMBER__:
            encoded_vals[feature_name] = ui_vals[feature_name] = st.number_input(label=label, value=float(feature_maps['default']), format="%.6f")
    submit_btn = st.form_submit_button("Get Prediction")


if submit_btn:
    try:
        # print(ui_vals)
        # print(encoded_vals)
        ui_df = pd.DataFrame([ui_vals], columns=config.FEATURE_COLUMNS_DEFAULTS.keys())
        X = encoded_df = pd.DataFrame([encoded_vals], columns=config.FEATURE_COLUMNS_DEFAULTS.keys())

        # st.dataframe(ui_df)
        # st.dataframe(encoded_df)

        ##** Classifier Model Integration **##
        classifier_model = load_model()
        if classifier_model is None:
            raise Exception(config.MESSAGE['MODEL_NOT_LOADED'])
        else:
            prediction = classifier_model.predict(X)
            probabilities = classifier_model.predict_proba(X) if hasattr(classifier_model, "predict_proba") else None
            display_df = ui_df.copy()
            display_df[config.__PREDICTION__] = prediction
            label_map = config.json_maps['sig_label_map']
            display_df[config.__CLINICAL_SIGNIFIANCE__] = [key for key, value in label_map.items() if value == prediction]
            if probabilities is not None:
                display_df[config.__PROBABILITY__] = probabilities.max(axis=1)
            st.success(config.MESSAGE['PREDICTION_SUCCESS'])
            st.dataframe(display_df.T.astype(str))

            ##** dspy integration **##
            dspy_config = config.dspy_config
            user_dspy_llm = user_dspy_llm(dspy=dspy, lmname=dspy_config['LM_NAME'], lm_api_key=st.secrets[dspy_config['LM_KEY_NAME']], 
                                          signature=dspy_config['SIGNATURE'], instruction = dspy_config['LLM_ROLE_GOAL_INSTRUCTION'])
            llm = get_configured_dspy_llm()
            # user_dspy_llm.dspy_configure_lm()
            output_explanation = user_dspy_llm.get_prediction_explanation(df_to_json(display_df))
            st.markdown(format_output(output_explanation), unsafe_allow_html=True)

    except Exception as e:
        logger.exception("Error occurred: %s", e, exc_info=True)
        trace_str = traceback.format_exc()
        st.error(f"An error occurred: {e}")
        st.text_area("Detailed Traceback:", trace_str, height=200)