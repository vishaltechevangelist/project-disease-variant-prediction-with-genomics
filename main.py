import streamlit as st
import pandas as pd
import config, joblib, traceback, dspy, json, os, sys
from user_dspy_llm import user_dspy_llm
from helper import df_to_json, format_output, load_json_for_dd

logger = config.logging.getLogger(__name__)

st.set_page_config(page_title=config.__PAGE_TITLE__, layout="wide")
st.title(config.__PAGE_TITLE__)

st.sidebar.header(config.__SIDEBAR_TITLE__)
# st.header(config.__SIDEBAR_TITLE__)


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
        model = joblib.load(os.path.join(path, model_file_name))
        return model
    except Exception as e:
        st.error(config.__MESSAGE__['MODEL_NOT_LOADED'])
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

chromo_id_label_dd = load_json_for_dd(config.__DD_JSON__, config.__CHROMO_ID_LABEL_FILE__)
chromo_gene_map = load_json_for_dd(config.__DD_JSON__, config.__CHROMO_GENE_MAP_FILE__)
gene_id_label_dd = load_json_for_dd(config.__DD_JSON__, config.__GENE_ID_LABEL_FILE__)

if 'Chromosome_Encoded' not in st.session_state:
        st.session_state['Chromosome_Encoded'] = config.__FEATURE_COLUMNS_DEFAULTS__['Chromosome_Encoded']['default']
# if 'Gene_Symbol_Encoded' not in st.session_state:
#         st.session_state['Gene_Symbol_Encoded'] = config.__FEATURE_COLUMNS_DEFAULTS__['Gene_Symbol_Encoded']['default']

def update_gene_dd():
        sel_chromo = st.session_state['Chromosome_Encoded']
        genes_in_chromo = chromo_gene_map.get(str(sel_chromo), [])
        st.session_state['Gene_Symbol_Encoded'] = str(genes_in_chromo[0]) if genes_in_chromo else ''

ui_vals = {}
encoded_vals = {}

with st.sidebar:
    ui_vals['Chromosome_Encoded'] = st.selectbox('Select Chromosome:', options=list(chromo_id_label_dd.keys()), key='Chromosome_Encoded',
                                                     format_func=lambda x : f"{chromo_id_label_dd[x]}", on_change=update_gene_dd)
    
with st.sidebar.form(config.__FORM_NAME__, width=500):
# with st.form(config.__FORM_NAME__, width=500):
   
    print(f"chromsome state = {st.session_state['Chromosome_Encoded']}")
    # gene_dd_option_list = chromo_gene_map[ui_vals['Chromosome_Encoded']]
    gene_dd_option_list = chromo_gene_map[str(st.session_state['Chromosome_Encoded'])]
    # print(gene_dd_option_list)
    # sys.exit(0)
    # gene_dd_option_list = chromo_gene_map["28"]
    gene_dd_option = []
    for num in gene_dd_option_list:
        gene_dd_option.append(str(num))
    # print(st.session_state['Chromosome_Encoded'])
    print(type(gene_dd_option))
    print(type(gene_id_label_dd[gene_dd_option[0]]))
    ui_vals['Gene_Symbol_Encoded'] = st.selectbox('Select Gene:', options=gene_dd_option, 
                                                      key='Gene_Symbol_Encoded', 
                                                      format_func=lambda x : f"{gene_id_label_dd[str(x)]}"
                                                    )
    
    mutation_type = st.radio("Mutation Type:", options=["SNP (Single Nucleotide Polymorphism)", "INDEL (Insertion/Deletion)"], index=1)

    ui_vals['IS_SNP'] = 1 if "SNP" in mutation_type else 0
    ui_vals['IS_INDEL'] = 1 if "INDEL" in mutation_type else 0

    ui_vals['Chromosome_Encoded'] = int(ui_vals['Chromosome_Encoded'])
    
    model_maps = config.__MODEL_LIST__
    selected_model = st.selectbox(label='Select Model', options=list(config.__MODEL_LIST__.keys()), key='sel_model', 
                                  format_func= lambda x : f"{x}. {model_maps[x]['name']} ({model_maps[x]['accuracy']})"
                                  )

    submit_btn = st.form_submit_button("Get Prediction")

if submit_btn:
    try:
        # print(ui_vals)
        df = pd.DataFrame([ui_vals])
        # st.dataframe(df)
        
        ##** Classifier Model Integration **##
        classifier_model = load_model(path=config.__MODEL_PATH__, 
                                      model_file_name=config.__MODEL_LIST__[selected_model]['model_name'])
        if classifier_model is None:
            raise Exception(config.__MESSAGE__['MODEL_NOT_LOADED'])
        else:
            prediction = classifier_model.predict(df)
            probabilities = classifier_model.predict_proba(df) if hasattr(classifier_model, "predict_proba") else None
            
            display_df = df.copy()
            rename_col = {}
            for col in display_df.columns:
                rename_col[col] = config.__FEATURE_COLUMNS_DEFAULTS__[col]['out_df_label']

            display_df.rename(columns=rename_col, inplace=True)
            display_df[config.__PREDICTION__] = prediction
            label_map = config.json_maps['sig_label_map']
            display_df[config.__CLINICAL_SIGNIFIANCE__] = [key for key, value in label_map.items() if value == prediction]
            
            if probabilities is not None:
                display_df[config.__PROBABILITY__] = probabilities.max(axis=1)

            st.success(f"{config.__MESSAGE__['PREDICTION_SUCCESS']} {config.__MODEL_LIST__[selected_model]['name']} model")
            st.dataframe(display_df.T.astype(str))

            ##** dspy integration **##
            dspy_config = config.dspy_config
            user_dspy_llm = user_dspy_llm(dspy=dspy, lmname=dspy_config['LM_NAME'], lm_api_key=st.secrets[dspy_config['LM_KEY_NAME']], 
                                          signature=dspy_config['SIGNATURE'], instruction = dspy_config['LLM_ROLE_GOAL_INSTRUCTION'])
            llm = get_configured_dspy_llm()
            # user_dspy_llm.dspy_configure_lm()
            output_explanation = user_dspy_llm.get_prediction_explanation(df_to_json(display_df))
            st.subheader(config.__MESSAGE__['HEADING_FOR_LLMTEXT'])
            st.markdown(format_output(output_explanation), unsafe_allow_html=True)

    except Exception as e:
        logger.exception("Error occurred: %s", e, exc_info=True)
        trace_str = traceback.format_exc()
        st.error(f"An error occurred: {e}")
        st.text_area("Detailed Traceback:", trace_str, height=200)