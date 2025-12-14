import os, logging, sys
from helpers.helper import load_json_for_dd

''' Application related configuration'''
__APP_PATH__ = os.getcwd()
__MODEL_PATH__ = os.path.join(__APP_PATH__, 'models')
__MODEL_FILE_NAME__ = 'xgb_model.joblib'
__LOG_PATH__ = os.path.join(__APP_PATH__, 'logs')
__DD_JSON__ = os.path.join(__APP_PATH__, 'dd_json')
__CHROMO_ID_LABEL_FILE__ = 'chromo_id_label.json'
__CHROMO_GENE_MAP_FILE__ = 'chromo_gene_map.json'
__GENE_ID_LABEL_FILE__ = 'gene_id_label.json'
__SIG_LABEL_FILE__ = 'sig_label_map.json'
__ERROR_LOG_FILE_NAME__ = 'error.log'
__DISEASE_LOOKUP_FILE__ = '/Users/vishalsaxena/Downloads/proj_data/disease_name_lookup.tsv'

'''Logger configuration and filepath'''
logging.basicConfig(filename=os.path.join(__LOG_PATH__, __ERROR_LOG_FILE_NAME__), level=logging.ERROR)

'''Json Maps have categorical to numeric mapping ported from preprocessing phase of project'''
json_maps = {
    'chromo_id_label_dd': load_json_for_dd(__DD_JSON__, __CHROMO_ID_LABEL_FILE__),
    'chromo_gene_map' : load_json_for_dd(__DD_JSON__, __CHROMO_GENE_MAP_FILE__),
    # 'review_map': {"no_assertion_criteria_provided": 0, "criteria_provided,_single_submitter": 1, "criteria_provided,_multiple_submitters,_no_conflicts": 2, "reviewed_by_expert_panel": 3, "practice_guideline": 4, "criteria_provided,_conflicting_classifications": 5, "no_classification_provided": 6, "no_classifications_from_unflagged_records": 7, "no_classification_for_the_single_variant": 8},
    'sig_label_map': load_json_for_dd(__DD_JSON__, __SIG_LABEL_FILE__),
    'gene_id_label_dd' :  load_json_for_dd(__DD_JSON__, __GENE_ID_LABEL_FILE__)
    }

'''Streamlit UI fields and their defaults'''
__SELECT__ = 'SELECT'
__CHECKBOX__ = 'CHECKBOX'
__NUMBER__ = 'NUMBER'
__FEATURE_COLUMNS_DEFAULTS__ = {
    "Chromosome_Encoded": {
        'label': 'Chromosome',
        'out_df_label': 'Chromosome',
        # 'type': __SELECT__,
        'map_name' : 'chromo_id_label_dd',
        'default': "28"
    },
    # "Clinical_Review_Status_Encoded": {
    #     'label': 'Clinical Review Status & their Id',
    #     'out_df_label': 'Clinical Review Status',
    #     'type': __SELECT__,
    #     'map_name' : 'review_map',    
    #     'default' : 1
    # },                   
    "Gene_Symbol_Encoded": {
        'label': 'Gene Symbol',
        'out_df_label': 'Gene Symbol',
        # 'type': __SELECT__,
        'map_name':'gene_id_label_dd',
        # 'dependent': 'Chromosome_Encoded',
        'default':3969
    },
    # "POS_Percentile": {
    #     'label': 'Position of change',
    #     'out_df_label': 'Position of change',
    #     'type': __NUMBER__,
    #     'default': 0.888870
    # },
    "IS_SNP": {
        'label': 'Single Nucleotide Change (IS_SNP)',
        'out_df_label': 'Single Nucleotide Change',
        # 'type': __CHECKBOX__,
        'default': 0,
        'radio': 1
    },
    "IS_INDEL": {
        'label': 'Insertion or Deletion (IS_INDEL)',
        'out_df_label': 'Nucleotide Insertion or Deletion',
        # 'type': __CHECKBOX__,
        'default': 1,
        'radio':1
    }
}

__PAGE_TITLE__ = 'GenomeDx: Genomics meets AI'
__SIDEBAR_TITLE__ = 'Genetic Input Features'
__FORM_NAME__ = 'genetic_input_feature'

__MESSAGE__ = {
    'MODEL_NOT_LOADED' : f'Model not loaded. Place model at "{__MODEL_PATH__}/xgb_model.joblib" or update __MODEL_PATH__ in config',
    'PREDICTION_SUCCESS' : 'Prediction completed using ',
    'HEADING_FOR_LLMTEXT' : 'Explanation (Gemini via DSPy)',
}

__MODEL_LIST__ = {
       1: {
           'name' : 'Random Forest',
           'model_name': 'rft_model.joblib',
           'accuracy' :  'Has Acc: 58%'
       },
       2: {
            'name' : 'XGBoost',
            'model_name' : 'xgb_model.joblib',
            'accuracy' : 'Has Acc: 64%'
        }
}

'''Prediction label added to model output dataframe'''
__PREDICTION__ = 'Prediction'
__CLINICAL_SIGNIFIANCE__ =  'Clinical Significance'
__PROBABILITY__ = 'Confidence'
__HARMFUL_DISEASE_LABEL__ = 'pathogenic'

'''dspy configuration'''
dspy_config = {
    # 'LM_NAME' : 'gemini/gemini-2.0-flash',
    # 'LM_KEY_NAME' : 'GOOGLE_API_KEY',
    'LM_NAME' : 'groq/llama-3.1-8b-instant',
    'LM_KEY_NAME' : 'GROQ_API_KEY',
    'SIGNATURE' : "input_features -> model_prediction_explanation",
    'LLM_ROLE_GOAL_INSTRUCTION' : """
                ### ROLE
                You are an expert genomic interpreter who explains model results in simple, layman-friendly language.

                ### GOAL
                Given model inputs describing a DNA variant and its predicted clinical significance, produce a short, clear, and friendly explanation that any educated person without genetics training can understand.
                ---
                ### STYLE RULES
                1. Start with a **short summary (1-2 sentences)** that tells what the model predicts (e.g., 'likely harmless', 'potentially disease-causing') and how confident it is (convert numeric confidence into plain terms: low, moderate, high).
                2. Then add **2 bullet points** (each one short sentence) that explain *why* in everyday terms:
                    - Mention whether it's a **single-letter change (SNP)** or **insertion/deletion (INDEL)**.
                    - Explain only from the provided data — do **not invent** new biology facts.
                3. Parse disease json list have disease name, number of submission, count of evidence, review status and explain in simole terma  
                4. End with a **single closing sentence** suggesting a non-prescriptive next step, such as:
                    - “If you are concerned, you can share this report with a clinician.”
                    - “This result is mostly reassuring but always best discussed with a professional.”
                5. Use a warm, informative tone — short words, active voice, no jargon.
                6. Output text should be consistent contains a short introductory paragraph, then a "What it Means" bullet section, then a "Disease Association" paragraph, and finally, a closing suggestion.
                """,
}

'''Embedding model & QDrant configs'''
# __EMBEDDING_MODEL__ = 'pritamdeka/S-BioBERT-snli-stsb'
__EMBEDDING_MODEL__ = 'pritamdeka/S-BioBert-snli-multinli-stsb' 
__EMBEDDING_COLLECTION_NAME__ = 'GenomeDx'


