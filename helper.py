import numpy as np
import json, os

def df_to_json(df):
    json_str = {}
    features = df.columns
    # print(df.info())
    for column in features:
        if df[column].dtype in [np.float16, np.float32, np.float64]:
            json_str[column] = float(df[column][0])
        elif df[column].dtype in [np.int16, np.int32, np.int64]:
            json_str[column] = int(df[column][0])
        elif df[column][0] == np.True_ or df[column][0] == np.False_:
            json_str[column] = bool(df[column][0])
        elif df[column].dtype in [object]:
            json_str[column] = str(df[column][0])
    return json_str

def format_output(display_output):
    return f'<div style="background-color:#2b6cb0;color:white;border-left:6px solid #2196F3;padding:10px 16px;border-radius:8px;">{display_output}</div>'

def load_json_for_dd(path, filename):
    with open(os.path.join(path, filename), 'r') as f:
        return json.load(f)
    
def combine_disease_data_for_llm(disease_list):
    """
    Combines the list of disease dictionaries into a single, clean markdown string
    for easy consumption by the LLM.

    The primary disease (highest score) is highlighted, and others are listed as minor findings.
    """
    if not disease_list:
        return "No associated disease data found in the lookup database."

    # Assume the list is already sorted by score (as per get_top_diseases logic)
    main_disease = disease_list[0]
    other_diseases = disease_list[1:]

    # Start with the main finding
    output_str = "### Clinical Context from Reference Databases\n"
    output_str += (
        f"The top probable disease is **{main_disease['disease']}** (Score: {main_disease['score']}). "
        f"This disease is supported by {main_disease['pathogenic_submissions']} pathogenic submissions "
        f"and {main_disease['freq']} total records. "
        f"Review statuses include: {', '.join(set(main_disease['review_statuses']))}.\n"
    )

    # List secondary findings if they exist
    if other_diseases:
        output_str += "\nOther diseases found in the database (minor findings):\n"
        for d in other_diseases:
            output_str += f"- {d['disease']} (Score: {d['score']}, Submissions: {d['pathogenic_submissions']})\n"
    
    return output_str