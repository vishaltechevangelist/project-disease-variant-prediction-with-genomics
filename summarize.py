from dotenv import load_dotenv
import os
import dspy
import json

map_list = {
        'sig_label_map' : 1, 
        'gene_id_map' : 2, 
        'review_map' : 3, 
        'chrom_map' : 4                
        }

for map_name in map_list.keys():
    filename = f'../json/{map_name}.json'
    with open(filename, 'r') as f:
        map_list[map_name] = json.load(f)


load_dotenv()
llm = dspy.LM(model='gemini/gemini-2.0-flash', api_key=os.getenv("GOOGLE_API_KEY"))
dspy.settings.configure(lm=llm)


signature = {
    "inputs": {
        "features": "Dict[str, Any]",      
        "prediction_label": "int",
        "confidence": "float"
    },
    "outputs": {
        "user_facing_summary": "str",
        "supporting_explanation": "str",
        "recommendation": "str"
    }
}

model = dspy.ChainOfThought(signature=signature, expose_cot=False)

sample_row = {
    "Clinical_Significance_Encoded": 2,          # Pathogenic
    "Chromosome_Encoded": 23,                    # X
    "Clinical_Review_Status_Encoded": 3,         # reviewed_by_expert_panel
    "Gene_Symbol_Encoded": 2,                    # SAMD11
    "POS_Percentile": 0.84,
    "IS_SNP": 1,
    "IS_INDEL": 0       
}

decoded_features = {
    "Clinical_Significance": [key for key, value in map_list[map_name].items() if value == sample_row["Clinical_Significance_Encoded"]],
    "Chromosome": [key for key, value in map_list[map_name].items() if value == sample_row["Chromosome_Encoded"]],
    "Clinical_Review_Status": [key for key, value in map_list[map_name].items() if value == sample_row["Clinical_Review_Status_Encoded"]],
    "Gene_Symbol": [key for key, value in map_list[map_name].items() if value == sample_row["Gene_Symbol_Encoded"]],
    "POS_Percentile": sample_row["POS_Percentile"],
    "IS_SNP": "Yes" if sample_row["IS_SNP"] == 1 else "No",
    "IS_INDEL": "Yes" if sample_row["IS_INDEL"] == 1 else "No"
}

prediction_label = sample_row["Clinical_Significance_Encoded"]
confidence = 0.91

result = model.run(features=decoded_features,
                   prediction_label=prediction_label,
                   confidence=confidence)

print(result)