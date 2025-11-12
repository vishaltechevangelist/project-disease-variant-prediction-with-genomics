import dspy

class PredictionExplanation(dspy.Signature):
    input_features = dspy.InputField(desc="JSON string containing the variant features and model prediction/confidence.")
    instruction = dspy.InputField(desc="JSON string containing the instruction to llm regarding role, goal and style rules to display text")
    clinical_context = dspy.InputField(desc="A JSON string containing Chormosome_Label, Gene_Symbol, Clinical_Significance, Clinical_Review_Status, \
                                       Clinical_Disease_Name important keys for context")
    model_prediction_explanation = dspy.OutputField(desc="A detailed, layman-friendly explanation following all style rules.")


class user_dspy_llm:
    def __init__(self, dspy, lmname, lm_api_key):
        self.dspy = dspy
        self.lmname = lmname
        self.lm_api_key = lm_api_key
        self.predictor = dspy.Predict(PredictionExplanation)
        

    def get_prediction_explanation(self, input_features, clinical_context, instruction):
        response = self.predictor(
            input_features=input_features, 
            clinical_context=clinical_context,
            instruction=instruction
        )
        return response.model_prediction_explanation