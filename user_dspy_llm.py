import json
class user_dspy_llm():
    def __init__(self, dspy, lmname, lm_api_key, signature="input->output", instruction = ""):
        self.lmname = lmname
        self.lm_api_key = lm_api_key
        self.signature = signature
        self.dspy = dspy
        self.instruction = {'instruction': instruction}
    
    def dspy_configure_lm(self):
        try:
            lm = self.dspy.LM(model=self.lmname, api_key=self.lm_api_key)
            self.dspy.settings.configure(lm=lm)
        except Exception as e:
            raise Exception("Problem in configuring in dspy with llm")
        return lm
    
    def get_prediction_explanation(self, input_to_lm):
        lm_model = self.dspy.ChainOfThought(signature=self.signature, expose_cot=False)
        input_to_lm = {**self.instruction, **input_to_lm}
        # print(input_to_lm)
        summary = self.signature.split('->')[1].strip()
        return lm_model(input_features=json.dumps(input_to_lm))[summary]
