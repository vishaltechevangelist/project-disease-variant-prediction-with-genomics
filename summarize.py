import dspy

llm = dspy.LM(model='gemini/gemini-2.0-flash', api_key='AIzaSyBkwTmFdjVs8XloV2fEKbT3qb98ULHfTvo')
dspy.settings.configure(lm=llm)
model = dspy.ChainOfThought()
