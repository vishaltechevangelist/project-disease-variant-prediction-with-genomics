class embedding_generator():
    def __init__(self, transformers, embedding_model):
        self.transformers = transformers
        self.embedding_model = embedding_model

    def get_embedding(self, text):
        model = self.transformers(self.embedding_model)
        return model.encode(text)