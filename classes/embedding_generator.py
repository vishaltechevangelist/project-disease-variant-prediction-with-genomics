class embedding_generator():
    def __init__(self, transformers, embedding_model, device):
        self.transformers = transformers
        self.embedding_model = embedding_model
        self.device = device

    def get_embedding(self, text):
        model = self.transformers(self.embedding_model, device=self.device)
        return model.encode(text, convert_to_numpy=True, device=self.device)