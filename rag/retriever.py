class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder  # Embedder already handles the model and index

    def search(self, query, k=3):
        return self.embedder.search(query, top_k=k)