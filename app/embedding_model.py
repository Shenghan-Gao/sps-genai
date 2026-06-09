import spacy


class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def get_embedding(self, word):
        doc = self.nlp(word)

        if len(doc) == 0:
            return {
                "word": word,
                "error": "No valid token found"
            }

        token = doc[0]

        return {
            "word": word,
            "has_vector": token.has_vector,
            "vector_norm": float(token.vector_norm),
            "embedding": token.vector.tolist()
        }