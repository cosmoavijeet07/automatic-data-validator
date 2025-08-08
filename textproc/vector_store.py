from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix = None
        self.docs = []
        self.meta = []

    def fit(self, texts: List[str], meta: List[Dict[str, Any]] = None):
        self.docs = texts
        self.meta = meta or [{} for _ in texts]
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, text: str, top_k: int = 5):
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.matrix).flatten()
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.docs[i], self.meta[i], float(sims[i])) for i in idxs]
