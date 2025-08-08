import nltk
from typing import List, Dict, Any
import pandas as pd

# Ensure resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def clean_text_column(df: pd.DataFrame, col: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    df2 = df.copy()
    sw = set(stopwords.words("english")) if cfg.get("remove_stopwords", True) else set()
    stemmer = PorterStemmer() if cfg.get("stemming", False) else None
    lemm = WordNetLemmatizer() if cfg.get("lemmatization", True) else None

    def process(text: str) -> str:
        if not isinstance(text, str):
            return ""
        tokens = nltk.word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha()]
        if sw:
            tokens = [t for t in tokens if t not in sw]
        if stemmer:
            tokens = [stemmer.stem(t) for t in tokens]
        if lemm:
            tokens = [lemm.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    df2[col] = df2[col].apply(process)
    return df2
