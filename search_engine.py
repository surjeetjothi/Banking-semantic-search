"""
Banking Semantic Search Engine

Loads trained FastText or Word2Vec models, builds the BankingSemanticSearch class,
and provides loaders for single or side-by-side search evaluation.

IMPORTANT: Tokenizer and vectorizer functions MUST stay identical to train_models.py.
"""

import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is available
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# Tokenizer & vectorizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase → strip non-letter chars → word_tokenize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return word_tokenize(text)


def get_sentence_vector(model, tokens: list[str]) -> np.ndarray:
    """Mean-pool word vectors. FastText handles OOV via char n-grams."""
    vectors = []
    for token in tokens:
        try:
            vectors.append(model.wv[token])
        except KeyError:
            continue
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def get_oov_tokens(model, tokens: list[str]) -> list[str]:
    """Return list of tokens not present in the model's primary vocabulary dictionary."""
    oov = []
    wv = getattr(model, "wv", model)
    key_to_index = getattr(wv, "key_to_index", wv)
    for t in tokens:
        if t not in key_to_index:
            oov.append(t)
    return oov


# ---------------------------------------------------------------------------
# BankingSemanticSearch class
# ---------------------------------------------------------------------------

class BankingSemanticSearch:
    """Cosine-similarity search over pre-computed training embeddings."""

    def __init__(self, model, reference_df: pd.DataFrame,
                 reference_matrix: np.ndarray, tokenizer, vectorizer, model_type: str = "fasttext"):
        """
        Args:
            model: Trained gensim FastText or Word2Vec model.
            reference_df: DataFrame with columns [id, query, category, …].
            reference_matrix: (N, D) numpy array of training embeddings.
            tokenizer: Callable(str) -> list[str].
            vectorizer: Callable(model, list[str]) -> np.ndarray.
            model_type: "fasttext" or "word2vec"
        """
        self.model = model
        self.reference_df = reference_df.reset_index(drop=True)
        self.reference_matrix = reference_matrix
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.model_type = model_type

    def _query_vector(self, query: str) -> tuple[np.ndarray, list[str]]:
        tokens = self.tokenizer(query)
        vec = self.vectorizer(self.model, tokens)
        oov_tokens = get_oov_tokens(self.model, tokens)
        return vec, oov_tokens

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
        """Return the `top_k` most similar training queries and OOV tokens."""
        q_vec, oov_tokens = self._query_vector(query)
        if np.allclose(q_vec, 0):
            empty_df = pd.DataFrame(columns=["id", "query", "category", "similarity_score"])
            return empty_df, oov_tokens

        sims = cosine_similarity(q_vec.reshape(1, -1), self.reference_matrix)[0]
        indices = np.argsort(sims)[::-1][:top_k]

        results = self.reference_df.iloc[indices].copy()
        results["similarity_score"] = sims[indices]
        results = results[results["similarity_score"] >= min_score]
        return results[["id", "query", "category", "similarity_score"]].reset_index(drop=True), oov_tokens

    def search_by_category(self, query: str, category: str,
                           top_k: int = 5) -> tuple[pd.DataFrame, list[str]]:
        """Search restricted to a single category."""
        q_vec, oov_tokens = self._query_vector(query)
        if np.allclose(q_vec, 0):
            empty_df = pd.DataFrame(columns=["id", "query", "category", "similarity_score"])
            return empty_df, oov_tokens

        mask = self.reference_df["category"] == category
        cat_matrix = self.reference_matrix[mask]
        cat_df = self.reference_df[mask].reset_index(drop=True)

        if len(cat_df) == 0:
            empty_df = pd.DataFrame(columns=["id", "query", "category", "similarity_score"])
            return empty_df, oov_tokens

        sims = cosine_similarity(q_vec.reshape(1, -1), cat_matrix)[0]
        indices = np.argsort(sims)[::-1][:top_k]

        results = cat_df.iloc[indices].copy()
        results["similarity_score"] = sims[indices]
        return results[["id", "query", "category", "similarity_score"]].reset_index(drop=True), oov_tokens

    def predict_category(self, query: str) -> str:
        """Predict category via top-1 nearest-neighbor."""
        results, _ = self.search(query, top_k=1)
        if results.empty:
            return "Unknown"
        return results.iloc[0]["category"]


# ---------------------------------------------------------------------------
# Factory Loaders
# ---------------------------------------------------------------------------

def load_search_engine(model_type: str = "fasttext", models_dir: str | None = None) -> BankingSemanticSearch:
    """Load specified model (fasttext or word2vec) and return BankingSemanticSearch instance."""
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    model_file = "fasttext_banking.model" if model_type == "fasttext" else "word2vec_banking.model"
    model_path = os.path.join(models_dir, model_file)
    train_path = os.path.join(models_dir, "train.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_models.py first.")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Run train_models.py first.")

    reference_df = pd.read_csv(train_path)

    if model_type == "fasttext":
        model = FastText.load(model_path)
    else:
        model = Word2Vec.load(model_path)

    # Recompute reference vectors if pre-built npy not available for word2vec
    vectors_file = f"train_vectors_{model_type}.npy"
    vectors_path = os.path.join(models_dir, vectors_file)

    if os.path.exists(vectors_path):
        reference_matrix = np.load(vectors_path)
    else:
        # Build reference matrix on the fly
        train_queries = reference_df["query_clean"] if "query_clean" in reference_df.columns else reference_df["query"]
        reference_matrix = np.array([
            get_sentence_vector(model, tokenize(q))
            for q in train_queries
        ])

    return BankingSemanticSearch(
        model=model,
        reference_df=reference_df,
        reference_matrix=reference_matrix,
        tokenizer=tokenize,
        vectorizer=get_sentence_vector,
        model_type=model_type,
    )


def load_all_engines(models_dir: str | None = None) -> dict[str, BankingSemanticSearch]:
    """Load both FastText and Word2Vec engines for comparison mode."""
    return {
        "fasttext": load_search_engine("fasttext", models_dir),
        "word2vec": load_search_engine("word2vec", models_dir),
    }
