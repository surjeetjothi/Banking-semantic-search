"""
Tests for search_engine.py
"""

import os
import pytest
import numpy as np
import pandas as pd
from search_engine import tokenize, get_sentence_vector, BankingSemanticSearch, load_search_engine
from train_models import main as train_main


def test_tokenize():
    text = "How do I open a NEW savings account?!"
    tokens = tokenize(text)
    assert tokens == ["how", "do", "i", "open", "a", "new", "savings", "account"]


def test_get_sentence_vector():
    class DummyWV:
        def __getitem__(self, token):
            if token == "bank":
                return np.array([1.0, 2.0])
            elif token == "account":
                return np.array([3.0, 4.0])
            raise KeyError(token)

    class DummyModel:
        wv = DummyWV()
        vector_size = 2

    model = DummyModel()
    # In-vocab
    vec = get_sentence_vector(model, ["bank", "account"])
    np.testing.assert_array_almost_equal(vec, [2.0, 3.0])

    # OOV
    vec_oov = get_sentence_vector(model, ["unknown_word"])
    np.testing.assert_array_almost_equal(vec_oov, [0.0, 0.0])


def test_banking_semantic_search():
    df = pd.DataFrame([
        {"id": 1, "query": "How to block lost card?", "category": "Card Services"},
        {"id": 2, "query": "What is mortgage interest rate?", "category": "Loans"},
    ])
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    class DummyModel:
        wv = {"card": np.array([1.0, 0.0]), "mortgage": np.array([0.0, 1.0])}
        vector_size = 2

    def dummy_tokenize(t):
        return t.lower().replace("?", "").split()

    def dummy_vectorizer(m, tokens):
        vecs = [m.wv[t] for t in tokens if t in m.wv]
        if not vecs:
            return np.zeros(m.vector_size)
        return np.mean(vecs, axis=0)

    engine = BankingSemanticSearch(
        model=DummyModel(),
        reference_df=df,
        reference_matrix=matrix,
        tokenizer=dummy_tokenize,
        vectorizer=dummy_vectorizer,
    )

    # Search card query
    res, oov = engine.search("I lost my card", top_k=2)
    assert len(res) == 2
    assert res.iloc[0]["category"] == "Card Services"

    # Search by category
    res_cat, oov_cat = engine.search_by_category("mortgage", category="Loans", top_k=2)
    assert len(res_cat) == 1
    assert res_cat.iloc[0]["id"] == 2


    # Predict category
    pred = engine.predict_category("card")
    assert pred == "Card Services"
