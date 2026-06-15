"""Hidden-state extraction and the probing machinery (Stages 4-6, 8).

All probes are leakage-controlled by relation id (``GroupKFold``) and benchmarked
against a TF-IDF lexical baseline and a random-label control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .stats import RNG


def hs_prompt(x: str, y: str) -> str:
    """Prompt whose final-token hidden state we read for probing."""
    return f"Consider the variables: {x} and {y}. The relation between them is"


def selected_layers_for(n_layers: int) -> list[int]:
    """Five evenly spaced layers (plus first/last) to probe."""
    return sorted(set([1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]))


class HiddenExtractor:
    """Reads last-token hidden states at the selected layers for a list of items."""

    def __init__(self, model, tok, selected_layers: list[int]):
        self.model = model
        self.tok = tok
        self.selected_layers = selected_layers

    @torch.no_grad()
    def hidden_for(self, text: str) -> np.ndarray:
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        hs = self.model(**ids).hidden_states
        last = ids["attention_mask"][0].sum().item() - 1
        return np.stack([hs[L][0, last, :].float().cpu().numpy() for L in self.selected_layers])

    def extract(self, items: list[dict], fields: list[str]):
        """Return (X[N, n_layers, d] float16, meta DataFrame)."""
        X, meta = [], []
        for it in items:
            x, y = it["x_text"], it["y_text"]
            X.append(self.hidden_for(hs_prompt(x, y)).astype(np.float16))
            meta.append({f: it.get(f) for f in fields} | {"x_text": x, "y_text": y})
        return np.stack(X), pd.DataFrame(meta)


def oof_predictions(X, y, groups, shuffle: bool = False) -> np.ndarray:
    """Out-of-fold correctness per item using a StandardScaler->PCA->LogReg probe.

    ``shuffle=True`` permutes labels first — the random-label control.
    """
    if shuffle:
        y = RNG.permutation(y)
    k = min(5, len(np.unique(groups)))
    gkf = GroupKFold(k)
    correct = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups=groups):
        nc = max(2, min(128, len(tr) - 1, X.shape[1]))
        clf = make_pipeline(
            StandardScaler(), PCA(n_components=nc, random_state=0),
            LogisticRegression(max_iter=1000),
        )
        clf.fit(X[tr], y[tr])
        correct[te] = (clf.predict(X[te]) == y[te]).astype(float)
    return correct


def tfidf_acc(texts, y, groups) -> float:
    """GroupKFold accuracy of a TF-IDF bag-of-words probe (the lexical baseline)."""
    k = min(5, len(np.unique(groups)))
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000),
    )
    return cross_val_score(
        pipe, texts, y, groups=groups, cv=GroupKFold(k),
        scoring="accuracy", n_jobs=-1,
    ).mean()
