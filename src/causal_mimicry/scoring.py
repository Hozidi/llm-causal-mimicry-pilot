"""Continuation-likelihood scoring — the behavioural read used by every stage.

This is the canonical scorer for the whole project. We score how much the model
"believes" a full causal continuation given a neutral prompt, length-normalised over
the continuation tokens, and turn three candidate continuations (forward / reverse /
none) into probabilities with a softmax.

Note: an earlier single-token A/B/C multiple-choice scorer was tried and abandoned —
base models put almost no probability mass on the letter tokens (it failed with
valid mass ~0.001), which is why continuation likelihood is used instead.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import NEUTRAL_PROMPT, ASSOC_PROMPT


def continuations(x: str, y: str) -> dict[str, str]:
    """The three candidate completions scored for a relation."""
    return {
        "forward": f"{x} directly causes {y}",
        "reverse": f"{y} directly causes {x}",
        "none": f"{x} and {y} are not directly causally related",
    }


class Scorer:
    """Wraps a loaded model + tokenizer with the continuation-scoring methods."""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok

    @torch.no_grad()
    def cond_logprob(self, prompt: str, continuation: str) -> float:
        """logP(continuation | prompt), length-normalised over continuation tokens."""
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)
        full = self.tok(prompt + continuation, return_tensors="pt").input_ids.to(self.model.device)
        n = p_ids.shape[1]
        if full.shape[1] <= n:
            return float("-inf")
        lp = F.log_softmax(self.model(full).logits, dim=-1)
        token_lp = lp[0, n - 1:-1, :].gather(-1, full[0, n:].unsqueeze(-1)).squeeze(-1)
        return token_lp.mean().item()

    def score_dirs(self, x: str, y: str) -> dict[str, float]:
        """Probabilities for forward / reverse / none via softmax over their scores."""
        pr = NEUTRAL_PROMPT.format(x=x, y=y)
        cont = continuations(x, y)
        keys = ["forward", "reverse", "none"]
        p = F.softmax(torch.tensor([self.cond_logprob(pr, cont[k]) for k in keys]), 0)
        return {k: p[i].item() for i, k in enumerate(keys)}

    def cooc_logprob(self, x: str, y: str) -> float:
        """Non-causal co-occurrence likelihood: logP of 'X and Y' with no causal verb.

        Used by Stage 3B to test whether the dose effect is just lexical co-occurrence.
        """
        return self.cond_logprob(ASSOC_PROMPT, f"{x} and {y}")

    @torch.no_grad()
    def sentence_logprob_per_token(self, text: str) -> float:
        """Mean logP/token of a full statement — the model's own fluency for it.

        Higher (less negative) = more natural under the model. Used by the
        naturalness / frame-fluency controls.
        """
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        if ids.shape[1] < 2:
            return float("nan")
        lp = F.log_softmax(self.model(ids).logits, dim=-1)
        token_lp = lp[0, :-1, :].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        return float(token_lp.mean().item())
