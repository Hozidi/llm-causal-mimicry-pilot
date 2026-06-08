"""
STAGE 2 — REPRESENTATION QUALITY  (variable ②, the candidate MEDIATOR)
======================================================================
WHAT THIS DOES:
  For each relation, fit a Linear Relational Embedding (LRE) and compute its
  causality/faithfulness score = how cleanly/linearly (near-bijectively) the
  relation is encoded in the model's activations.

CRITICAL — WHAT THIS IS *NOT*:
  This score measures FACTUAL-ASSOCIATION LINEARITY, i.e. whether the model has
  a clean internal map subject -> object. It does NOT measure causal-direction
  reasoning. It is the MEDIATOR (variable ②), never the outcome (variable ③).
  Conflating this with mimicry would make the mediation circular -> meaningless.

METHOD:
  Hernandez et al. (2024) / Merullo et al. (2025) LRE: fit an affine transform
  W, b from subject hidden state to object hidden state over n=8 examples;
  'causality' = how often editing the subject rep flips the prediction correctly.

INPUT:  relations/relations.json + the model
OUTPUT: representation/rep_quality.json -> { relation_id: causality_score }
"""

# import json, torch
# from lre import fit_lre   # Hernandez/Merullo repo

def measure_representation_quality(relations_path, model):
    """Return {relation_id: lre_causality_score} — the mediator."""
    # TODO:
    # 1. load relations.json
    # 2. for each relation: lre = fit_lre(model, relation, n=8)
    #    score = lre.causality_score()   # soft or hard causality, per the paper
    # 3. save to representation/rep_quality.json
    raise NotImplementedError("wire up the LRE fitting code")


if __name__ == "__main__":
    # measure_representation_quality("relations/relations.json", model)
    pass
