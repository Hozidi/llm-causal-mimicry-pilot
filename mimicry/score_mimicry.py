"""
STAGE 4 — CAUSAL-MIMICRY SCORE  (variable ③, the OUTCOME Y) — *** THE CONTRIBUTION ***
=====================================================================================
THIS IS THE ONE PIECE THAT IS YOURS. There is NO off-the-shelf paper that hands
you a "causal mimicry score." Defining it well IS the novel contribution. Do not
go searching for a paper that scores this — it does not exist, and that absence
is exactly why the work matters.

WHAT THIS DOES:
  Turns the RAW elicited responses (elicitation/raw_responses.json) into ONE
  number per relation: how well the model's causal-DIRECTION behaviour matches
  the ground truth.

THE NON-NEGOTIABLE RULE:
  This score MUST be computable WITHOUT reference to variable ② (LRE quality).
  If Y depends on rep_quality, the mediation freq->rep->mimicry is circular and
  the whole result is meaningless. Y is scored purely from elicited BEHAVIOUR
  vs. GROUND TRUTH. Keep it independent.

==============================  LOCKED DEFINITION  ==============================
ELICITATION FORMAT (confirmed from Long et al. 2024, arXiv:2405.13551; Susanti &
Färber 2504.10936): THREE-WAY pairwise choice, context reset per query, sample k:
    (A) {subject} causes {object}
    (B) {object} causes {subject}
    (C) no causal relationship  [= correlational; correlation has no direction]
Read the (A)/(B)/(C) token probabilities. k samples -> a distribution per edge.

TWO SUB-SCORES (kept separate — they measure different competencies):

  1) DIRECTION sub-score  (on genuinely causal edges, is_causal=true)
       observed = mean over k of P(correct direction token)
                  [mean-LOG-prob or mean-prob; mean-log is size-invariant across
                   multi-edge graphs — do NOT use the raw product, it collapses to 0]
       chance   = 1/3  (three options: A, B, C)
       -> measures: does it orient cause->effect correctly.

  2) DISCRIMINATION sub-score  (across causal AND correlational edges)
       observed = accuracy of the causal-vs-"neither" call
                  (assert causation when is_causal=true; say C when is_causal=false)
       chance   = 1/2  (binary: causal vs not)
       -> measures: does it distinguish correlation from causation.

ABOVE-CHANCE NORMALIZATION (makes the two sub-scores comparable on one 0-1 scale):
       normalized = (observed - chance) / (1 - chance)
       0 = no better than guessing,  1 = perfect,  <0 = worse than guessing.
       Applied per sub-score with its OWN chance (0.33 for direction, 0.5 for disc.).

WHY normalize: the two sub-scores have different floors (0.33 vs 0.5); raw accuracies
aren't comparable. Normalizing to "fraction of the chance->perfect gap captured"
puts both on the same skill scale so they can share an axis in the mediation/figure.

INDEPENDENCE CHECK (non-negotiable): both sub-scores come ONLY from elicited
behaviour (the model's stated A/B/C + probabilities) vs. ground truth. They never
touch the LRE representation metric (variable 2). -> mediation stays non-circular.

  mimicry_score(relation) = { "direction": normalized_direction_subscore,
                              "discrimination": normalized_discrimination_subscore }
  (report BOTH; optionally a composite, but keep the two visible — which sub-score
   frequency predicts is itself a finding.)
=================================================================================

INPUT:  elicitation/raw_responses.json + relations/relations.json (for ground truth)
OUTPUT: mimicry/mimicry_scores.json -> { relation_id: mimicry_score }
"""

# import json

def score_mimicry(raw_responses_path, relations_path):
    """Turn raw elicited responses into per-relation mimicry sub-scores (variable 3).

    Returns {relation_id: {"direction": float, "discrimination": float}}.
    Both normalized via (observed - chance)/(1 - chance). Independent of variable 2.
    """
    # TODO (spec is locked above):
    # 1. load raw_responses.json (k samples of A/B/C + token probs) + relations.json
    # 2. DIRECTION sub-score (is_causal=true edges):
    #       observed = mean over k samples of P(correct direction token)   # mean-log ok
    #       norm_dir = (observed - 1/3) / (1 - 1/3)
    # 3. DISCRIMINATION sub-score (all edges):
    #       observed = accuracy of causal-vs-"neither" call vs is_causal
    #       norm_disc = (observed - 1/2) / (1 - 1/2)
    # 4. save {relation_id: {"direction": norm_dir, "discrimination": norm_disc}}
    #    to mimicry/mimicry_scores.json
    raise NotImplementedError("implement against the LOCKED DEFINITION above")


if __name__ == "__main__":
    # score_mimicry("elicitation/raw_responses.json", "relations/relations.json")
    pass
