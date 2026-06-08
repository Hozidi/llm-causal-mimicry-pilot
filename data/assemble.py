"""
STAGE 5 — ASSEMBLE THE TABLE
============================
WHAT THIS DOES:
  Joins the three computed variables (+ covariates) into ONE analysis-ready
  table: one row per relation. THIS is pilot_data.csv.

CLARIFICATION (you asked):
  pilot_data.csv is NOT the elicited data. The raw elicitations live in
  elicitation/raw_responses.json. This file is the FINAL joined table where
  all three variables are already computed numbers.

INPUTS:
  relations/relations.json        (covariates, ground truth)
  frequency/frequencies.json      (variable ①)
  representation/rep_quality.json (variable ②)
  mimicry/mimicry_scores.json     (variable ③)

OUTPUT: data/pilot_data.csv with columns:
  relation_id, frequency, rep_quality, mimicry_score,
  is_causal, concept_commonality, relation_type
"""

# import json, csv

def assemble(relations_p, freq_p, rep_p, mimicry_p, out="data/pilot_data.csv"):
    """Join all stages into one row-per-relation CSV."""
    # TODO: load the four json files, join on relation_id, write CSV
    raise NotImplementedError("join the four artifacts on relation_id")


if __name__ == "__main__":
    pass
