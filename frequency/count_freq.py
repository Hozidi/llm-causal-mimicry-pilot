"""
STAGE 1 — CO-OCCURRENCE FREQUENCY  (variable ①, the independent variable)
=========================================================================
WHAT THIS DOES:
  For each relation, count how often `subject` and `object` CO-OCCUR in the
  model's pretraining corpus (Dolma, for OLMo). This is the paper's exact method.

WHY IT'S COUNTABLE (not estimated):
  We use an OPEN-DATA model. The corpus is searchable, so frequency is a real
  count — NOT the noisy regression-prediction you'd be stuck with on a closed
  model (e.g. Mistral). This is the whole reason for choosing OLMo/Pythia.

TOOL:
  allenai BatchSearch (exact per-batch counts) or WIMBD (full-corpus counts).
  WIMBD is simpler for a final checkpoint; BatchSearch if you want per-step.

INPUT:  relations/relations.json
OUTPUT: frequency/frequencies.json  -> { relation_id: cooccurrence_count }

NOTE: subject-object co-occurrence is the paper's validated proxy for how often
      the *fact/relation* is mentioned (Elsahar et al.). Use co-occurrence, not
      subject-only or object-only frequency (the paper shows co-occ correlates best).
"""

# import json
# from batchsearch import count_cooccurrence   # allenai tool

def count_frequencies(relations_path, corpus):
    """Return {relation_id: cooccurrence_count} by searching the open corpus."""
    # TODO:
    # 1. load relations.json
    # 2. for each relation: count = count_cooccurrence(corpus, subject, object)
    # 3. save to frequency/frequencies.json
    raise NotImplementedError("wire up BatchSearch/WIMBD over Dolma")


if __name__ == "__main__":
    # count_frequencies("relations/relations.json", corpus="dolma")
    pass
