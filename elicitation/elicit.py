"""
STAGE 3 — ELICITATION  (generate prompts, query model, collect RAW responses)
=============================================================================
WHAT THIS DOES:
  Turns each ground-truth relation into causal-direction PROMPTS, queries the
  model, and saves the RAW responses. This is the INPUT to mimicry scoring —
  it is NOT the final table and NOT the mimicry score itself.

  (You correctly flagged this as a missing step — elicitation and scoring are
   separate: generate+collect here, turn-into-a-number in mimicry/.)

PROMPT DESIGN — follow the accepted elicitation practice:
  - Ask for causal DIRECTION explicitly: "Does {X} cause {Y}, or does {Y} cause
    {X}, or neither?" (covers the correlational case via 'neither').
  - Ask for VERBALISED CONFIDENCE (Tian et al. 'just ask') — better calibrated
    than logits on instruct models; needed for the confidence-coupling signal.
  - STATELESS: fresh context per query (blocks choice-supportive bias, Kumaran).
  - SAMPLE k times per item (e.g. k=20) -> a distribution per relation, not a point.
  - Base the phrasing on Wan et al.'s causal-discovery elicitation review.

INPUT:  relations/relations.json + the model
OUTPUT: elicitation/raw_responses.json
        -> { relation_id: [ {sample_i: direction_choice, confidence}, ... ] }
"""

# import json
# from model_api import query   # OLMo inference wrapper

DIRECTION_PROMPT = (
    "Consider '{subject}' and '{object}'.\n"
    "Does {subject} cause {object}, does {object} cause {subject}, "
    "or is there no causal relationship?\n"
    "First consider each possibility, then answer with one of: "
    "'A causes B', 'B causes A', 'neither', and your confidence (0-100%)."
)

def elicit(relations_path, model, k=20):
    """Query the model k times per relation, stateless; collect raw direction+confidence."""
    # TODO:
    # 1. load relations.json
    # 2. for each relation, for i in range(k):
    #       resp = query(model, DIRECTION_PROMPT.format(**relation), fresh_context=True)
    #       parse (direction_choice, confidence) from resp
    # 3. save list per relation to elicitation/raw_responses.json
    raise NotImplementedError("wire up stateless k-sampling against OLMo")


if __name__ == "__main__":
    # elicit("relations/relations.json", model, k=20)
    pass
