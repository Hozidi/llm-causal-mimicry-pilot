"""
STAGE 4 — CAUSAL-MIMICRY SCORE

Turns raw elicited A/B/C token probabilities into per-relation mimicry scores.

Input:
  elicitation/raw_responses.json
  relations/relations.json

Output:
  mimicry/mimicry_scores.json

Core rule:
  This computes the outcome variable Y from elicited behaviour only.
  It does not use frequency or representation quality.
"""

import json
from pathlib import Path


DIRECTION_CHANCE = 1 / 3
DISCRIMINATION_CHANCE = 1 / 2


def load_json(path):
    """Load a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    """Save a Python object as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_relations(path):
    """
    Load relations from relations.json.

    Your relations file is a dictionary. The actual relation rows live under:
      data["relations"]
    """
    data = load_json(path)

    if not isinstance(data, dict):
        raise TypeError("relations.json should be a dictionary.")

    if "relations" not in data:
        raise KeyError("relations.json is missing the 'relations' key.")

    relations = data["relations"]

    if not isinstance(relations, list):
        raise TypeError("data['relations'] should be a list.")

    return relations


def normalize_above_chance(observed, chance):
    """
    Normalize performance above chance.

    Formula:
      normalized = (observed - chance) / (1 - chance)

    Meaning:
      0 = chance-level
      1 = perfect
      <0 = worse than chance
    """
    return (observed - chance) / (1 - chance)


def renormalize_abc(sample):
    """
    Renormalize p_A, p_B, p_C over the valid A/B/C answer space.

    The model may assign probability to other tokens.
    For the causal-choice score, we condition on the valid choices A/B/C.

    Input sample expected:
      {"p_A": ..., "p_B": ..., "p_C": ...}

    Optional:
      If the sample already contains "valid_mass", we ignore it and recompute it.
    """
    try:
        raw_a = float(sample["p_A"])
        raw_b = float(sample["p_B"])
        raw_c = float(sample["p_C"])
    except KeyError as e:
        raise KeyError(f"Sample missing required key {e}: {sample}") from e

    valid_mass = raw_a + raw_b + raw_c

    if valid_mass <= 0:
        raise ValueError(f"p_A + p_B + p_C must be > 0. Got sample: {sample}")

    return {
        "p_A": raw_a / valid_mass,
        "p_B": raw_b / valid_mass,
        "p_C": raw_c / valid_mass,
        "valid_mass": valid_mass,
        "invalid_mass": max(0.0, 1.0 - valid_mass),
    }


def correct_direction_token(relation):
    """
    Return the correct direction token for causal relations.

    A = subject causes object
    B = object causes subject
    C = no causal relationship
    """
    true_direction = relation["true_direction"]

    if true_direction == "subject->object":
        return "A"

    if true_direction == "object->subject":
        return "B"

    if true_direction == "none":
        return None

    raise ValueError(f"Unknown true_direction: {true_direction}")


def mean(values):
    """Compute the arithmetic mean."""
    if not values:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(values) / len(values)


def score_one_relation(relation, raw_samples):
    """
    Score one relation using probability mass, not argmax choices.

    Direction score:
      Only for causal relations.
      observed = mean probability on the correct direction token.

    Discrimination score:
      For causal relations:
        observed = mean probability mass on A+B.
      For non-causal relations:
        observed = mean probability mass on C.
    """
    relation_id = relation["id"]

    if not raw_samples:
        raise ValueError(f"No raw response samples for relation {relation_id}")

    samples = [renormalize_abc(sample) for sample in raw_samples]

    is_causal = bool(relation["is_causal"])
    direction_token = correct_direction_token(relation)

    # Direction: only defined for genuinely causal relations.
    if is_causal:
        if direction_token == "A":
            direction_observed = mean([s["p_A"] for s in samples])
        elif direction_token == "B":
            direction_observed = mean([s["p_B"] for s in samples])
        else:
            raise ValueError(
                f"Relation {relation_id} is causal but has no valid direction token."
            )

        mimicry_direction = normalize_above_chance(
            direction_observed,
            DIRECTION_CHANCE,
        )
    else:
        direction_observed = None
        mimicry_direction = None

    # Discrimination: causal vs non-causal.
    if is_causal:
        discrimination_observed = mean([
            s["p_A"] + s["p_B"]
            for s in samples
        ])
    else:
        discrimination_observed = mean([
            s["p_C"]
            for s in samples
        ])

    mimicry_discrimination = normalize_above_chance(
        discrimination_observed,
        DISCRIMINATION_CHANCE,
    )

    # Optional composite. Keep sub-scores visible in analysis.
    if mimicry_direction is None:
        mimicry_score = mimicry_discrimination
    else:
        mimicry_score = (mimicry_direction + mimicry_discrimination) / 2

    return {
        "mimicry_score": mimicry_score,
        "mimicry_direction": mimicry_direction,
        "mimicry_discrimination": mimicry_discrimination,
        "direction_observed": direction_observed,
        "discrimination_observed": discrimination_observed,
        "mean_valid_mass": mean([s["valid_mass"] for s in samples]),
        "mean_invalid_mass": mean([s["invalid_mass"] for s in samples]),
        "n_samples": len(samples),
    }


def score_mimicry(
    raw_responses_path="elicitation/raw_responses.json",
    relations_path="relations/relations.json",
    output_path="mimicry/mimicry_scores.json",
):
    """
    Turn raw elicited A/B/C probabilities into per-relation mimicry scores.
    """
    relations = load_relations(relations_path)
    raw_responses = load_json(raw_responses_path)

    scores = {}

    for relation in relations:
        relation_id = relation["id"]

        if relation_id not in raw_responses:
            raise KeyError(f"Missing raw responses for relation id: {relation_id}")

        scores[relation_id] = score_one_relation(
            relation=relation,
            raw_samples=raw_responses[relation_id],
        )

    save_json(scores, output_path)
    print(f"Wrote mimicry scores for {len(scores)} relations to {output_path}")

    return scores


if __name__ == "__main__":
    score_mimicry()