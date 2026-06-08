"""
STAGE 5 — ASSEMBLE THE TABLE
============================

Joins the computed pilot variables into one analysis-ready CSV.

Inputs:
  relations/relations.json        ground truth + covariates
  frequency/frequencies.json      variable ① frequency
  representation/rep_quality.json variable ② representation quality
  mimicry/mimicry_scores.json     variable ③ mimicry outcome

Output:
  data/pilot_data.csv

One row = one relation.
"""

import csv
import json
from pathlib import Path


def load_json(path):
    """Load a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_relations(path):
    """
    Load relations from relations.json.

    Your relations file is a dictionary with documentation fields,
    and the actual relation list is stored under data["relations"].
    """
    data = load_json(path)

    if not isinstance(data, dict):
        raise TypeError("relations.json should be a dictionary with a 'relations' key.")

    if "relations" not in data:
        raise KeyError("relations.json is missing the 'relations' key.")

    relations = data["relations"]

    if not isinstance(relations, list):
        raise TypeError("data['relations'] should be a list.")

    return relations


def normalize_by_id(data, value_name):
    """
    Convert different possible JSON formats into a dictionary keyed by relation id.

    Supported formats:

    1. Already keyed by id:
       {
         "r001": 123,
         "r002": 45
       }

    2. Wrapped keyed format:
       {
         "frequencies": {
           "r001": 123,
           "r002": 45
         }
       }

    3. List of row objects:
       [
         {"id": "r001", "frequency": 123},
         {"id": "r002", "frequency": 45}
       ]
    """
    if isinstance(data, dict):
        # If it has a wrapper key, unwrap it.
        for possible_key in [value_name, value_name + "s", "scores", "data"]:
            if possible_key in data and isinstance(data[possible_key], (dict, list)):
                return normalize_by_id(data[possible_key], value_name)

        # Otherwise assume it is already id -> value or id -> object.
        return data

    if isinstance(data, list):
        out = {}
        for row in data:
            if "id" in row:
                relation_id = row["id"]
            elif "relation_id" in row:
                relation_id = row["relation_id"]
            else:
                raise KeyError(f"Row is missing 'id' or 'relation_id': {row}")

            if value_name in row:
                out[relation_id] = row[value_name]
            else:
                out[relation_id] = row

        return out

    raise TypeError(f"Cannot normalize JSON object of type {type(data)}")


def get_value(mapping, relation_id, field_name):
    """
    Extract a value for one relation.

    Handles both:
      mapping["r001"] = 123
    and:
      mapping["r001"] = {"frequency": 123}
    """
    if relation_id not in mapping:
        raise KeyError(f"Missing {field_name} for relation id: {relation_id}")

    value = mapping[relation_id]

    if isinstance(value, dict):
        if field_name not in value:
            raise KeyError(f"Relation {relation_id} exists, but lacks field '{field_name}'.")
        return value[field_name]

    return value


def assemble(
    relations_p="relations/relations.json",
    freq_p="frequency/frequencies.json",
    rep_p="representation/rep_quality.json",
    mimicry_p="mimicry/mimicry_scores.json",
    out="data/pilot_data.csv",
):
    """
    Join all stage outputs into one row-per-relation CSV.
    """
    relations = load_relations(relations_p)

    frequencies = normalize_by_id(load_json(freq_p), "frequency")
    rep_quality = normalize_by_id(load_json(rep_p), "rep_quality")
    mimicry = normalize_by_id(load_json(mimicry_p), "mimicry_score")

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "relation_id",
        "subject",
        "object",
        "frequency",
        "rep_quality",
        "mimicry_score",
        "is_causal",
        "concept_commonality",
        "relation_type",
        "true_direction",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for relation in relations:
            relation_id = relation["id"]

            row = {
                "relation_id": relation_id,
                "subject": relation["subject"],
                "object": relation["object"],
                "frequency": get_value(frequencies, relation_id, "frequency"),
                "rep_quality": get_value(rep_quality, relation_id, "rep_quality"),
                "mimicry_score": get_value(mimicry, relation_id, "mimicry_score"),
                "is_causal": relation["is_causal"],
                "concept_commonality": relation["concept_commonality"],
                "relation_type": relation["relation_type"],
                "true_direction": relation["true_direction"],
            }

            writer.writerow(row)

    print(f"Wrote {len(relations)} rows to {output_path}")


if __name__ == "__main__":
    assemble()