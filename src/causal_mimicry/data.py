"""Data loading and the polarity-locked construction of the experimental sample.

``polarity`` in the dataset is authoritative and is NOT re-derived here. Stages 1-9
run on the 300 positive-polarity causal relations (plus their reversals and the
spurious forks); the 100 negative-polarity relations are held back for the
polarity-dissociation test.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from .config import LEVELS, ORD_LEVELS  # noqa: F401  (re-exported for convenience)


# --------------------------------------------------------------------------- #
# Intensity accessors
# --------------------------------------------------------------------------- #
def x_for(item: dict, level: str) -> str:
    """Graded cause phrasing (Dose-A): base / low / moderate / high."""
    if level == "base":
        return item["x_text"]
    return item.get(f"x_{level}", item["x_text"])


def x_ordinal(item: dict, level: str) -> str:
    """Ordinal cause phrasing (Dose-B): 'less X' / 'X' / 'more X'."""
    x = item["x_text"]
    if level == "base":
        return x
    return f"less {x}" if level == "less" else f"more {x}"


@dataclass
class Dataset:
    """The fully constructed experimental sample for one run."""
    items: list[dict]            # main set: causal (positive) + anti-causal + spurious
    rev: list[dict]              # fork originals + reversals (word-order baseline)
    neg_items: list[dict]        # negative-polarity causal (held for dissociation test)
    n_causal_pos: int
    n_causal_neg: int

    def summary(self) -> dict:
        return {
            "main_items": len(self.items),
            "by_category": dict(Counter(it["category"] for it in self.items)),
            "reversal_items": len(self.rev),
            "negative_polarity_held": len(self.neg_items),
        }


def load_dataset(causal_path: str, spurious_path: str) -> Dataset:
    """Load the two JSON files and build the polarity-locked sample.

    Mirrors notebook section 2 exactly.
    """
    causal_raw = json.load(open(causal_path))["relations"]
    spurious_raw = json.load(open(spurious_path))["relations"]

    for r in causal_raw:
        assert "polarity" in r, f"{r['id']} missing polarity field"
    causal_pos = [r for r in causal_raw if r["polarity"] == "positive"]
    causal_neg = [r for r in causal_raw if r["polarity"] == "negative"]

    items: list[dict] = []
    # causal (positive polarity, forward arrow)
    for r in causal_pos:
        items.append({
            "id": r["id"], "pair_id": r["pair_id"], "category": "causal",
            "x_text": r["x_text"], "y_text": r["y_text"],
            "true_direction": "subject->object", "polarity": "positive",
            "x_low": r.get("x_low"), "x_moderate": r.get("x_moderate"), "x_high": r.get("x_high"),
        })
    # anti-causal: same relations reversed (X/Y swapped) -> tests if the model only
    # tracks first-mention word order.
    for r in causal_pos:
        items.append({
            "id": f"anti_{r['id']}", "pair_id": f"anti_{r['pair_id']}", "category": "anti-causal",
            "x_text": r["y_text"], "y_text": r["x_text"],
            "true_direction": "object->subject", "polarity": "positive",
            "x_low": f"slight {r['y_text']}", "x_moderate": f"moderate {r['y_text']}",
            "x_high": f"strong {r['y_text']}",
        })
    # spurious / fork (common cause, no direct edge)
    for r in spurious_raw:
        items.append({
            "id": r["id"], "pair_id": r["pair_id"], "category": "spurious",
            "x_text": r["x_text"], "y_text": r["y_text"],
            "true_direction": "none", "polarity": "none",
            "x_low": f"low {r['x_text']}", "x_moderate": f"moderate {r['x_text']}",
            "x_high": f"high {r['x_text']}",
            "confounder_z": r.get("confounder_z"),
            "keep_for_headline": r.get("keep_for_headline", True),
        })

    # negative-polarity causal, held for the dissociation test (Delta_more should be < 0)
    neg_items = [{
        "id": r["id"], "pair_id": r["pair_id"], "category": "causal", "polarity": "negative",
        "x_text": r["x_text"], "y_text": r["y_text"],
        "x_low": r.get("x_low"), "x_moderate": r.get("x_moderate"), "x_high": r.get("x_high"),
    } for r in causal_neg]

    # word-order control: fork originals + reversals
    rev: list[dict] = []
    for r in spurious_raw:
        pid = r["pair_id"]
        rev += [
            {"id": f"{pid}_orig", "pair_id": pid, "category": "spurious",
             "x_text": r["x_text"], "y_text": r["y_text"]},
            {"id": f"{pid}_rev", "pair_id": pid, "category": "anti_spurious",
             "x_text": r["y_text"], "y_text": r["x_text"]},
        ]

    return Dataset(
        items=items, rev=rev, neg_items=neg_items,
        n_causal_pos=len(causal_pos), n_causal_neg=len(causal_neg),
    )
