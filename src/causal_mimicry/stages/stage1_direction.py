"""Stage 1 - Behavioural direction.

Does the model's output prefer the true causal arrow? We score the forward/reverse/
none continuations, take a directional margin logP(fwd) - logP(rev) per relation, and
compare causal / anti-causal / spurious with group-bootstrap CIs over relation ids.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import CAT2KEY, PALETTE
from ..stats import grouped_bootstrap_ci


def run(ctx) -> pd.DataFrame:
    score_dirs = ctx.scorer.score_dirs
    rows, skipped = [], 0
    for it in ctx.items:
        cat = it.get("category")
        if cat not in CAT2KEY:
            skipped += 1
            continue
        p = score_dirs(it["x_text"], it["y_text"])
        pf, pr, pn = float(p["forward"]), float(p["reverse"]), float(p["none"])
        rows.append({
            "model": ctx.cfg.model_key, "id": it["id"], "pair_id": it["pair_id"], "category": cat,
            "p_forward": pf, "p_reverse": pr, "p_none": pn,
            "margin_dir": np.log(max(pf, 1e-9)) - np.log(max(pr, 1e-9)),
            "p_correct": float(p[CAT2KEY[cat]]), "correct_key": CAT2KEY[cat],
        })
    base_df = pd.DataFrame(rows)
    if base_df.empty:
        raise ValueError("base_df is empty - check item categories and CAT2KEY.")
    base_df.to_csv(ctx.tag("behavioural_direction.csv"), index=False)
    ctx.artifacts["base_df"] = base_df
    print(f"Stage 1: {len(base_df)} rows" + (f" ({skipped} skipped)" if skipped else ""))

    cats = ["causal", "anti-causal", "spurious"]
    ms, los, his = [], [], []
    for cat in cats:
        sub = base_df[base_df.category == cat]
        m, lo, hi = grouped_bootstrap_ci(sub, "margin_dir", "pair_id")
        ms.append(m); los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    yerr = [np.array(ms) - np.array(los), np.array(his) - np.array(ms)]
    ax.bar(range(3), ms, yerr=yerr, capsize=4, color=[PALETTE[c] for c in cats])
    ax.axhline(0, ls="--", color="grey", alpha=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(cats)
    ax.set_ylabel("directional margin logP(fwd)-logP(rev) [95% CI]")
    ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 1: behavioural direction")
    plt.tight_layout(); plt.savefig(ctx.tag("stage1_direction.png")); plt.close(fig)
    return base_df
