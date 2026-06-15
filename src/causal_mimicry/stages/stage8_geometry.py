"""Stage 8 - Dose-shift geometry.

When intensity changes the hidden state, does that shift direction itself encode
whether a relation is causal? We learn a direction from training-split high-minus-base
shifts (causal mean minus spurious mean) and project held-out shifts onto it, scoring
the causal-vs-spurious separation by AUC under GroupKFold.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from ..config import GAP_COLOR
from ..data import x_for
from ..hidden import hs_prompt


def run(ctx) -> pd.DataFrame:
    layers = ctx.selected_layers
    geo_items = [it for it in ctx.items if it["category"] in ("causal", "spurious")]

    gX, gids, glvls, gcat = [], [], [], []
    for it in geo_items:
        for lvl in ["base", "high"]:
            gX.append(ctx.extractor.hidden_for(hs_prompt(x_for(it, lvl), it["y_text"])).astype(np.float16))
            gids.append(it["id"]); glvls.append(lvl); gcat.append(it["category"])
    gX = np.stack(gX); gids = np.array(gids); glvls = np.array(glvls); gcat = np.array(gcat)
    idx = {(gids[r], glvls[r]): r for r in range(len(gids))}

    geo = []
    for li, L in enumerate(layers):
        rows_d = []
        for rid in np.unique(gids):
            if (rid, "base") not in idx or (rid, "high") not in idx:
                continue
            hb = gX[idx[(rid, "base")], li, :].astype(np.float32)
            hh = gX[idx[(rid, "high")], li, :].astype(np.float32)
            cat = gcat[gids == rid][0]
            rows_d.append({"id": rid, "category": cat, "delta": hh - hb})
        sub = pd.DataFrame(rows_d)
        sub = sub[sub.category.isin(["causal", "spurious"])].reset_index(drop=True)
        if sub.category.nunique() < 2:
            continue
        gkf = GroupKFold(n_splits=min(5, sub.id.nunique()))
        pc, ps = [], []
        for tr, te in gkf.split(sub, sub.category, groups=sub.id):
            td = sub.iloc[tr]
            dC = np.mean(np.stack(td[td.category == "causal"]["delta"].values), 0)
            dS = np.mean(np.stack(td[td.category == "spurious"]["delta"].values), 0)
            direction = dC - dS
            direction /= (np.linalg.norm(direction) + 1e-8)
            for _, rr in sub.iloc[te].iterrows():
                (pc if rr.category == "causal" else ps).append(float(np.dot(rr["delta"], direction)))
        if len(pc) > 2 and len(ps) > 2:
            u, p = stats.mannwhitneyu(pc, ps, alternative="greater")
            auc = u / (len(pc) * len(ps))
            geo.append({"layer": int(L), "causal_proj": round(float(np.mean(pc)), 3),
                        "spurious_proj": round(float(np.mean(ps)), 3),
                        "p": round(p, 4), "auc": round(auc, 3)})
            print(f"  L{L}: causal proj {np.mean(pc):+.3f} vs spurious {np.mean(ps):+.3f} "
                  f"p={p:.4f} AUC={auc:.3f}")

    GEO = pd.DataFrame(geo)
    GEO.to_csv(ctx.tag("stage8_dose_geometry.csv"), index=False)
    if len(GEO):
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(GEO.layer, GEO.auc, "o-", color=GAP_COLOR, lw=2, ms=7)
        ax.axhline(0.5, ls="--", color="grey", alpha=0.6, label="chance")
        ax.set_xlabel("layer"); ax.set_ylabel("held-out AUC (causal>spurious shift)")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 8: dose-shift geometry"); ax.legend()
        plt.tight_layout(); plt.savefig(ctx.tag("stage8_dose_geometry.png")); plt.close(fig)
    return GEO
