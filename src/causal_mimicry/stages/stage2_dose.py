"""Stage 2 - Cause-intensity (dose).

2A: graded adverbial dose (base/low/moderate/high) - does p(correct) scale with
    cause intensity, for causal relations specifically?
2B: ordinal more/less dose (frequency-matched) - does the effect survive, and is
    more-vs-less symmetric (grading is the cue) or asymmetric (direction matters)?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from wordfreq import zipf_frequency

from ..config import CAT2KEY, LEVELS, ORD_LEVELS, PALETTE
from ..data import x_for, x_ordinal
from ..stats import bootstrap_ci_mean, grouped_bootstrap_ci

_ORDER = {"base": 0, "low": 1, "moderate": 2, "high": 3}


def _freq_for(it, level):
    if level == "base":
        return 0.0
    return zipf_frequency(x_for(it, level).strip().split()[0].lower(), "en")


def run_2a(ctx) -> pd.DataFrame:
    """Graded adverbial dose. Returns the per-(item, level) table (also used by Stage 3A)."""
    score_dirs = ctx.scorer.score_dirs
    rows = []
    for it in ctx.items:
        if it["category"] not in CAT2KEY:
            continue
        ck = CAT2KEY[it["category"]]
        for lvl in LEVELS:
            p = score_dirs(x_for(it, lvl), it["y_text"])
            rows.append({
                "model": ctx.cfg.model_key, "id": it["id"], "pair_id": it["pair_id"],
                "category": it["category"], "level": lvl,
                "p_correct": p[ck], "p_forward": p["forward"], "p_reverse": p["reverse"],
                "p_none": p["none"], "word_freq": _freq_for(it, lvl),
            })
    gdf = pd.DataFrame(rows)
    gdf.to_csv(ctx.tag("dose_graded.csv"), index=False)
    ctx.artifacts["gdf"] = gdf

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cat in ["causal", "anti-causal", "spurious"]:
        sub = gdf[gdf.category == cat]
        ms, los, his = [], [], []
        for lvl in LEVELS:
            m, lo, hi = grouped_bootstrap_ci(sub[sub.level == lvl], "p_correct", "pair_id")
            ms.append(m); los.append(lo); his.append(hi)
        xs = [_ORDER[l] for l in LEVELS]
        ax.plot(xs, ms, "o-", color=PALETTE[cat], lw=2.2, ms=7, label=cat)
        ax.fill_between(xs, los, his, color=PALETTE[cat], alpha=0.15)
    ax.axhline(1 / 3, ls="--", color="grey", alpha=0.6, label="chance")
    ax.set_xticks(list(_ORDER.values())); ax.set_xticklabels(LEVELS)
    ax.set_xlabel("cause intensity"); ax.set_ylabel("p(correct) [95% CI]")
    ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 2A: graded dose"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(ctx.tag("stage2A_graded_dose.png")); plt.close(fig)
    print(f"Stage 2A: {len(gdf)} graded-dose rows")
    return gdf


def _level_index(level, polarity):
    i = {"less": 0, "base": 1, "more": 2}[level]
    return 2 - i if polarity == "negative" else i


def run_2b(ctx) -> pd.DataFrame:
    """Ordinal more/less dose. Returns the per-relation delta table (used by Stage 3B)."""
    score_dirs = ctx.scorer.score_dirs
    rows = []
    for it in ctx.items:
        if it["category"] not in CAT2KEY:
            continue
        ck = CAT2KEY[it["category"]]
        for lvl in ORD_LEVELS:
            p = score_dirs(x_ordinal(it, lvl), it["y_text"])
            rows.append({
                "model": ctx.cfg.model_key, "id": it["id"], "pair_id": it["pair_id"],
                "category": it["category"], "polarity": it.get("polarity", "none"), "level": lvl,
                "p_forward": p["forward"], "p_reverse": p["reverse"], "p_none": p["none"],
                "p_correct": p[ck],
            })
    mldf = pd.DataFrame(rows)
    mldf.to_csv(ctx.tag("dose_moreless.csv"), index=False)
    mldf["lvl_idx"] = [_level_index(l, p) for l, p in zip(mldf.level, mldf.polarity)]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cat in ["causal", "anti-causal", "spurious"]:
        sub = mldf[mldf.category == cat]
        ms, los, his = [], [], []
        for i in range(3):
            m, lo, hi = grouped_bootstrap_ci(sub[sub.lvl_idx == i], "p_correct", "pair_id")
            ms.append(m); los.append(lo); his.append(hi)
        ax.plot(range(3), ms, "o-", color=PALETTE[cat], lw=2.2, ms=7, label=cat)
        ax.fill_between(range(3), los, his, color=PALETTE[cat], alpha=0.15)
    ax.axhline(1 / 3, ls="--", color="grey", alpha=0.6, label="chance")
    ax.set_xticks(range(3)); ax.set_xticklabels(["less\u2192", "base", "more\u2192"])
    ax.set_xlabel("ordinal cause level (frequency-matched)"); ax.set_ylabel("p(correct) [95% CI]")
    ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 2B: less/base/more dose"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(ctx.tag("stage2B_moreless_dose.png")); plt.close(fig)

    # per-relation deltas + symmetry / specificity tests
    piv = mldf.pivot_table(index=["id", "pair_id", "category", "polarity"],
                           columns="level", values="p_correct").reset_index()
    piv["delta_more"] = piv["more"] - piv["base"]
    piv["delta_less"] = piv["less"] - piv["base"]
    piv.to_csv(ctx.tag("dose_moreless_deltas.csv"), index=False)
    ctx.artifacts["piv"] = piv

    print("Stage 2B: grading shift vs bare base (Delta_more, Delta_less)")
    for cat in ["causal", "anti-causal", "spurious"]:
        dm = piv[piv.category == cat]["delta_more"].dropna().values
        dl = piv[piv.category == cat]["delta_less"].dropna().values
        mm, mlo, mhi = bootstrap_ci_mean(dm)
        ml, llo, lhi = bootstrap_ci_mean(dl)
        print(f"  {cat:12s} dMore={mm:+.4f} [{mlo:+.4f},{mhi:+.4f}] | "
              f"dLess={ml:+.4f} [{llo:+.4f},{lhi:+.4f}]")
    cz = piv[piv.category == "causal"]["delta_more"].dropna().values
    sp = piv[piv.category == "spurious"]["delta_more"].dropna().values
    if len(cz) > 2 and len(sp) > 2:
        u, p = stats.mannwhitneyu(cz, sp, alternative="greater")
        print(f"  causal dMore > spurious dMore: p={p:.4f} AUC={u / (len(cz) * len(sp)):.3f}")
    czl = piv[piv.category == "causal"]["delta_less"].dropna().values
    if len(cz) > 2 and len(czl) > 2:
        _, psym = stats.mannwhitneyu(cz, czl, alternative="two-sided")
        print(f"  causal symmetry dMore vs dLess: p={psym:.4f} (NS=symmetric / SIG=asymmetric)")
    return piv


def run(ctx):
    """Run both dose experiments; returns (gdf, piv)."""
    return run_2a(ctx), run_2b(ctx)
