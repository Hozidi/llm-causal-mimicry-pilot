"""Stage 3 - Lexical confound controls.

3A: is the graded-dose rise just the intensity adverb's unigram frequency? Residualise
    p(correct) on Zipf frequency and check the curve survives, per category.
3B (the hinge): is the more/less confidence change driven by lexical co-occurrence?
    Correlate the per-relation change in confidence with the change in a purely
    non-causal co-occurrence likelihood. High causal r => behavioural dose is largely
    lexical => the real evidence must come from representations (Stages 4-9).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from ..config import LEVELS, ORD_LEVELS, PALETTE
from ..data import x_ordinal
from ..stats import grouped_bootstrap_ci, pearson_boot

_ORDER = {"base": 0, "low": 1, "moderate": 2, "high": 3}


def run_3a(ctx, gdf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Frequency control for graded dose. Needs the Stage 2A table (``gdf``)."""
    gdf = gdf if gdf is not None else ctx.artifacts["gdf"]
    gdf_r = gdf.copy()
    lr = LinearRegression().fit(gdf_r[["word_freq"]].values, gdf_r["p_correct"].values)
    gdf_r["p_resid"] = gdf_r["p_correct"].values - lr.predict(gdf_r[["word_freq"]].values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [(axes[0], "p_correct", "mean p(correct) [95% CI]", "raw"),
              (axes[1], "p_resid", "residualised p(correct) [95% CI]", "frequency-residualised")]
    for ax, col, ylab, ttl in panels:
        for cat in ["causal", "anti-causal", "spurious"]:
            sub = gdf_r[gdf_r.category == cat]
            ms, los, his = [], [], []
            for lvl in LEVELS:
                m, lo, hi = grouped_bootstrap_ci(sub[sub.level == lvl], col, "pair_id")
                ms.append(m); los.append(lo); his.append(hi)
            xs = [_ORDER[l] for l in LEVELS]
            ax.plot(xs, ms, "o-", color=PALETTE[cat], lw=2.2, ms=7, label=cat)
            ax.fill_between(xs, los, his, color=PALETTE[cat], alpha=0.15)
        ax.axhline(1 / 3 if col == "p_correct" else 0, ls="--", color="grey", alpha=0.6)
        ax.set_xticks(list(_ORDER.values())); ax.set_xticklabels(LEVELS)
        ax.set_xlabel("cause intensity"); ax.set_ylabel(ylab)
        ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 3A \u00b7 {ttl}"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(ctx.tag("stage3A_graded_freqcontrol.png")); plt.close(fig)
    print("Stage 3A: r(causal)~r(anti)~r(spur)>0 => fluency artifact; "
          "r(causal)>0 but r(anti)<=0 => frequency cannot explain the differential.")
    return gdf_r


def run_3b(ctx, piv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Co-occurrence control (the hinge). Needs the Stage 2B delta table (``piv``)."""
    piv = piv if piv is not None else ctx.artifacts["piv"]
    cooc = ctx.scorer.cooc_logprob

    crows = []
    for it in ctx.items:
        if it["category"] not in {"causal", "anti-causal", "spurious"}:
            continue
        for lvl in ORD_LEVELS:
            crows.append({
                "id": it["id"], "pair_id": it["pair_id"], "category": it["category"],
                "level": lvl, "cooc": cooc(x_ordinal(it, lvl), it["y_text"]),
            })
    cdf = pd.DataFrame(crows)
    cpiv = cdf.pivot_table(index=["id", "pair_id", "category"],
                           columns="level", values="cooc").reset_index()
    cpiv["delta_cooc_more"] = cpiv["more"] - cpiv["base"]
    cpiv["delta_cooc_less"] = cpiv["less"] - cpiv["base"]

    J = cpiv.merge(piv[["id", "category", "delta_more", "delta_less"]],
                   on=["id", "category"], how="inner")
    J.to_csv(ctx.tag("stage3B_cooccurrence.csv"), index=False)

    print("Stage 3B: corr( Delta_cooc , Delta_p(correct) ) for the MORE shift, per category")
    corr_rows = []
    for cat in ["causal", "anti-causal", "spurious"]:
        d = J[J.category == cat].dropna(subset=["delta_cooc_more", "delta_more"])
        if len(d) < 5:
            continue
        r, lo, hi = pearson_boot(d["delta_cooc_more"].values, d["delta_more"].values)
        verdict = ("CO-OCCURRENCE-DRIVEN" if lo > 0.3
                   else ("partial" if lo > 0 else "not co-occ-driven"))
        print(f"  {cat:12s} r={r:+.3f} [{lo:+.3f},{hi:+.3f}] -> {verdict}")
        corr_rows.append({"model": ctx.cfg.model_key, "category": cat,
                          "r": r, "r_lo": lo, "r_hi": hi, "n": len(d)})
    pd.DataFrame(corr_rows).to_csv(ctx.tag("stage3B_corr.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, cat in zip(axes, ["causal", "anti-causal", "spurious"]):
        d = J[J.category == cat].dropna(subset=["delta_cooc_more", "delta_more"])
        if len(d) < 5:
            ax.set_title(f"{cat} - n/a"); continue
        r, lo, hi = pearson_boot(d["delta_cooc_more"].values, d["delta_more"].values)
        ax.scatter(d["delta_cooc_more"], d["delta_more"], s=20, alpha=0.4, color=PALETTE[cat])
        mf, bf = np.polyfit(d["delta_cooc_more"], d["delta_more"], 1)
        xs = np.linspace(d["delta_cooc_more"].min(), d["delta_cooc_more"].max(), 50)
        ax.plot(xs, mf * xs + bf, color=PALETTE[cat], lw=2)
        ax.axhline(0, ls="--", color="grey", alpha=0.4); ax.axvline(0, ls="--", color="grey", alpha=0.4)
        ax.set_xlabel("Delta co-occurrence logP (more-base)")
        ax.set_ylabel("Delta p(correct) (more-base)")
        ax.set_title(f"{cat}\nr={r:+.3f} [{lo:+.3f},{hi:+.3f}]")
    plt.suptitle(f"{ctx.cfg.model_key} \u00b7 Stage 3B: is the more/less shift co-occurrence-driven?", y=1.02)
    plt.tight_layout(); plt.savefig(ctx.tag("stage3B_cooccurrence_scatter.png")); plt.close(fig)
    print("High causal r => behavioural dose is largely co-occurrence => go internal (Stages 4-9).")
    return J


def run(ctx):
    """Run both controls; returns (gdf_resid, cooc_join)."""
    return run_3a(ctx), run_3b(ctx)
