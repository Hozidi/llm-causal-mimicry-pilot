"""Stage 5 - Word-order baseline + gap test ("word order isn't enough").

Could the decodable "direction" just be word order? Flipping X/Y in a fork is the same
word-order operation with no causal content. We probe that reversal task and test
whether causal-vs-anti direction accuracy exceeds the fork-reversal floor, via an
unpaired group-bootstrap gap (the two tasks use different items).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..hidden import oof_predictions, tfidf_acc
from ..stats import group_bootstrap_accuracy, unpaired_gap_distribution


def run(ctx) -> dict:
    """Needs Stage 4 in ctx.artifacts (PM, boot_store, Xrev, mrev). Returns the gap result."""
    PM = ctx.artifacts["PM"]
    boot_store = ctx.artifacts["boot_store"]
    Xr, Mr, layers = ctx.artifacts["Xrev"], ctx.artifacts["mrev"], ctx.artifacts["layers"]

    yr, gr, txr = Mr["category"].values, Mr["pair_id"].values, Mr["text"].values
    tbr = tfidf_acc(txr, yr, gr)
    rev_rows, rev_boot = [], {}
    for li, L in enumerate(layers):
        X = Xr[:, li, :].astype("float32")
        corr = oof_predictions(X, yr, gr)
        rcorr = oof_predictions(X, yr, gr, shuffle=True)
        acc, lo, hi, boots = group_bootstrap_accuracy(corr, gr)
        racc, *_ = group_bootstrap_accuracy(rcorr, gr)
        rev_boot[int(L)] = boots
        rev_rows.append({
            "model": ctx.cfg.model_key, "task": "spurious_vs_antispurious", "layer": int(L),
            "acc": round(acc, 3), "acc_lo": round(lo, 3), "acc_hi": round(hi, 3),
            "tfidf": round(tbr, 3), "random": round(racc, 3),
        })
    RV = pd.DataFrame(rev_rows)
    RV.to_csv(ctx.tag("reversal.csv"), index=False)

    ca = PM[PM.task == "causal_vs_anti"].sort_values("layer")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ca.layer, ca.acc, "o-", color="#2563eb", lw=2, label="causal-vs-anti (direction)")
    ax.fill_between(ca.layer, ca.acc_lo, ca.acc_hi, color="#2563eb", alpha=0.15)
    ax.plot(RV.layer, RV.acc, "o-", color="#dc2626", lw=2, label="fork reversal (word-order)")
    ax.fill_between(RV.layer, RV.acc_lo, RV.acc_hi, color="#dc2626", alpha=0.15)
    ax.axhline(0.5, ls="--", color="grey", alpha=0.6, label="chance"); ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("layer"); ax.set_ylabel("accuracy [95% CI]")
    ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 5: direction vs word-order baseline")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(ctx.tag("stage5_direction_vs_order.png")); plt.close(fig)

    # gap test (peak vs peak, unpaired group-bootstrap)
    ca_pL = int(ca.sort_values("acc").iloc[-1]["layer"])
    rv_pL = int(RV.sort_values("acc").iloc[-1]["layer"])
    ca_b, rv_b = boot_store[("causal_vs_anti", ca_pL)], rev_boot[rv_pL]
    gap, lo, hi, p_le0 = unpaired_gap_distribution(ca_b, rv_b)
    significant = bool(lo > 0)
    print(f"Stage 5: causal-vs-anti peak {ca_b.mean():.3f} (L{ca_pL}) | "
          f"fork-reversal peak {rv_b.mean():.3f} (L{rv_pL})")
    print(f"  GAP = {gap:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  P(gap<=0)={p_le0:.4f}")
    print("  VERDICT:", "SIGNIFICANT - direction exceeds the word-order floor"
          if significant else "NOT significant - requalify")
    result = {"model": ctx.cfg.model_key, "causal_anti_peak": ca_b.mean(),
              "reversal_peak": rv_b.mean(), "gap": gap, "ci_lo": lo, "ci_hi": hi,
              "p_le0": p_le0, "significant": significant}
    pd.DataFrame([result]).to_csv(ctx.tag("gap_test.csv"), index=False)
    return result
