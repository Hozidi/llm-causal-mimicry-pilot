"""Stage 4 - Hidden-state probes.

Is causal direction (and relation type) linearly decodable from the model's hidden
states, above a TF-IDF lexical baseline and a random-label control? Extracts last-token
hidden states for the main set and the word-order (reversal) set, then probes each
task per layer with GroupKFold over relations.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..hidden import oof_predictions, tfidf_acc
from ..stats import group_bootstrap_accuracy

_TASKS = {
    "causal_vs_anti": ["causal", "anti-causal"],
    "causal_vs_spurious": ["causal", "spurious"],
    "anti_vs_spurious": ["anti-causal", "spurious"],
    "direct_vs_spurious": "direct",
    "3way": ["causal", "anti-causal", "spurious"],
}


def _task_select(task, cats):
    spec = _TASKS[task]
    if spec == "direct":
        mask = np.isin(cats, ["causal", "anti-causal", "spurious"])
        return mask, np.where(cats == "spurious", "spurious", "direct")
    return np.isin(cats, spec), cats


def run(ctx) -> pd.DataFrame:
    layers = ctx.selected_layers

    # extract hidden states (kept in memory + saved) for main + reversal sets
    Xmain, mmain = ctx.extractor.extract(ctx.items, ["id", "pair_id", "category", "true_direction"])
    mmain["text"] = mmain.x_text.astype(str) + " " + mmain.y_text.astype(str)
    np.savez_compressed(ctx.tag("hidden_main.npz"), X=Xmain, layers=np.array(layers))
    mmain.to_csv(ctx.tag("meta_main.csv"), index=False)

    Xrev, mrev = ctx.extractor.extract(ctx.rev, ["id", "pair_id", "category"])
    mrev["text"] = mrev.x_text.astype(str) + " " + mrev.y_text.astype(str)
    np.savez_compressed(ctx.tag("hidden_rev.npz"), X=Xrev, layers=np.array(layers))
    mrev.to_csv(ctx.tag("meta_rev.csv"), index=False)

    ctx.artifacts.update(Xmain=Xmain, mmain=mmain, Xrev=Xrev, mrev=mrev, layers=layers)
    print(f"Stage 4: hidden main {Xmain.shape} | reversal {Xrev.shape}")

    cats = mmain["category"].values
    ids = mmain["pair_id"].values
    texts = mmain["text"].values

    probe_rows, boot_store = [], {}
    for task in _TASKS:
        mask, lab = _task_select(task, cats)
        g, y, tx = ids[mask], lab[mask], texts[mask]
        chance = 1 / len(np.unique(y))
        tb = tfidf_acc(tx, y, g)
        for li, L in enumerate(layers):
            X = Xmain[mask, li, :].astype(np.float32)
            corr = oof_predictions(X, y, g)
            rcorr = oof_predictions(X, y, g, shuffle=True)
            acc, lo, hi, boots = group_bootstrap_accuracy(corr, g)
            racc, *_ = group_bootstrap_accuracy(rcorr, g)
            boot_store[(task, int(L))] = boots
            probe_rows.append({
                "model": ctx.cfg.model_key, "task": task, "layer": int(L),
                "n": int(mask.sum()), "chance": round(chance, 3),
                "acc": round(acc, 3), "acc_lo": round(lo, 3), "acc_hi": round(hi, 3),
                "tfidf": round(tb, 3), "random": round(racc, 3),
            })
        print(f"  {task:20s} done")

    PM = pd.DataFrame(probe_rows)
    PM.to_csv(ctx.tag("probes.csv"), index=False)
    ctx.artifacts["PM"] = PM
    ctx.artifacts["boot_store"] = boot_store

    fig, axes = plt.subplots(1, len(_TASKS), figsize=(4 * len(_TASKS), 4.2), sharey=True)
    for ax, task in zip(axes, _TASKS):
        t = PM[PM.task == task].sort_values("layer")
        ax.plot(t.layer, t.acc, "o-", color="#2563eb", lw=2, label="hidden probe")
        ax.fill_between(t.layer, t.acc_lo, t.acc_hi, color="#2563eb", alpha=0.15)
        ax.plot(t.layer, t.tfidf, "s--", color="#f59e0b", label="TF-IDF")
        ax.plot(t.layer, t.random, "^:", color="#16a34a", label="random")
        ax.axhline(t.chance.iloc[0], ls="--", color="grey", alpha=0.5)
        ax.set_title(task, fontsize=10); ax.set_xlabel("layer")
    axes[0].set_ylabel("accuracy [95% CI over relations]"); axes[0].legend(fontsize=8)
    plt.suptitle(f"{ctx.cfg.model_key} \u00b7 Stage 4: hidden-state separability", y=1.03)
    plt.tight_layout(); plt.savefig(ctx.tag("stage4_probes.png")); plt.close(fig)
    return PM
