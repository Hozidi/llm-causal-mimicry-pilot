"""Stage 6 - Probe-output alignment (the dissociation).

Does the internally decodable direction signal actually reach the output? We take the
out-of-fold hidden margin toward the correct direction and correlate it with the
prompt-conditioned output margin. Strong hidden + near-zero output + low correlation =
"encoded internally, not read out".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..stats import pearson_boot


def _output_margin(rr):
    f = np.log(max(rr["p_forward"], 1e-9))
    r = np.log(max(rr["p_reverse"], 1e-9))
    if rr["category"] == "causal":
        return f - r
    if rr["category"] == "anti-causal":
        return r - f
    return np.nan


def run(ctx) -> dict:
    """Needs Stage 1 (base_df) and Stage 4 (PM, Xmain, mmain, layers) in ctx.artifacts."""
    base_df = ctx.artifacts["base_df"]
    PM = ctx.artifacts["PM"]
    Xall, M, layers = ctx.artifacts["Xmain"], ctx.artifacts["mmain"], ctx.artifacts["layers"]

    om = {r["id"]: _output_margin(r) for _, r in base_df.iterrows()}

    cm = np.isin(M["category"].values, ["causal", "anti-causal"])
    Mca = M[cm].reset_index(drop=True)
    best_L = int(PM[PM.task == "causal_vs_anti"].sort_values("acc").iloc[-1]["layer"])
    li = list(layers).index(best_L)
    Xca = Xall[cm, li, :].astype(np.float32)
    yca, gca = Mca["category"].values, Mca["pair_id"].values

    k = min(5, len(np.unique(gca)))
    gkf = GroupKFold(k)
    hidden = np.full(len(yca), np.nan)
    for tr, te in gkf.split(Xca, yca, groups=gca):
        nc = max(2, min(128, len(tr) - 1, Xca.shape[1]))
        clf = make_pipeline(StandardScaler(), PCA(n_components=nc, random_state=0),
                            LogisticRegression(max_iter=1000))
        clf.fit(Xca[tr], yca[tr])
        df_ = clf.decision_function(Xca[te])
        sign = 1.0 if clf.classes_[1] == "causal" else -1.0
        hidden[te] = df_ * sign

    Mca["hidden_margin_correct"] = np.where(Mca["category"].values == "causal", hidden, -hidden)
    Mca["output_margin"] = Mca["id"].map(om).values
    al = Mca.dropna(subset=["hidden_margin_correct", "output_margin"])

    r, lo, hi = pearson_boot(al.hidden_margin_correct.values, al.output_margin.values)
    rho, _ = stats.spearmanr(al.hidden_margin_correct, al.output_margin)
    print(f"Stage 6: Pearson r={r:.3f} [{lo:.3f},{hi:.3f}] | Spearman rho={rho:.3f} | "
          f"n={len(al)} | best layer {best_L}")
    Mca.to_csv(ctx.tag("alignment.csv"), index=False)
    result = {"model": ctx.cfg.model_key, "pearson_r": r, "r_lo": lo, "r_hi": hi,
              "spearman_rho": rho, "n": len(al), "best_layer": best_L}
    pd.DataFrame([result]).to_csv(ctx.tag("alignment_stats.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.3, 6))
    sns.scatterplot(data=al, x="hidden_margin_correct", y="output_margin", hue="category",
                    palette={"causal": "#2563eb", "anti-causal": "#f59e0b"}, s=55, alpha=0.8, ax=ax)
    sns.regplot(data=al, x="hidden_margin_correct", y="output_margin", scatter=False,
                color="#475569", ci=95, ax=ax)
    ax.axhline(0, ls="--", color="grey", alpha=0.5); ax.axvline(0, ls="--", color="grey", alpha=0.5)
    ax.set_xlabel("hidden margin (correct direction, out-of-fold)")
    ax.set_ylabel("output margin (logP correct - wrong)")
    ax.set_title(f"{ctx.cfg.model_key} \u00b7 Stage 6: probe-output alignment\n"
                 f"r={r:.2f} [{lo:.2f},{hi:.2f}], rho={rho:.2f}")
    plt.tight_layout(); plt.savefig(ctx.tag("stage6_alignment.png")); plt.close(fig)
    print("  strong hidden + output~0 + low r => direction encoded internally, not read out.")
    return result
