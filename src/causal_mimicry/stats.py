"""Statistics used everywhere: bootstrap CIs that resample **relations**, not rows.

The unit of independence is the relation (``pair_id``), not the individual scored
item — the four graded versions of "smoking -> cancer" are not four independent
observations. Resampling whole relation groups keeps the confidence intervals honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# A single shared generator so results are reproducible across the pipeline.
RNG = np.random.default_rng(0)


def bootstrap_ci_mean(values, n_boot: int = 10000, ci: int = 95):
    """Plain bootstrap CI for a mean (no grouping). Returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), np.nan, np.nan
    b = np.array([RNG.choice(v, len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(b, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return v.mean(), lo, hi


def grouped_bootstrap_ci(df: pd.DataFrame, value_col: str, group_col: str,
                         n_boot: int = 5000, ci: int = 95):
    """Mean CI by resampling whole groups (relations) with replacement.

    Returns (mean, lo, hi).
    """
    groups = df[group_col].unique()
    gmap = {g: df[df[group_col] == g][value_col].values for g in groups}
    b = np.array([
        np.concatenate([gmap[g] for g in RNG.choice(groups, len(groups), replace=True)]).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(b, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return df[value_col].mean(), lo, hi


def group_bootstrap_accuracy(item_correct, item_groups, n_boot: int = 5000, ci: int = 95):
    """Accuracy CI by resampling relation groups. Returns (acc, lo, hi, boots).

    ``boots`` is the full bootstrap distribution, needed for the unpaired gap test.
    """
    item_correct = np.asarray(item_correct, float)
    item_groups = np.asarray(item_groups)
    groups = np.unique(item_groups)
    gmap = {g: item_correct[item_groups == g] for g in groups}
    boots = np.array([
        np.concatenate([gmap[g] for g in RNG.choice(groups, len(groups), replace=True)]).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return item_correct.mean(), lo, hi, boots


def unpaired_gap_distribution(boots_a, boots_b, ci: int = 95):
    """Honest gap between two INDEPENDENT group-bootstrap distributions.

    The two tasks (direction vs word-order) run on different items, so the
    distributions cannot be paired item-for-item. Returns (gap, lo, hi, P(gap<=0)).
    """
    n = min(len(boots_a), len(boots_b))
    diff = boots_a[:n] - RNG.permutation(boots_b[:n])
    lo, hi = np.percentile(diff, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return diff.mean(), lo, hi, (diff <= 0).mean()


def pearson_boot(x, y, n_boot: int = 5000):
    """Pearson r with a bootstrap 95% CI. Returns (r, lo, hi)."""
    x, y = np.asarray(x), np.asarray(y)
    r0, _ = stats.pearsonr(x, y)
    idx = np.arange(len(x))
    samples = [RNG.choice(idx, len(idx), replace=True) for _ in range(n_boot)]
    b = np.array([stats.pearsonr(x[s], y[s])[0] for s in samples])
    return (r0, *np.percentile(b, [2.5, 97.5]))
