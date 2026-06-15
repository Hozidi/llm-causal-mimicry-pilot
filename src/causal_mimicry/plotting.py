"""House plotting style and the date-stamp-on-save hook.

``apply_house_style`` sets the shared rcParams and (once) wraps ``Figure.savefig`` so
every saved figure is stamped with the model key and run date in the corner. The wrap
is guarded so calling it twice cannot double-wrap (which previously caused infinite
recursion on a re-run).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

from .config import PALETTE, GAP_COLOR  # noqa: F401  (re-exported)

_RCPARAMS = {
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "font.size": 11, "legend.frameon": True,
}


def apply_house_style(model_key: str = "?", run_date: str = "") -> None:
    """Set rcParams and install the date-stamp savefig wrapper (idempotent)."""
    plt.rcParams.update(_RCPARAMS)

    if getattr(matplotlib.figure.Figure.savefig, "_is_stamped", False):
        return  # already wrapped — do not wrap again

    _orig_savefig = matplotlib.figure.Figure.savefig

    def _savefig_stamped(self, fname, *args, **kwargs):
        try:
            self.text(0.995, 0.005, f"{model_key} \u00b7 {run_date}",
                      ha="right", va="bottom", fontsize=7, color="#999")
        except Exception:
            pass
        return _orig_savefig(self, fname, *args, **kwargs)

    _savefig_stamped._is_stamped = True
    matplotlib.figure.Figure.savefig = _savefig_stamped
