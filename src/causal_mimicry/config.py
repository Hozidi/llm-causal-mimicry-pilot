"""Static configuration: the model registry, run-level config, and shared constants.

This module is intentionally free of heavy dependencies (no torch / transformers),
so it can be imported anywhere — including by tooling that just needs the model list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

# --------------------------------------------------------------------------- #
# Model registry: key -> (HuggingFace id, parameter count in billions)
# Size-ordered so a sweep runs small -> large.
# --------------------------------------------------------------------------- #
MODELS: dict[str, tuple[str, float]] = {
    "pythia-410m":            ("EleutherAI/pythia-410m",                 0.41),
    "olmo-1b":                ("allenai/OLMo-1B-0724-hf",                1.18),
    "pythia-6.9b":            ("EleutherAI/pythia-6.9b",                 6.9),
    "olmo-7b":                ("allenai/OLMo-7B-0724-hf",                6.9),
    "qwen-7b-base":           ("Qwen/Qwen2.5-7B",                        7.6),
    "gemma-2-9b-base":        ("google/gemma-2-9b",                      9.2),    # gated
    "pythia-12b":             ("EleutherAI/pythia-12b",                  11.8),
    "mistral-nemo-12b-base":  ("mistralai/Mistral-Nemo-Base-2407",       12.2),
    "olmo-2-13b":             ("allenai/OLMo-2-1124-13B",                13.7),
    "qwen-14b-base":          ("Qwen/Qwen2.5-14B",                       14.8),
    "mistral-small-24b-base": ("mistralai/Mistral-Small-24B-Base-2501",  24.0),
    "gemma-2-27b-base":       ("google/gemma-2-27b",                     27.2),   # gated
    "qwen-32b-base":          ("Qwen/Qwen2.5-32B",                       32.8),
}
RUN_ORDER: list[str] = list(MODELS.keys())

# --------------------------------------------------------------------------- #
# Dose levels and category -> correct-continuation key
# --------------------------------------------------------------------------- #
LEVELS = ["base", "low", "moderate", "high"]      # Dose-A (graded adverbial)
ORD_LEVELS = ["less", "base", "more"]             # Dose-B (ordinal, frequency-matched)

# score_dirs() returns probabilities under keys forward / reverse / none.
CAT2KEY = {
    "causal": "forward",
    "anti-causal": "reverse",
    "anti_causal": "reverse",
    "spurious": "none",
    "fork_spurious": "none",
    "fork-spurious": "none",
}

# --------------------------------------------------------------------------- #
# Prompt templates (single source of truth for the continuation scorer)
# --------------------------------------------------------------------------- #
NEUTRAL_PROMPT = "Consider the variables: {x} and {y}. The relation between them is that "
ASSOC_PROMPT = "The following are often mentioned together: "   # non-causal conjunction (Stage 3B)

# --------------------------------------------------------------------------- #
# House-style colour palette
# --------------------------------------------------------------------------- #
PALETTE = {
    "causal": "#2563eb",
    "anti-causal": "#f59e0b",
    "spurious": "#16a34a",
    "fork_spurious": "#16a34a",
}
GAP_COLOR = "#7c3aed"


@dataclass
class RunConfig:
    """Everything that varies between one model run and the next.

    Replaces the notebook globals MODEL_KEY / MODEL_ID / OUTDIR / tag / USE_NF4 / ...
    Build one of these per model, then hand it to ``load_model`` and the stages.
    """
    model_key: str
    out_root: str = "./results"
    data_version: str = "v4"
    run_date: str = field(default_factory=lambda: date.today().isoformat())
    hf_token: str | None = None

    # filled in __post_init__ from MODELS
    model_id: str = field(init=False)
    size_b: float = field(init=False)
    is_gemma: bool = field(init=False)
    is_gated: bool = field(init=False)
    use_nf4: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.model_key not in MODELS:
            raise KeyError(
                f"unknown model_key {self.model_key!r}; choose one of {RUN_ORDER}"
            )
        self.model_id, self.size_b = MODELS[self.model_key]
        self.is_gemma = "gemma" in self.model_key.lower()
        self.is_gated = self.is_gemma                       # extend if more gated models added
        self.use_nf4 = (self.size_b >= 24) or self.is_gemma  # 4-bit for very large / gemma

    @property
    def outdir(self) -> str:
        d = os.path.join(self.out_root, self.model_key)
        os.makedirs(d, exist_ok=True)
        return d

    def tag(self, name: str) -> str:
        """Date-stamped, per-model output path: <outdir>/<key>_<ver>_<date>_<name>."""
        return os.path.join(
            self.outdir, f"{self.model_key}_{self.data_version}_{self.run_date}_{name}"
        )
