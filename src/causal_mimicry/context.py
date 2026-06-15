"""The RunContext — one object that replaces the notebook's pile of globals.

A stage takes a ``RunContext`` and reads what it needs (the config + tag(), the
loaded model/tokenizer, the scorer, the dataset, the probing helpers). Cross-stage
artifacts (e.g. extracted hidden states) are written to disk via ``cfg.tag`` and a
small ``artifacts`` dict, mirroring how the notebook hands state between stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RunConfig
from .data import Dataset
from .hidden import HiddenExtractor, selected_layers_for
from .scoring import Scorer


@dataclass
class RunContext:
    cfg: RunConfig
    model: Any
    tok: Any
    data: Dataset
    scorer: Scorer = field(init=False)
    n_layers: int = field(init=False)
    selected_layers: list[int] = field(init=False)
    extractor: HiddenExtractor = field(init=False)
    artifacts: dict = field(default_factory=dict)   # cross-stage handoff (DataFrames, paths)

    def __post_init__(self) -> None:
        self.scorer = Scorer(self.model, self.tok)
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.selected_layers = selected_layers_for(self.n_layers)
        self.extractor = HiddenExtractor(self.model, self.tok, self.selected_layers)

    # convenience pass-throughs
    def tag(self, name: str) -> str:
        return self.cfg.tag(name)

    @property
    def items(self):
        return self.data.items

    @property
    def rev(self):
        return self.data.rev
