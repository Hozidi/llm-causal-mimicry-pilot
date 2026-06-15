"""causal_mimicry - do base LLMs encode causal direction and dose, and do they use it?

Lightweight, dependency-free entry points (config, stats, data) are imported eagerly.
Heavy submodules (model / scoring / hidden / stages, which pull in torch and
transformers) are imported lazily on first attribute access, so ``import
causal_mimicry`` is cheap and works even where torch is unavailable.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from . import config, data, stats
from .config import MODELS, RUN_ORDER, RunConfig

__version__ = "0.1.0"

_LAZY = {"model", "scoring", "hidden", "plotting", "context", "stages"}


def __getattr__(name: str):
    if name in _LAZY:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # for editors/type-checkers only
    from . import context, hidden, model, plotting, scoring, stages  # noqa: F401

__all__ = ["config", "data", "stats", "MODELS", "RUN_ORDER", "RunConfig", "__version__"]
