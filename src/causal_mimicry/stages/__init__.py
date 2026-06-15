"""The nine-stage experiment ladder, one module per stage.

Each ``run(ctx, ...)`` takes a RunContext, writes its CSV/figure outputs via
``ctx.tag``, stashes any cross-stage handoff in ``ctx.artifacts``, and returns its
main result table. Run them in order with ``scripts/run_model.py`` or a notebook.
"""
from . import (stage1_direction, stage2_dose, stage3_controls, stage4_probes,
               stage5_wordorder, stage6_alignment, stage8_geometry)

__all__ = ["stage1_direction", "stage2_dose", "stage3_controls", "stage4_probes",
           "stage5_wordorder", "stage6_alignment", "stage8_geometry"]
