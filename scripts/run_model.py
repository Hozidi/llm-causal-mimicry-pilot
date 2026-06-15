#!/usr/bin/env python
"""Run the causal-mimicry stage ladder end-to-end for a single model.

Examples
--------
    python scripts/run_model.py --model olmo-2-13b \
        --causal data/causal_400.json --spurious data/spurious_forks_300.json \
        --out results

One model per process; for a sweep, call once per key in causal_mimicry.RUN_ORDER
(restart between models so GPU memory is clean). Large/gated models need an HF token
(``--hf-token`` or the HF_TOKEN environment variable).
"""
from __future__ import annotations

import argparse
import os

from causal_mimicry import RUN_ORDER, RunConfig
from causal_mimicry.context import RunContext
from causal_mimicry.data import load_dataset
from causal_mimicry.model import load_model
from causal_mimicry.plotting import apply_house_style
from causal_mimicry.stages import (stage1_direction, stage2_dose, stage3_controls,
                                    stage4_probes, stage5_wordorder, stage6_alignment,
                                    stage8_geometry)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=RUN_ORDER, help="model key to run")
    p.add_argument("--causal", default="data/causal_400.json", help="path to causal_400.json")
    p.add_argument("--spurious", default="data/spurious_forks_300.json",
                   help="path to spurious_forks_300.json")
    p.add_argument("--out", default="results", help="output root (per-model subfolder created)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for gated models (or set HF_TOKEN)")
    p.add_argument("--stages", default="1,2,3,4,5,6,8",
                   help="comma-separated stages to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    want = set(args.stages.replace(" ", "").split(","))

    cfg = RunConfig(model_key=args.model, out_root=args.out, hf_token=args.hf_token)
    apply_house_style(cfg.model_key, cfg.run_date)
    print(f"{cfg.model_key} ({cfg.size_b}B) -> {cfg.model_id} | NF4={cfg.use_nf4} "
          f"gated={cfg.is_gated} | outputs -> {cfg.outdir}/")

    dataset = load_dataset(args.causal, args.spurious)
    print("data:", dataset.summary())

    model, tok = load_model(cfg)
    ctx = RunContext(cfg=cfg, model=model, tok=tok, data=dataset)
    print(f"loaded {cfg.model_key}: {ctx.n_layers} layers | probe layers {ctx.selected_layers}")

    # The ladder, in dependency order.
    if "1" in want:
        stage1_direction.run(ctx)
    if "2" in want:
        stage2_dose.run(ctx)                 # -> gdf, piv (stashed in ctx.artifacts)
    if "3" in want:
        stage3_controls.run(ctx)             # uses gdf, piv
    if "4" in want:
        stage4_probes.run(ctx)               # -> PM, boot_store, hidden states
    if "5" in want:
        stage5_wordorder.run(ctx)            # uses PM, boot_store, reversal hidden
    if "6" in want:
        stage6_alignment.run(ctx)            # uses base_df, PM, main hidden
    if "8" in want:
        stage8_geometry.run(ctx)

    print(f"\n{cfg.model_key} complete. Outputs in {cfg.outdir}/ "
          f"(date-stamped {cfg.run_date}).")


if __name__ == "__main__":
    main()
