# Causal Mimicry in LLMs — Pilot

## The question
Does **representation quality** *mediate* the effect of **pretraining co-occurrence
frequency** on **causal-mimicry quality**?

    frequency  ──►  representation quality  ──►  causal mimicry
       (1)                  (2)                       (3)
    [counted]          [LRE metric]              [MY definition]

## The three variables — kept STRICTLY separate
1. **Co-occurrence frequency** (independent variable).
   How often subject & object co-occur in the pretraining corpus.
   COUNTED via BatchSearch/WIMBD on an open-DATA model (OLMo/Dolma).
   NOT predicted, NOT estimated — counted, because the data is open.

2. **Representation quality** (candidate mediator).
   The LRE causality/faithfulness score (Merullo/Hernandez method).
   Measures how cleanly/linearly the relation is encoded.
   THIS IS NOT THE OUTCOME. It is factual-association linearity,
   NOT causal-direction reasoning.

3. **Causal-mimicry quality** (the outcome, Y) — MINE TO DEFINE.
   Whether the model gets causal DIRECTION/STRUCTURE right.
   There is NO off-the-shelf paper that scores this — defining it is
   the contribution. Must be scored INDEPENDENTLY of variable (2),
   or the mediation is circular and meaningless.

## Model
OLMo-7B (Dolma corpus) — open weights AND open data, so frequency is countable.

## Pipeline (each folder = one stage)
- relations/      : the ground-truth item set (spec sheet for everything downstream)
- frequency/      : count co-occurrence -> variable (1)
- representation/ : fit LRE, get quality score -> variable (2)
- elicitation/    : generate prompts from ground truth (Wan-style), query model,
                    collect RAW responses  [INPUT to scoring, not the final table]
- mimicry/        : turn raw responses into the mimicry score -> variable (3)
- data/           : assemble one row per relation with all 3 vars -> pilot_data.csv
- analysis/       : mediation test (freq -> rep -> mimicry) + the accuracy-freq gap
- figures/        : the output plots

## Status
Scaffolding. Next substantive step: write the Y-definition (mimicry/) in plain
English BEFORE any code.

## Limitations (to keep honest)
- Small n, single model, preliminary — "suggestive evidence," nothing stronger.
- Y-definition is a design choice; state it explicitly.

## NOT using PyMC in the pilot
The mediation is 3 variables — standard mediation regression (statsmodels) is
correct. PyMC would be decoration here. It earns a place ONLY if the PhD later
builds a genuinely messy/hierarchical model. Not now.
