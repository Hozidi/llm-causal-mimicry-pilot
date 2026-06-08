"""
STAGE 6 — MEDIATION ANALYSIS  (the result)
==========================================
WHAT THIS DOES:
  Tests the chain  frequency -> representation quality -> causal mimicry.
  Reports DIRECT, INDIRECT (mediated), and TOTAL effects.

CRITICAL CLARIFICATION (you asked):
  This is a DAG over your THREE ANALYSIS VARIABLES (freq -> rep -> mimicry).
  It is NOT a DAG over each relation's internal causal structure (X -> Y).
  Two different levels:
    - relation-level structure (X->Y) = the GROUND TRUTH you authored (relations.json)
    - analysis-level structure (freq->rep->mimicry) = THIS, the mediation hypothesis
  Don't conflate them.

METHOD (simple, defensible, correct for 3 variables):
  a) rep_quality ~ frequency + covariates         (does freq drive the mediator?)
  b) mimicry ~ rep_quality + frequency + covariates (does the mediator drive Y,
                                                      controlling for freq?)
  indirect (mediated) effect ≈ a[freq] * b[rep_quality]
  direct effect              = b[freq]
  Report effect sizes with HONEST confidence intervals (small n -> wide; say so).
  Use statsmodels, or a mediation package (e.g. pingouin.mediation_analysis).

*** NO PyMC HERE ***
  Three variables, clean regression. PyMC would be decoration, not computation,
  and reintroduces the 'using PyMC to look like a PyMC expert' trap we cut.
  PyMC earns a place ONLY if the PHD later builds a genuinely messy/hierarchical
  model (many latent factors, hierarchy across models/relations). Not the pilot.

ALSO COMPUTE — the accuracy–frequency gap (the parrot-without-structure signal):
  relations where mimicry-accuracy is HIGH but frequency / rep_quality is LOW.
  These are 'passes as causal without the structure' — invisible to accuracy alone,
  and the thing that justifies the whole apparatus.

INPUT:  data/pilot_data.csv
OUTPUT: printed effects + figures/ (handled in plotting)
"""

# import pandas as pd
# import statsmodels.formula.api as smf

def run_mediation(data_path="data/pilot_data.csv"):
    """Estimate direct/indirect/total effects of frequency on mimicry via rep_quality."""
    # TODO:
    # df = pd.read_csv(data_path)
    # a = smf.ols("rep_quality ~ frequency + concept_commonality", df).fit()
    # b = smf.ols("mimicry_score ~ rep_quality + frequency + concept_commonality", df).fit()
    # indirect = a.params['frequency'] * b.params['rep_quality']
    # direct   = b.params['frequency']
    # report with bootstrap CIs
    raise NotImplementedError("standard mediation regression — NOT PyMC")

def accuracy_frequency_gap(data_path="data/pilot_data.csv"):
    """Flag relations: high mimicry-accuracy, low frequency/rep_quality."""
    raise NotImplementedError("identify the mimicry-without-structure cases")


if __name__ == "__main__":
    pass
