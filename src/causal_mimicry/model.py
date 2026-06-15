"""Model + tokenizer loading, with the per-model quirks handled in one place:

* very large models (>=24B) and Gemma load in 4-bit NF4,
* Gemma-2 needs eager attention for correct logits,
* gated models need a HuggingFace token.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RunConfig


def load_model(cfg: RunConfig):
    """Load (model, tokenizer) for ``cfg``. The model is returned in eval mode."""
    token_kwargs = {}
    if cfg.is_gated and cfg.hf_token:
        token_kwargs["token"] = cfg.hf_token

    tok = AutoTokenizer.from_pretrained(cfg.model_id, **token_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(device_map="auto", output_hidden_states=True, **token_kwargs)

    if cfg.use_nf4:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    if cfg.is_gemma:
        load_kwargs["attn_implementation"] = "eager"  # correct gemma-2 logits

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs).eval()
    return model, tok
