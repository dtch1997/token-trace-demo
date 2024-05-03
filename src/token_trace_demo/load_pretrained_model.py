import functools
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from transformer_lens import HookedTransformer

from token_trace_demo.types import SAEDict
from token_trace_demo.constants import DEFAULT_MODEL_NAME, DEFAULT_SAE_SOURCE, DEVICE


@functools.lru_cache(maxsize=1)
def load_model(model_name: str = DEFAULT_MODEL_NAME) -> HookedTransformer:
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unknown model: {model_name}")
    return HookedTransformer.from_pretrained(model_name, device=DEVICE)


@functools.lru_cache(maxsize=1)
def load_sae_dict(
    model_name: str = DEFAULT_MODEL_NAME, sae_source: str = DEFAULT_SAE_SOURCE
) -> SAEDict:
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unknown model: {model_name}")
    if sae_source != DEFAULT_SAE_SOURCE:
        raise ValueError(f"Unknown SAE source: {sae_source}")
    saes, _ = get_gpt2_res_jb_saes()
    return saes


if __name__ == "__main__":
    # Download the artefacts from huggingface
    load_model()
    load_sae_dict()
