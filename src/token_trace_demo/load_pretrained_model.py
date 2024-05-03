import functools
import pickle
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from sae_lens import SparseAutoencoder
from sae_lens.training.utils import BackwardsCompatibleUnpickler
from transformer_lens import HookedTransformer

from token_trace_demo.types import ModuleName, SAEDict
from token_trace_demo.constants import DEFAULT_MODEL_NAME, DEFAULT_SAE_SOURCE, DEVICE


@functools.lru_cache(maxsize=1)
def load_model(model_name: str = DEFAULT_MODEL_NAME) -> HookedTransformer:
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unknown model: {model_name}")
    return HookedTransformer.from_pretrained(model_name, device=DEVICE)


def load_sae(layer: int, model_name: str = DEFAULT_MODEL_NAME) -> SparseAutoencoder:
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unknown model: {model_name}")
    filename = (
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    )
    path = hf_hub_download(repo_id=DEFAULT_SAE_SOURCE, filename=filename)
    # Hacky way to get torch to unpickle an old version of SAELens model
    fake_pickle = SimpleNamespace()
    fake_pickle.Unpickler = BackwardsCompatibleUnpickler
    fake_pickle.__name__ = pickle.__name__
    data = torch.load(path, map_location=DEVICE, pickle_module=fake_pickle)
    cfg = data["cfg"]
    cfg.device = DEVICE
    sparse_autoencoder = SparseAutoencoder(cfg=cfg)
    sparse_autoencoder.load_state_dict(data["state_dict"])
    sparse_autoencoder.train()
    return sparse_autoencoder.to(DEVICE)


@functools.lru_cache(maxsize=1)
def load_sae_dict(
    model_name: str = DEFAULT_MODEL_NAME, sae_source: str = DEFAULT_SAE_SOURCE
) -> SAEDict:
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unknown model: {model_name}")
    if sae_source != DEFAULT_SAE_SOURCE:
        raise ValueError(f"Unknown SAE source: {sae_source}")
    # TODO: un-hardcode n_layers
    n_layers = 12

    sae_dict: SAEDict = {}
    for layer in range(n_layers):
        sae = load_sae(layer, model_name=model_name)
        name = ModuleName(sae.cfg.hook_point)
        sae_dict[name] = sae
    return sae_dict


if __name__ == "__main__":
    # Download the artefacts from huggingface
    load_model()
    load_sae_dict()