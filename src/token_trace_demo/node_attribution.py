import logging
from typing import cast

import pandas as pd
from transformer_lens import HookedTransformer

from token_trace_demo.load_pretrained_model import load_model
from token_trace_demo.sae_activation_cache import SAEActivationCache
from token_trace_demo.utils import get_layer_from_module_name, setup_logger

logger = setup_logger(__name__, logging.INFO)


def get_nodes_in_module(node_ie_df: pd.DataFrame, *, module_name: str) -> pd.DataFrame:
    return node_ie_df[node_ie_df.module_name == module_name]


def filter_nodes(
    node_ie_df: pd.DataFrame,
    *,
    min_node_abs_ie: float = 0.0,
    max_n_nodes: int = -1,
) -> pd.DataFrame:
    node_index_cols = ["module_name", "act_idx", "token_idx"]

    # Mean across examples
    node_ie_df["node_abs_ie"] = node_ie_df.groupby(node_index_cols)["abs_ie"].transform(
        "mean"
    )

    # Filter out nodes with low indirect effect
    df = node_ie_df[node_ie_df["node_abs_ie"] > min_node_abs_ie]

    if max_n_nodes == -1:
        # Return all nodes
        return df
    else:
        # Select top nodes
        df = df[node_index_cols + ["node_abs_ie"]].drop_duplicates()
        df = df.sort_values("node_abs_ie", ascending=False)
        df = df.head(max_n_nodes)
        df = node_ie_df.merge(df, on=node_index_cols + ["node_abs_ie"], how="inner")
        logger.debug(f"{len(df)} nodes selected.")
        return df


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


def compute_node_attribution(
    model: HookedTransformer, sae_activation_cache: SAEActivationCache, text: str
) -> pd.DataFrame:
    # Get the token strings.
    text_tokens = model.to_str_tokens(text)

    # Construct dataframe.
    rows = []
    for module_name, module_activations in sae_activation_cache.items():
        logger.info(f"Processing module {module_name}")
        layer = get_layer_from_module_name(module_name)
        n_features = module_activations.n_features
        acts = module_activations.activations.coalesce()
        grads = module_activations.gradients.coalesce()
        effects = acts * grads
        effects = effects.coalesce()

        for index, ie_atp, act, grad in zip(
            effects.indices().t(), effects.values(), acts.values(), grads.values()
        ):
            example_idx, token_idx, act_idx = index
            assert example_idx == 0

            rows.append(
                {
                    "layer": layer,
                    "module_name": module_name,
                    "module_type": "resid",
                    "example_idx": example_idx.item(),
                    "example_str": text,
                    "act_idx": act_idx.item(),
                    "act_type": "feature" if act_idx < n_features else "error",
                    "token_idx": token_idx.item(),
                    "token_str": text_tokens[token_idx],
                    "value": act.item(),
                    "grad": grad.item(),
                    "ie": ie_atp.item(),
                    "abs_ie": abs(ie_atp.item()),
                }
            )

    df = pd.DataFrame(rows)
    # Filter out zero indirect effects
    df = df[df["ie"] != 0]
    print(f"{len(df)} non-zero indirect effects found.")
    return df
