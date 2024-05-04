import json
import pathlib
import pandas as pd

from transformer_lens import HookedTransformer
from token_trace_demo.types import MetricFunction, SAEDict
from token_trace_demo.load_pretrained_model import load_model, load_sae_dict
from token_trace_demo.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXT,
    DEFAULT_SAE_SOURCE,
)
from token_trace_demo.circuit import (
    SparseFeatureCircuit,
    SparseFeatureCircuitMetadata,
)
from token_trace_demo.utils import last_token_prediction_loss
from token_trace_demo.sae_activation_cache import (
    get_sae_activation_cache,
    SAEActivationCache,
)
from token_trace_demo.node_attribution import compute_node_attribution, filter_nodes


class SparseFeatureCircuitBuilder:
    model: HookedTransformer
    sae_dict: SAEDict
    metric_fn: MetricFunction
    # TODO: support multiple text strings.
    text: str
    sae_activation_cache: SAEActivationCache

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        sae_source: str = DEFAULT_SAE_SOURCE,
        text: str = DEFAULT_TEXT,
        min_node_abs_ie: float = 0.0,
        max_n_nodes: int = -1,
    ):
        self.model_name = model_name
        self.sae_source = sae_source
        self.text = text
        self.min_node_abs_ie = min_node_abs_ie
        self.max_n_nodes = max_n_nodes
        self.node_ie_df = pd.DataFrame()

    @property
    def circuit(self):
        """Get the circuit at the current stage"""
        return SparseFeatureCircuit(
            node_ie_df=self.node_ie_df,
        )

    @property
    def metadata(self):
        token_strs = self.model.to_str_tokens(self.text)
        return SparseFeatureCircuitMetadata(
            model_name=self.model_name,
            sae_source=self.sae_source,
            text=self.text,
            token_strs=token_strs,
            # TODO: prompt info
            prompt_info="TODO",
        )

    def compute_sae_activation_cache(self) -> "SparseFeatureCircuitBuilder":
        self.model = load_model(self.model_name)
        self.sae_dict = load_sae_dict(self.model_name)
        self.metric_fn = last_token_prediction_loss

        self.sae_activation_cache = get_sae_activation_cache(
            self.model, self.sae_dict, self.metric_fn, self.text
        )
        return self

    def compute_node_attributions(self) -> "SparseFeatureCircuitBuilder":
        self.node_ie_df = compute_node_attribution(
            model=self.model,
            sae_activation_cache=self.sae_activation_cache,
            text=self.text,
        )
        return self

    def get_filtered_nodes(
        self,
        min_node_abs_ie: float | None = None,
        max_n_nodes: int | None = None,
    ) -> pd.DataFrame:
        """Get filtered nodes by total absolute indirect effect"""
        if min_node_abs_ie is not None:
            self.min_node_abs_ie = min_node_abs_ie
        if max_n_nodes is not None:
            self.max_n_nodes = max_n_nodes
        assert self.node_ie_df is not None  # keep pyright happy
        return filter_nodes(
            self.node_ie_df,
            min_node_abs_ie=self.min_node_abs_ie,
            max_n_nodes=self.max_n_nodes,
        )

    def filter_nodes(
        self,
        min_node_abs_ie: float | None = None,
        max_n_nodes: int | None = None,
    ) -> "SparseFeatureCircuitBuilder":
        """Filter nodes by total absolute indirect effect"""
        self.node_ie_df = self.get_filtered_nodes(
            min_node_abs_ie=min_node_abs_ie,
            max_n_nodes=max_n_nodes,
        )
        return self

    def compute_circuit(self) -> "SparseFeatureCircuitBuilder":
        return (
            self.compute_sae_activation_cache()
            .compute_node_attributions()
            .filter_nodes()
            # TODO: edges
        )

    def save_args(self, save_dir: pathlib.Path):
        args = {
            "model_name": self.model_name,
            "text": self.text,
            "min_node_abs_ie": self.min_node_abs_ie,
            "max_n_nodes": self.max_n_nodes,
        }
        with open(save_dir / "args.json", "w") as f:
            json.dump(args, f)
