import json
import pathlib
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class SparseFeatureCircuitMetadata:
    """Metadata for a SparseFeatureCircuit"""

    model_name: str  # The base model name
    sae_source: str  # The source of the SAEs used
    text: str
    token_strs: list[str]
    prompt_info: str

    def save(self, save_dir: pathlib.Path):
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(save_dir: pathlib.Path) -> "SparseFeatureCircuitMetadata":
        with open(save_dir / "metadata.json") as f:
            metadata = json.load(f)
        return SparseFeatureCircuitMetadata(**metadata)


@dataclass
class SparseFeatureCircuit:
    """A circuit consisting of SAE features"""

    # Represent the sub-graph
    node_ie_df: pd.DataFrame
    metadata: SparseFeatureCircuitMetadata | None = None

    """ Utility functions """

    def copy(self):
        return SparseFeatureCircuit(
            node_ie_df=self.node_ie_df.copy(),  # type: ignore
            metadata=self.metadata,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_ie_df)

    def get_nodes_in_module(self, module_name: str) -> pd.DataFrame:
        return self.node_ie_df[self.node_ie_df["module"] == module_name]

    """ Save and load """

    def save(self, save_dir: pathlib.Path):
        # Save results
        if hasattr(self, "node_ie_df"):
            self.node_ie_df.to_csv(save_dir / "node.csv")

    @staticmethod
    def load(save_dir: pathlib.Path) -> "SparseFeatureCircuit":
        if (save_dir / "node.csv").exists():
            node_ie_df = pd.read_csv(save_dir / "node.csv", index_col=0)
        else:
            node_ie_df = pd.DataFrame()

        if (save_dir / "metadata.json").exists():
            metadata = SparseFeatureCircuitMetadata.load(save_dir)
        else:
            metadata = None

        return SparseFeatureCircuit(
            node_ie_df=node_ie_df,
            metadata=metadata,
        )
