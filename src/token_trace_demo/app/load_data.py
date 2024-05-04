import json
import pathlib

from hashlib import md5
from token_trace_demo.constants import DATA_DIR
from token_trace_demo.circuit import SparseFeatureCircuit

def hash_text(text: str) -> str:
    return md5(text.encode()).hexdigest()[:16]

def list_existing_circuits() -> list[str]:
    existing_texts = []
    savedirs = [path for path in DATA_DIR.iterdir() if path.is_dir()]
    # get the text for each circuit
    for savedir in savedirs:
        try:
            with open(savedir / "args.json") as f:
                args = json.load(f)
                existing_texts.append(args["text"])
        except FileNotFoundError:
            continue
    return existing_texts


def load_circuit(text: str, data_dir: pathlib.Path = DATA_DIR) -> SparseFeatureCircuit:
    """Load or compute the circuit data."""
    circuit = SparseFeatureCircuit.load(data_dir / hash_text(text))
    return circuit


def load_metadata(text: str, data_dir: pathlib.Path = DATA_DIR) -> dict:
    """Load the metadata for the circuit."""
    with open(data_dir / hash_text(text) / "metadata.json") as f:
        metadata = json.load(f)
    return metadata
