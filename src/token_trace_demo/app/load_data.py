import json
import pathlib

from hashlib import md5
from token_trace_demo.circuit import SparseFeatureCircuit

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "app" / "data"


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


def load_circuit(
    text: str, data_dir: pathlib.Path = DATA_DIR
) -> SparseFeatureCircuit:
    """Load or compute the circuit data."""
    prefix = md5(text.encode()).hexdigest()[:16]
    circuit = SparseFeatureCircuit.load(data_dir / prefix)
    return circuit

def load_metadata(
    text: str, data_dir: pathlib.Path = DATA_DIR
) -> dict:
    """Load the metadata for the circuit."""
    prefix = md5(text.encode()).hexdigest()[:16]
    with open(data_dir / prefix / "metadata.json") as f:
        metadata = json.load(f)
    return metadata