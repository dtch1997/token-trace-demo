import pathlib

DEFAULT_MODEL_NAME = "gpt2-small"
DEFAULT_SAE_SOURCE = "jbloom/GPT2-Small-SAEs"
DEFAULT_TEXT = "When John and Mary went to the shops, John gave the bag to Mary"
DEVICE = "cpu"

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
