from token_trace_demo.builder import SparseFeatureCircuitBuilder
from token_trace_demo.app.load_data import DATA_DIR

PROMPTS = [
    # IOI
    "When John and Mary went to the shops, John gave the bag to Mary",
    "When Tim and Jane went to the shops, Tim gave the bag to Jane",
    # Factual recall
    "Fact: Tokyo is a city in the country of Japan",
    "Fact: Delhi is a city in the country of India",
]

if __name__ == "__main__":
    for prompt in PROMPTS:
        print(f"Computing circuit for prompt: {prompt}")
        builder = SparseFeatureCircuitBuilder(text=prompt)
        builder.compute_circuit()
        circuit = builder.circuit
        metadata = builder.metadata
        circuit.save(save_dir=DATA_DIR)
        metadata.save(save_dir=DATA_DIR)
