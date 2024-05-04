from token_trace_demo.builder import SparseFeatureCircuitBuilder
from token_trace_demo.app.load_data import DATA_DIR, hash_text

PROMPTS = [
    # IOI
    "When John and Mary went to the shops, John gave the bag to Mary",
    "When Tim and Jane went to the shops, Tim gave the bag to Jane",
    # Factual recall
    "Fact: Tokyo is a city in the country of Japan",
    "Fact: Delhi is a city in the country of India",
]

if __name__ == "__main__":
    print(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for prompt in PROMPTS:
        print(f"Computing circuit for prompt: {prompt}")
        builder = SparseFeatureCircuitBuilder(text=prompt)
        builder.compute_circuit()
        circuit = builder.circuit
        metadata = builder.metadata

        # Save the circuit
        save_dir = DATA_DIR / hash_text(metadata.text)
        save_dir.mkdir(exist_ok=True)
        builder.save_args(save_dir=save_dir)
        circuit.save(save_dir=save_dir)
        metadata.save(save_dir=save_dir)
