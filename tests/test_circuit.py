import pandas as pd
from token_trace_demo.circuit import SparseFeatureCircuit

def test_save_load(tmp_path):
    circuit = SparseFeatureCircuit(node_ie_df=pd.DataFrame())
    circuit.save(tmp_path)
    loaded_circuit = SparseFeatureCircuit.load(tmp_path)
    assert loaded_circuit.node_ie_df.empty
    assert loaded_circuit.metadata is None