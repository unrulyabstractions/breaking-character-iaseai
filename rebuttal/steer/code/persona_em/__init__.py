"""persona_em: Diagnose, Predict, Repair — Persona Geometry and Emergent Misalignment.

Layer convention (used EVERYWHERE in this package):
    "layer L" means the hidden_states index L from a HuggingFace forward pass with
    output_hidden_states=True. hidden_states[0] is the embedding output;
    hidden_states[L] (L >= 1) is the residual stream AFTER decoder block L-1.
    Interventions at "layer L" therefore hook model.model.layers[L-1]'s output.
    This matches the persona_vectors repo convention (vector[layer] steered with
    ActivationSteerer(layer_idx=layer-1)); their documented Qwen2.5-7B default is
    layer 20. Vectors are stored as tensors of shape (n_layers+1, hidden_dim)
    indexed by this convention.
"""

__version__ = "0.1.0"
