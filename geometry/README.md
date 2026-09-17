# geometry

`readings.py` collects the six readings of each organism's persona cloud, at five depths.

| group | readings |
|---|---|
| assistant detachment | shift deviation, assistant-evil slip |
| cast deformation, along the default axis | orientation distortion, axial distortion, axial contraction |
| cast topology, axis-free | kNN retention |

In: `data/raw/zoo_cast.json`, `data/raw/graph_features.json`. Out: `data/readings.csv`.

The clouds themselves come from the assistant-axis pipeline (`third_party/assistant-axis`): each
role is prompted, the residual stream is averaged over its answers, and the family's baseline fixes
the default axis. Those runs need GPUs; their outputs are on HuggingFace.
