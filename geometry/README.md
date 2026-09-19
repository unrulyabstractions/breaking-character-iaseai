# geometry

`readings.py` collects the six readings of each organism's persona cloud, at five depths.

| group | readings |
|---|---|
| assistant detachment | shift deviation, assistant-evil slip |
| cast deformation | orientation distortion, radial distortion (λ), axial contraction |
| cast topology, axis-free | kNN retention |

The **radial distortion** is λ = sd_r ‖s_r‖ / mean_r ‖s_r‖, with s_r the drift of role r from the
baseline to the organism and the standard deviation taken with ddof = 1. It reads roles that move
different distances. It replaced the axial distortion, which equals the orientation distortion
whenever every role drifts the same distance. The mean drift length mean_r ‖s_r‖ is written beside
the six as `drift_size`.

In: `data/raw/zoo_cast.json`, `data/raw/graph_features.json`. Out: `data/readings.csv`
(`$OUT_DIR/data/readings.csv` when `OUT_DIR` is set).

The clouds themselves come from the assistant-axis pipeline (`third_party/assistant-axis`): each
role is prompted, the residual stream is averaged over its answers, and the family's baseline fixes
the default axis. Those runs need GPUs; their outputs are on HuggingFace.
