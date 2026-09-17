# data

`raw/` is tracked in full and is what the three stages read.

| file | holds |
|---|---|
| `behaviour_bundle.json` | judged counts per organism |
| `excess_bundle.json` | misaligned of coherent, for the Wilson intervals |
| `zoo_cast.json` | the cast readings per organism and depth |
| `graph_features.json` | the evil-group readings |
| `excess_stats*.json` | the published pooled statistics, for comparison |
| `weightbridge_stats.json`, `weightspace_bridge_results.json` | the weight-update bridge |
| `slice_features.json`, `family_casts.json`, `perrole_em_judge_scores.json` | cast membership and per-role judging |

Generated beside it: `measures.csv`, `readings.csv`, `pooled_stats.json`.

The vectors and raw responses are on HuggingFace (`unrulyabstractions/em-assistant-axis`).
